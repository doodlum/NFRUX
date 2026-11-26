// FSR FG4 network passes 54 / 83 / 112: coarse-to-fine flow accumulation, backward warps of both frames and the
// 32-channel stacks the next decoder reads (AMD FidelityFX SDK v2.4.0, MIT; spec: tools/ffx_extract/mlfg_glue.py
// pass54 / pass83 / pass112). One shader, c_mode selects the pass.
//
// Reads   t1 net_in  [netH][netW][32] fp16 (kMlfgNetInput + encoder features): rgb0 0-2, rgb1 3-5, feat0 6-11,
//                    feat1 12-17; warps clamp to the size S (S1 for 54/83, S53 for 112)
//         t2 head    decoder head [headH][headW][16] fp16 (from the network; ch 0-3 flow residual, 4-12 passed on):
//                    54: f54 at tensorSizes[23] (level 2), 83: f83 at tensorSizes[38] (level 1),
//                    112: f112 at tensorSizes[54] (level 0)
//         t3 prev    83 / 112: the previous flow [H][W][4] fp16 (54's / 83's u5), level-0 pixels
// Writes  u4 stack   [outH][outW][32] fp16, zero outside the live size So:
//                    54: x54 at tensorSizes[24] (level 1), 83: x83 at tensorSizes[39], 112: x112 at tensorSizes[54]
//         u5 flow    [flowH][flowW][4] fp16 (A.x, A.y, B.x, B.y) in level-0 pixels:
//                    54: flow54 at 2 * tensorSizes[24], 83: flow83 at tensorSizes[39], 112: flow112 at tensorSizes[54]
// Flow A (ch 0-1) warps frame 0 (rgb0 + feat0), flow B (ch 2-3) frame 1; backward warp: sample at p + flow (p integer,
// no +0.5) as ((p + f) * 2 / (S-1)) * 0.5 * (S-1) in fp32, bilinear clamped to [0, S-1].
//   54:  up = bilinear_up4(f54) (scale 0.25), flow = 4 * up[0:4]; stack computed at level 0 and box-averaged 2x2:
//        fp16 sums ((s00 + s10) + s01) + s11, / 4, flow channels / 2 more (level-1 pixels), ch 18 = t
//   83:  up = bilinear_up2(f83) (scale 0.5), flow = 2 * up[0:4] + flow54 (fp16 product then fp16 sum)
//   112: flow = f112[0:4] + flow83 (fp16), head channels copied
// Stack channels: 0-2 warped rgb0 | 3-5 warped rgb1 | 6-11 warped feat0 | 12-17 warped feat1 | 18 timestep t |
//                 19-27 head ch 4-12 | 28-31 flow. The network reads 32-channel windows of it (AMD passes 57 / 86 / 115).
// Timestep: channel 18 is AMD's constant 0.5, the position of the generated frame between frame 0 and frame 1;
// here c_params.z = t (stored as fp16(t)). With t = 0.5 it is AMD's value. The flows themselves are network outputs:
// A = generated frame -> frame 0, B = generated frame -> frame 1, at whatever t the network was told.
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

cbuffer Constants : register(b0)
{
    int4   c_mode;    // 0 = pass 54, 1 = pass 83, 2 = pass 112
    int4   c_net;     // net_in width (row pitch in texels), warp clamp size S.xy, 0
    int4   c_head;    // head width (row pitch), head size (resample clamp).xy, 0
    int4   c_prev;    // previous flow width (row pitch), 0, 0, 0
    int4   c_out;     // stack width, height (the buffer's extent), live size So.xy
    int4   c_flow;    // flow width (row pitch), 0, 0, 0
    float4 c_params;  // resample scale (0.25 / 0.5 / unused), flow scale (4 / 2 / 1), timestep t, 0
};
ByteAddressBuffer   t_net : register(t1);
ByteAddressBuffer   t_head : register(t2);
ByteAddressBuffer   t_prev : register(t3);
RWByteAddressBuffer u_stack : register(u4);
RWByteAddressBuffer u_flow : register(u5);

// Backward warp of nine channels (three colour + six features) at level-0 pixel g by flow f.
void Warp(int2 g, float2 f, uint colour, uint feat, out float o[9])
{
    const float2 S   = float2(c_net.yz - 1);
    const float2 pos = (FDiv((float2(g) + f) * 2.0, S) * 0.5) * S;
    const float2 fl  = floor(pos);
    const float2 fr  = pos - fl;
    const int2   i   = TruncI(fl);
    const int    x0 = clamp(i.x, 0, c_net.y - 1), x1 = clamp(i.x + 1, 0, c_net.y - 1);
    const int    y0 = clamp(i.y, 0, c_net.z - 1), y1 = clamp(i.y + 1, 0, c_net.z - 1);
    const uint   w = uint(c_net.x);
    const uint   a = TexelBase(int2(x0, y0), w, 32u), b = TexelBase(int2(x1, y0), w, 32u);
    const uint   c = TexelBase(int2(x0, y1), w, 32u), d = TexelBase(int2(x1, y1), w, 32u);
    [unroll] for (uint k = 0u; k < 9u; ++k)
    {
        const uint ch = k < 3u ? colour + k : feat + (k - 3u);
        o[k] = Bilerp16(LoadHalf(t_net, a + ch), LoadHalf(t_net, b + ch), LoadHalf(t_net, c + ch), LoadHalf(t_net, d + ch), fr.x, fr.y);
    }
}

// The 32-channel stack and the flow at level-0 pixel g.
void Stack(int2 g, out float st[32], out float4 flow)
{
    float hd[13];
    const int mode = c_mode.x;
    if (mode == 2)
    {
        const uint hb = TexelBase(g, uint(c_head.x), 16u);
        [unroll] for (uint k = 0u; k < 13u; ++k)
            hd[k] = LoadHalf(t_head, hb + k);
    }
    else
    {
        int   x0, x1, y0, y1;
        float fx, fy;
        ResampleAxis(g.x, c_head.y, c_params.x, x0, x1, fx);
        ResampleAxis(g.y, c_head.z, c_params.x, y0, y1, fy);
        const uint w = uint(c_head.x);
        const uint a = TexelBase(int2(x0, y0), w, 16u), b = TexelBase(int2(x1, y0), w, 16u);
        const uint c = TexelBase(int2(x0, y1), w, 16u), d = TexelBase(int2(x1, y1), w, 16u);
        [unroll] for (uint k = 0u; k < 13u; ++k)
            hd[k] = Bilerp16(LoadHalf(t_head, a + k), LoadHalf(t_head, b + k), LoadHalf(t_head, c + k), LoadHalf(t_head, d + k), fx, fy);
    }
    const float4 h4 = float4(hd[0], hd[1], hd[2], hd[3]);
    if (mode == 0)
        flow = F16Round(h4 * c_params.y);
    else
    {
        const uint   pb   = TexelBase(g, uint(c_prev.x), 4u);
        const float4 prev = float4(LoadHalf2(t_prev, pb), LoadHalf2(t_prev, pb + 2u));
        flow = F16Round((mode == 1 ? F16Round(h4 * c_params.y) : h4) + prev);
    }
    float wa[9], wb[9];
    Warp(g, flow.xy, 0u, 6u, wa);
    Warp(g, flow.zw, 3u, 12u, wb);
    [unroll] for (uint k = 0u; k < 3u; ++k)
    {
        st[k]      = wa[k];
        st[3u + k] = wb[k];
    }
    [unroll] for (uint j = 0u; j < 6u; ++j)
    {
        st[6u + j]  = wa[3u + j];
        st[12u + j] = wb[3u + j];
    }
    st[18] = F16Round(c_params.z);  // was AMD's constant 0.5: the timestep t
    [unroll] for (uint m = 0u; m < 9u; ++m)
        st[19u + m] = hd[4u + m];
    st[28] = flow.x;
    st[29] = flow.y;
    st[30] = flow.z;
    st[31] = flow.w;
}

void StoreFlow(int2 g, float4 f)
{
    const uint b = TexelBase(g, uint(c_flow.x), 4u);
    StoreHalf2(u_flow, b, f.x, f.y);
    StoreHalf2(u_flow, b + 2u, f.z, f.w);
}

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int2 o = int2(id.xy);
    if (any(o >= c_out.xy))
        return;
    const uint ob = TexelBase(o, uint(c_out.x), 32u);
    const bool live = all(o < c_out.zw);
    float      s[32];
    if (!live)
    {
        [unroll] for (uint z = 0u; z < 32u; z += 2u)
            u_stack.Store((ob + z) << 1, 0u);
        if (c_mode.x == 0)
        {
            [unroll] for (int q = 0; q < 4; ++q)
                StoreFlow(o * 2 + int2(q & 1, q >> 1), float4(0, 0, 0, 0));
        }
        else
            StoreFlow(o, float4(0, 0, 0, 0));
        return;
    }
    // 83 / 112: the texel itself. 54: level 0 -> level 1, the 2x2 texels in the order (0,0), (1,0), (0,1), (1,1) as
    // (dx, dy), fp16 sums ((s00 + s10) + s01) + s11.
    const bool down = c_mode.x == 0;
    const int  n    = down ? 4 : 1;
    [loop] for (int q = 0; q < n; ++q)
    {
        const int2 g = (down ? o * 2 : o) + int2(q & 1, q >> 1);
        float      t[32];
        float4     flow;
        Stack(g, t, flow);
        StoreFlow(g, flow);
        [unroll] for (uint k = 0u; k < 32u; ++k)
            s[k] = q == 0 ? t[k] : F16Round(s[k] + t[k]);
    }
    if (down)
    {
        [unroll] for (uint k = 0u; k < 32u; ++k)
            s[k] = F16Round(k >= 28u ? (s[k] * 0.25) * 0.5 : s[k] * 0.25);  // AMD's / 4 and / 2: exact either way
        s[18] = F16Round(c_params.z);  // AMD overwrites the averaged constant with 0.5: the timestep t
    }
    [unroll] for (uint k = 0u; k < 32u; k += 2u)
        StoreHalf2(u_stack, ob + k, s[k], s[k + 1u]);
}
