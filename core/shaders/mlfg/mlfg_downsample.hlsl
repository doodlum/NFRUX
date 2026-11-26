// FSR FG4 network pass 18: 4x bilinear downsample of the 18-channel network input to the coarsest flow level
// (AMD FidelityFX SDK v2.4.0, MIT; spec: tools/ffx_extract/mlfg_glue.py pass18).
//
// Reads   t1 net_in  [srcH][srcW][32] fp16 (kMlfgNetInput's tensor after the encoder filled ch 6-17), clamp size S1
// Writes  u2 in_q    [dstH][dstW][32] fp16: ch 0-17 the downsampled input, ch 18-31 written 0; zero outside the live
//                    size S10. AMD: Tensor4 120x72x18 (tensorSizes[10]); its pass 20 reads a 32-channel window
//                    (ch 0-31) of it, see the note in mlfg_constants.h about AMD's overlapping reads.
// out(o) = bilinear at p = max((o + 0.5) * 4 - 0.5, 0) = 4o + 1.5, i.e. the 2x2 texels 4o+1..4o+2 (clamped to S1 - 1)
// with fp16 roundings after each lerp (mlfg_glue "glsl" mode).
// Size rule: S1 = tensorSizes[1] (the level-0 size), S10 = tensorSizes[10] (S1 / 4).
// No timestep here.
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

cbuffer Constants : register(b0)
{
    int4   c_src;    // source width (row pitch in texels), clamp size S1.xy, 0
    int4   c_dst;    // destination width, height (the buffer's extent), live size S10.xy
    float4 c_scale;  // resample scale (4), 0, 0, 0
};
ByteAddressBuffer   t_src : register(t1);
RWByteAddressBuffer u_dst : register(u2);

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int2 o = int2(id.xy);
    if (any(o >= c_dst.xy))
        return;
    const uint base = TexelBase(o, uint(c_dst.x), 32u);
    if (any(o >= c_dst.zw))
    {
        [unroll] for (uint c = 0u; c < 32u; c += 2u)
            u_dst.Store((base + c) << 1, 0u);
        return;
    }
    int   x0, x1, y0, y1;
    float fx, fy;
    ResampleAxis(o.x, c_src.y, c_scale.x, x0, x1, fx);
    ResampleAxis(o.y, c_src.z, c_scale.x, y0, y1, fy);
    const uint w = uint(c_src.x);
    const uint a = TexelBase(int2(x0, y0), w, 32u), b = TexelBase(int2(x1, y0), w, 32u);
    const uint c = TexelBase(int2(x0, y1), w, 32u), d = TexelBase(int2(x1, y1), w, 32u);
    [loop] for (uint k = 0u; k < 18u; k += 2u)
    {
        const float2 ta = LoadHalf2(t_src, a + k), tb = LoadHalf2(t_src, b + k);
        const float2 tc = LoadHalf2(t_src, c + k), td = LoadHalf2(t_src, d + k);
        StoreHalf2(u_dst, base + k, Bilerp16(ta.x, tb.x, tc.x, td.x, fx, fy), Bilerp16(ta.y, tb.y, tc.y, td.y, fx, fy));
    }
    [unroll] for (uint z = 18u; z < 32u; z += 2u)
        u_dst.Store((base + z) << 1, 0u);
}
