// FSR FG4 network pass 152: the stage-2 network input, 24 channels at 1/4 of the display
// (AMD FidelityFX SDK v2.4.0, MIT; spec: tools/ffx_extract/mlfg_io.py pass152_stage2_input).
//
// Reads   t1 rgb0, t2 rgb1      previous / current colour (any float4 texture), colourSize
//         t3 flow0toT, t4 flow1toT  kMlfgFlowOut's UV images (RG16F), flowSize
//         t5 key0, t6 word0      splat0toT (kMlfgSplatKey / kMlfgSplatFlow, proj t, flowIsUv 1), splatSize
//         t7 key1, t8 word1      splat1toT (proj 1 - t), splatSize
// Writes  u9 s2in  [outH][outW][32] fp16 (NHWC, pitch 32; AMD: Tensor4 960x540x24 for the 4K network, of which
//                  tensorSizes[72] is live); ch 24-31 and everything outside the live size written 0. The network reads
//                  a 32-channel window (AMD pass 154), see mlfg_constants.h about AMD's overlapping reads.
// Live size tensorSizes[72] (480x272 at 1080p: uv uses it, so rows sample (y + 0.5) / 272 of the colour); texels
// outside colourSize / 4 or the generation rect are 0. uv = (p + 0.5) * fp32(1 / size).
//   0-2  rgb0 at uv - s0        (s0 = splat0toT flow at the texel, 0 if none: backward-warped frame 0)
//   3-5  rgb1 at uv - s1        (s1 = splat1toT)
//   6-8  rgb0 at uv + flow0     (flow0 = flow0toT sampled at uv)
//   9-11 rgb1 at uv + flow1
//   12   coverage of splat0toT, 13 coverage of splat1toT (bilinear over valid texels)
//   14-16 rgb0 at uv, 17-19 rgb1 at uv
//   20 |Y(14-16) - Y(17-19)|, 21 |Y(0-2) - Y(3-5)|, 22 |Y(0-2) - Y(6-8)|, 23 |Y(6-8) - Y(9-11)|   (Y = BT.601 luma)
// Colour and flow reads are AMD's nearest samples (mlfg_sample.hlsli) at the generation-rect clamped uv, colour in
// the network colour space
// (transfer function 2 converted to PQ; 0 and 1 pass through).
// Timestep: none in the arithmetic; the splats it reads are projected to t by the host's projAmount and the flows
// are the network's for t (renamed from AMD's 0to05 / 1to05).
// SPDX-License-Identifier: MIT
#include "mlfg_sample.hlsli"

cbuffer Constants : register(b0)
{
    int4   c_out;      // output width, height (the buffer's extent), live size tensorSizes[72].xy
    int4   c_colour;   // colourSize.xy, flowSize.xy
    int4   c_splat;    // splatSize.xy, backBufferTransferFunction, 0
    float4 c_genRect;  // generationRectMinUV.xy, generationRectMaxUV.xy
    float4 c_lum;      // minMaxLuminance.xy, 0, 0
};
Texture2D<float4>   t_rgb0 : register(t1);
Texture2D<float4>   t_rgb1 : register(t2);
Texture2D<float4>   t_flow0 : register(t3);
Texture2D<float4>   t_flow1 : register(t4);
Texture2D<uint>     t_key0 : register(t5);
Texture2D<uint>     t_word0 : register(t6);
Texture2D<uint>     t_key1 : register(t7);
Texture2D<uint>     t_word1 : register(t8);
RWByteAddressBuffer u_out : register(u9);

// scRGB -> PQ (mlfg_io.to_pq, fp32), products summed left to right.
float3 ToPq(float3 c)
{
    const float  lo = c_lum.x * 0.0125, range = (c_lum.y - c_lum.x) * 0.0125;
    const float3 x  = max(FDiv(c - lo, float3(range, range, range)), 0.0);
    const float3 r0 = float3(0.627403974533081, 0.329281985759735, 0.0433136001229286);
    const float3 r1 = float3(0.0690969973802567, 0.919541001319885, 0.0113612003624439);
    const float3 r2 = float3(0.0163915995508432, 0.0880132019519806, 0.895595014095306);
    const float3 m  = (r0 * x.x + r1 * x.y) + r2 * x.z;  // x @ M
    const float3 y  = FPow((m * 300.0) * 1e-4, 0.1593019962310791);
    return FPow(FDiv(0.835937976837158 + y * 18.8516006469727, 1.0 + y * 18.6875), 78.8438034057617);
}

float3 Read(Texture2D<float4> t, float2 uv)
{
    const float3 c = SampleG(t, uv, c_colour.xy).rgb;
    return c_splat.z == 2 ? ToPq(c) : c;
}
float2 Clamp(float2 q) { return min(max(q, c_genRect.xy), c_genRect.zw); }
float  Luma(float3 c) { return (c.r * 0.298999994993210 + c.g * 0.587000012397766) + c.b * 0.114000000059605; }

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int2 g = int2(id.xy);
    if (any(g >= c_out.xy))
        return;
    const uint base = TexelBase(g, uint(c_out.x), 32u);
    float      ch[24];
    [unroll] for (uint z = 0u; z < 24u; ++z)
        ch[z] = 0.0;
    const float2 uv = TexelUv(g, FDiv(float2(1, 1), float2(c_out.zw)));
    const bool   outRect = any(uv > c_genRect.zw) || any(uv < c_genRect.xy) || any(g >= c_colour.xy / 4);
    if (all(g < c_out.zw) && !outRect)
    {
        const int2   ss  = c_splat.xy;
        const float2 s0  = SplatFlow(t_word0, uv, ss, c_out.z), s1 = SplatFlow(t_word1, uv, ss, c_out.z);
        const float2 f0  = F16Round(SampleG(t_flow0, uv, c_colour.zw).xy), f1 = F16Round(SampleG(t_flow1, uv, c_colour.zw).xy);
        // The six colour reads, one copy of the code: rgb0 at uv - s0, uv + f0, uv; rgb1 at uv - s1, uv + f1, uv.
        const float2 at[6] = {Clamp(uv - s0), Clamp(uv + f0), uv, Clamp(uv - s1), Clamp(uv + f1), uv};
        float3       col[6];
        [loop] for (uint r = 0u; r < 6u; ++r)
            col[r] = r < 3u ? Read(t_rgb0, at[r]) : Read(t_rgb1, at[r]);
        const float3 c0 = col[0], c6 = col[1], c14 = col[2], c3 = col[3], c9 = col[4], c17 = col[5];
        const float  m0  = SplatCoverage(t_key0, t_word0, uv, ss), m1 = SplatCoverage(t_key1, t_word1, uv, ss);
        const float  y0 = Luma(c0), y6 = Luma(c6);
        const float3 v[4] = {c0, c3, c6, c9};
        [unroll] for (uint k = 0u; k < 4u; ++k)
        {
            ch[3u * k]      = v[k].r;
            ch[3u * k + 1u] = v[k].g;
            ch[3u * k + 2u] = v[k].b;
        }
        ch[12] = m0;
        ch[13] = m1;
        ch[14] = c14.r; ch[15] = c14.g; ch[16] = c14.b;
        ch[17] = c17.r; ch[18] = c17.g; ch[19] = c17.b;
        ch[20] = abs(Luma(c14) - Luma(c17));
        ch[21] = abs(y0 - Luma(c3));
        ch[22] = abs(y0 - y6);
        ch[23] = abs(y6 - Luma(c9));
    }
    [loop] for (uint k = 0u; k < 24u; k += 2u)
        StoreHalf2(u_out, base + k, ch[k], ch[k + 1u]);
    [unroll] for (uint z2 = 24u; z2 < 32u; z2 += 2u)
        u_out.Store((base + z2) << 1, 0u);
}
