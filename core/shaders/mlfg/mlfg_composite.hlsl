// FSR FG4 network pass 190: the full-resolution composite of the generated frame (AMD FidelityFX SDK v2.4.0, MIT;
// spec: tools/ffx_extract/mlfg_io.py pass190_composite). Built per output format: MLFG_OUT_RGBA8 / _RGB10A2 / _RGBA16F.
//
// Reads   t1 weights   kMlfgSoftmax5's [H][W][8] fp16 buffer (ch 0-4), live size tensorSizes[90] (TW x TH, 1/4 res)
//         t2 rgb0, t3 rgb1     previous / current colour, colourSize (= the output's size)
//         t4 flow0toT, t5 flow1toT  kMlfgFlowOut's UV images, flowSize
//         t6 key0, t7 word0    splat0toT;  t8 key1, t9 word1  splat1toT (as kMlfgStage2Input), splatSize
//         t10 hudless          HUD-less colour (read only when hudlessAttachedFactor != 0; bind any colour otherwise)
// Writes  u11 output   colourSize, in the colour's own encoding (raw colours, no transfer function), alpha 0;
//                      quantised in the shader to the format's codes (round half up, as mlfg_io.to_unorm8) so the
//                      typed store's own float conversion never decides a value.
// Per pixel (uv = (p + 0.5) * fp32(1 / colourSize)), with s0/s1/flow0/flow1/m0/m1 as in kMlfgStage2Input but at the
// full-res pixel centre, and w = fp16 bilinear upsample (align corners off, clamped) of the five weights:
//   A = rgb0(uv - s0) * (1 - m1) + rgb1(uv - s1) * m1     (frame-0 backward warp, holes filled from frame 1)
//   B = rgb1(uv - s1) * (1 - m0) + rgb0(uv - s0) * m0     (m0 / m1 swapped relative to A's sources, as AMD)
//   out = A*w0 + B*w1 + rgb0(uv + flow0)*w2 + rgb1(uv + flow1)*w3 + rgb0[p] * fp16((1-t)*w4) + rgb1[p] * fp16(t*w4)
// Reset or outside the generation rect: out = rgb1[p]. Then the optional HUD-less UI restore and the debug bars
// (dispatchFlags bits 0 / 1).
// Timestep: AMD's last term is (rgb0[p] + rgb1[p]) * fp16(0.5 * w4): w4 is the weight of the plain blend of the two
// un-warped frames, which at the midpoint is the average. For a frame at t the blend is (1 - t) * rgb0 + t * rgb1, so
// the term becomes rgb0 * fp16((1 - t) * w4) + rgb1 * fp16(t * w4): the same bits as AMD's at t = 0.5. The warps
// need no change: the splats are projected to t (projAmount) and the flows are the network's for t.
// SPDX-License-Identifier: MIT
#include "mlfg_sample.hlsli"

cbuffer Constants : register(b0)
{
    int4   c_colour;   // colourSize.xy, flowSize.xy
    int4   c_splat;    // splatSize.xy, weights width (row pitch), 0
    int4   c_weights;  // weight live size tensorSizes[90].xy, reset, hudlessAttachedFactor
    int4   c_misc;     // backBufferTransferFunction, dispatchFlags, 0, NFRUX: asuint(0.5 * splatSize.x / colourSize.x) (with c_lum.w)
    float4 c_genRect;  // generationRectMinUV.xy, generationRectMaxUV.xy
    float4 c_lum;      // minMaxLuminance.xy, timestep t (AMD: 0.5), NFRUX: 1 / colourSize.x (0: computed here)
    float4 c_debug;    // debugBarColor.rgb, NFRUX: 1 / colourSize.y
};
ByteAddressBuffer t_weights : register(t1);
Texture2D<float4> t_rgb0 : register(t2);
Texture2D<float4> t_rgb1 : register(t3);
Texture2D<float4> t_flow0 : register(t4);
Texture2D<float4> t_flow1 : register(t5);
Texture2D<uint>   t_key0 : register(t6);
Texture2D<uint>   t_word0 : register(t7);
Texture2D<uint>   t_key1 : register(t8);
Texture2D<uint>   t_word1 : register(t9);
Texture2D<float4> t_hudless : register(t10);
#if defined(MLFG_OUT_RGBA8)
[[vk::image_format("rgba8")]] RWTexture2D<float4> u_out : register(u11);
#elif defined(MLFG_OUT_RGB10A2)
[[vk::image_format("rgb10a2")]] RWTexture2D<float4> u_out : register(u11);
#else
[[vk::image_format("rgba16f")]] RWTexture2D<float4> u_out : register(u11);
#endif

float3 Read(Texture2D<float4> t, float2 uv) { return SampleG(t, uv, c_colour.xy).rgb; }  // AMD's nearest sample
float2 Clamp(float2 q) { return min(max(q, c_genRect.xy), c_genRect.zw); }

float3 SrgbToLinear(float3 c)  // AMD tests the encoded value against the linear-side threshold
{
    float3 r;
    [loop] for (uint k = 0u; k < 3u; ++k)
        r[k] = c[k] < 0.00313080498017371 ? c[k] * 0.0773993805050850 : FPow(c[k] * 0.947867274284363 + 0.0521326996386051, 2.4);
    return r;
}
float3 ToLinear(float3 c)
{
    const uint tf = uint(c_misc.x);
    if (tf == 0u)
        return SrgbToLinear(c);
    if (tf == 1u)
    {
        const float3 p = FPow(c, 0.0126833003014326);
        return FPow(FDiv(clamp(p - 0.835937976837158, 0.0, 1.0), 18.8516006469727 - 18.6875 * p), 6.27739000320435) * FDiv(10000.0, c_lum.y);
    }
    const float lo = c_lum.x * 0.0125, range = (c_lum.y - c_lum.x) * 0.0125;
    return FDiv(c - lo, float3(range, range, range));
}

// The code k is stored as (k + 0.25) / max: both APIs allow the float -> unorm conversion to pick either neighbouring
// integer, and k + 0.25 has k as both its nearest and its truncation, so every implementation stores k.
float4 Quantise(float3 c)
{
#if defined(MLFG_OUT_RGBA8)
    return float4((floor(clamp(c, 0.0, 1.0) * 255.0 + 0.5) + 0.25) * (1.0 / 255.0), 0.0);
#elif defined(MLFG_OUT_RGB10A2)
    return float4((floor(clamp(c, 0.0, 1.0) * 1023.0 + 0.5) + 0.25) * (1.0 / 1023.0), 0.0);
#else
    return float4(F16Round(c), 0.0);
#endif
}

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int2 g = int2(id.xy);
    const int2 size = c_colour.xy;
    if (any(g >= size))
        return;
    // The uniform quotients 1 / size and 0.5 * splatSize.x / size.x: from the host (correctly rounded there, c_lum.w,
    // c_debug.w, c_misc.w), or computed here when not given (0).
    const bool   given = c_lum.w != 0.0;
    const float2 rcp   = given ? float2(c_lum.w, c_debug.w) : FDiv(float2(1, 1), float2(size));
    const float  so    = given ? asfloat(uint(c_misc.w)) : SplatFlowOffset(c_splat.xy, size.x);
    const float2 uv = TexelUv(g, rcp);
    const bool   outRect = any(uv > c_genRect.zw) || any(uv < c_genRect.xy);
    const float3 p0 = t_rgb0.Load(int3(g, 0)).rgb, p1 = t_rgb1.Load(int3(g, 0)).rgb;
    float3       o  = p1;
    if (!outRect && c_weights.z == 0)
    {
        // fp16 bilinear upsample of the weights (x4, align corners off, clamped to the live size).
        int   x0, x1, y0, y1;
        float fx, fy;
        ResampleAxis(g.x, c_weights.x, 0.25, x0, x1, fx);
        ResampleAxis(g.y, c_weights.y, 0.25, y0, y1, fy);
        const uint ww = uint(c_splat.z);
        const uint a = TexelBase(int2(x0, y0), ww, 8u), b = TexelBase(int2(x1, y0), ww, 8u);
        const uint c = TexelBase(int2(x0, y1), ww, 8u), d = TexelBase(int2(x1, y1), ww, 8u);
        // A texel's 8 halves are one 16-byte load (ch 0-4 the weights).
        const uint4 wa = t_weights.Load4(a << 1), wb = t_weights.Load4(b << 1), wc = t_weights.Load4(c << 1), wd = t_weights.Load4(d << 1);
        float       w[5];
        [unroll] for (uint k = 0u; k < 5u; ++k)
        {
            const uint sh = (k & 1u) * 16u;
            w[k] = Bilerp16(F16ToF32((wa[k >> 1] >> sh) & 0xFFFFu), F16ToF32((wb[k >> 1] >> sh) & 0xFFFFu), F16ToF32((wc[k >> 1] >> sh) & 0xFFFFu),
                            F16ToF32((wd[k >> 1] >> sh) & 0xFFFFu), fx, fy);
        }
        const int2   ss = c_splat.xy;
        const float  m1 = SplatCoverage(t_key1, t_word1, uv, ss), m0 = SplatCoverage(t_key0, t_word0, uv, ss);
        const float2 s0 = SplatFlow(t_word0, uv, ss, so), s1 = SplatFlow(t_word1, uv, ss, so);
        const float2 f0 = F16Round(SampleG(t_flow0, uv, c_colour.zw).xy), f1 = F16Round(SampleG(t_flow1, uv, c_colour.zw).xy);
        // The four colour reads, one copy of the code: rgb0 at uv - s0, uv + f0; rgb1 at uv - s1, uv + f1.
        const float2 at[4] = {Clamp(uv - s0), Clamp(uv + f0), Clamp(uv - s1), Clamp(uv + f1)};
        float3       col[4];
        [unroll] for (uint r = 0u; r < 4u; ++r)
            col[r] = r < 2u ? Read(t_rgb0, at[r]) : Read(t_rgb1, at[r]);
        const float3 c0 = col[0], c1 = col[2];
        // Was fp16(0.5 * w4) on both frames: the plain blend at the timestep t.
        const float  t   = c_lum.z;
        const float  w40 = F16Round((1.0 - t) * w[4]), w41 = F16Round(t * w[4]);
        o = (((((c0 * (1.0 - m1) + c1 * m1) * w[0] + (c1 * (1.0 - m0) + c0 * m0) * w[1]) + col[1] * w[2]) + col[3] * w[3]) + p0 * w40) + p1 * w41;
    }
    if (c_weights.w != 0)
    {
        const float3 hl = t_hudless.Load(int3(g, 0)).rgb;
        float3 lin[2] = {p1, hl};
        [loop] for (uint i = 0u; i < 2u; ++i)
            lin[i] = ToLinear(lin[i]);
        const float3 mx = max(lin[0], lin[1]), mn = min(lin[0], lin[1]);
        float        k = 0.0;
        [unroll] for (uint j = 0u; j < 3u; ++j)
        {
            const float ratio = mx[j] != 0.0 ? FDiv(mn[j], mx[j]) : 0.0;
            k = max(k, clamp((1.0 - ratio) * 10.0, 0.0, 1.0));
        }
        const float3 ui0 = k > 0.001 ? Mix(o, hl, k) : o;
        const bool   differs = any(abs(p1 - hl) > 0.0);
        o = outRect ? hl : (differs ? ui0 : o);
    }
    if ((c_misc.y & 1) != 0)
    {
        if (g.x < 16)
            o.g = 1.0;
        if (g.x >= size.x - 15)
            o += c_debug.rgb;
    }
    if ((c_misc.y & 2) != 0 && c_weights.z == 1 && g.y < int(max(32.0, float(size.x) * 0.015)))
        o.r = 1.0;
    u_out[g] = Quantise(o);
}
