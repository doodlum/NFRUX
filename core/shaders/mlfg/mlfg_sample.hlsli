// Image reads of the FSR FG4 passes 0, 152 and 190 (AMD FidelityFX SDK v2.4.0, MIT; spec: tools/ffx_extract/mlfg_io.py
// sample_point / splat_valid_weight / splat_flow_at, mlfg_glue.py _nearest_index).
//
// AMD reads colours and flow images through its binding-320 sampler. Despite the shader name g_sampler_linear, AMD's
// DLL creates every sampler as VK_FILTER_NEAREST (min / mag / mip), CLAMP_TO_EDGE, normalized coordinates, and binds it
// immutably at binding 320 of all its layouts (seen in vkCreateSampler / descriptor-set-layout captures), on every
// vendor. So these reads are point samples. The texel is chosen here in integer code, as mlfg_io.sample_point: the
// normalized coordinate truncated to 21 fractional bits, then floor(uv * size), clamped to the edge. That is the
// capture GPU's (NVIDIA) rule, bit-exact against AMD's captured output; a texture unit is never asked, so the choice
// at exact texel edges (pass 152 lands on one in every column) is the same on every GPU and API.
// SPDX-License-Identifier: MIT
#ifndef MLFG_SAMPLE_HLSLI
#define MLFG_SAMPLE_HLSLI
#include "mlfg_common.hlsli"

// floor(floor(u * 2^21) / 2^21 * n) clamped to [0, n-1], exactly (integers; n <= 8192).
int PointIndex(float u, int n)
{
    const uint k = uint(clamp(floor(u * 2097152.0), 0.0, 2097152.0));
    const uint w = uint(n);
    const int  i = int(((k >> 8) * w + (((k & 255u) * w) >> 8)) >> 13);
    return clamp(i, 0, n - 1);
}

// AMD's textureLod(sampler2D(t, g_sampler_linear), uv, 0): a nearest, clamp-to-edge sample.
float4 SampleG(Texture2D<float4> t, float2 uv, int2 size)
{
    return t.Load(int3(PointIndex(uv.x, size.x), PointIndex(uv.y, size.y), 0));
}

// Bilinear coverage of a splat at uv: texels count when written (key >= 1) with a valid flow (word != INVALID2).
// Taps as textureGather (clamp to edge), weights fract(uv * size + 0.5), as AMD's shader computes them.
float SplatCoverage(Texture2D<uint> key, Texture2D<uint> flow, float2 uv, int2 size)
{
    const float2 xs = uv * float2(size);
    const int2   i0 = TruncI(floor(xs - 0.5));
    const int    x0 = clamp(i0.x, 0, size.x - 1), x1 = clamp(i0.x + 1, 0, size.x - 1);
    const int    y0 = clamp(i0.y, 0, size.y - 1), y1 = clamp(i0.y + 1, 0, size.y - 1);
#define MLFG_COVER(X, Y) ((key.Load(int3(X, Y, 0)) >= 1u && flow.Load(int3(X, Y, 0)) != INVALID2) ? 1.0 : 0.0)
    const float g0 = MLFG_COVER(x0, y1), g1 = MLFG_COVER(x1, y1), g2 = MLFG_COVER(x1, y0), g3 = MLFG_COVER(x0, y0);
#undef MLFG_COVER
    const float fx = Frac(xs.x + 0.5), fy = Frac(xs.y + 0.5);
    return Mix(Mix(g3, g2, fx), Mix(g0, g1, fx), fy);
}

// The single-texel splat flow read: texel ivec2(uv * size - 0.5 * size.x / refWidth) (both axes use size.x),
// 0 outside the image, flow (0, 0) when the texel's x is 65504 (no flow). Keys are not consulted, as in AMD's shader.
// o = 0.5 * size.x / refWidth (SplatFlowOffset; uniform, so the passes compute it once a group).
float SplatFlowOffset(int2 size, int refWidth) { return 0.5 * FDiv(float(size.x), float(refWidth)); }
float2 SplatFlow(Texture2D<uint> flow, float2 uv, int2 size, float o)
{
    const int2   p = TruncI(uv * float2(size) - o);
    const float2 f = UnpackF16x2(Inside(p, size) ? flow.Load(int3(p, 0)) : 0u);
    return f.x != F16_MAX ? f : float2(0, 0);
}

#endif
