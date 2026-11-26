// FSR FG4 prepass "pipe192": forward splat with nearest-surface-wins (AMD FidelityFX SDK v2.4.0, MIT; spec:
// tools/ffx_extract/mlfg_io.py pipe192_splat). Built as two passes (MLFG_SPLAT_PASS 0 = kMlfgSplatKey, 1 =
// kMlfgSplatFlow), each twice: MLFG_INVERTED_DEPTH 0 / 1.
//
// AMD packs (depth key << 32 | f16 flow.y << 16 | f16 flow.x) and does one 64-bit imageAtomicMax on an R64_UINT image.
// 64-bit image atomics are optional on both APIs (VK_EXT_shader_image_atomic_int64, D3D12
// AtomicInt64OnTypedResourceSupported) and NFRUX's RHI has no 64-bit formats, so the same result is built from two
// 32-bit passes over the same sources:
//   pass 0: key[t]  = atomicMax over all sources landing on t of their 32-bit depth key        (R32_UINT, cleared to 0)
//   pass 1: flow[t] = atomicMax of the packed flow word over the sources whose key == key[t]   (R32_UINT, cleared to 0)
// The 64-bit maximum is lexicographic in (key, flow word), which is exactly this: the largest key wins and, among
// the sources with that key, the largest flow word. Untouched texels stay (key 0, flow 0) as AMD's cleared image.
// Readers use key = AMD's high 32 bits and flow = AMD's low 32 bits (the R32G32_UINT view's .y and .x).
//
// Reads   t1 flow   source flow / mv (RG16F), srcSize; 65504 in .x = no flow
//         t2 depth  source depth (R32F), srcSize
//         t3 key    (pass 1 only) the key image of pass 0
// Writes  u4 key    (pass 0) R32_UINT dstSize, atomics; u5 flow (pass 1) R32_UINT dstSize, atomics
// For each source texel p of dstSize (flow and depth nearest-sampled at the generation-rect-clamped texel centre):
//   f = (flow if flow.x != 65504 else 0) * projAmount * (flowIsUv ? -1 : 1)          (uv units)
//   t = ivec2(p + 0.5 + f*dstSize) (toward zero); the 2x2 texels t + {0,1}^2 inside dstSize receive
//   key = max(asuint(inverted ? |d| : 1 - |d|), 1), d = max(depth, 1e-37) and the word f16(v.y) << 16 | f16(v.x),
//   v = f, or (65504, 65504) when the source had no flow.
// Timestep: projAmount is the fraction of the source's motion to push its pixels along. AMD's three splats per frame
// are proj 1 (frame 1 -> frame 0), 0.5 (frame 0 -> midpoint, flowIsUv 1) and 0.5 (frame 1 -> midpoint). For a
// generated frame at t (0 = previous, 1 = current) the host passes:
//   splat1to0: 1        splat0toT: t (frame-0 pixels move t of the way to frame 1)
//   splat1toT: 1 - t    (frame-1 pixels move back 1 - t of the way to frame 0, since mv(p1) = p0 - p1)
// (mlfg_constants.h SplatProjAmount / SplatSign). With t = 0.5 both are AMD's 0.5.
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

#ifndef MLFG_INVERTED_DEPTH
#define MLFG_INVERTED_DEPTH 0
#endif
#ifndef MLFG_SPLAT_PASS
#define MLFG_SPLAT_PASS 0
#endif

cbuffer Constants : register(b0)
{
    int4   c_sizes;    // dstSize.xy, srcSize.xy (flow / depth; AMD: the same)
    float4 c_proj;     // projAmount, flow sign (-1 when flowIsUv, else 1), 0, 0
    float4 c_genRect;  // generationRectMinUV.xy, generationRectMaxUV.xy
};
Texture2D<float4> t_flow : register(t1);
Texture2D<float4> t_depth : register(t2);
#if MLFG_SPLAT_PASS == 0
[[vk::image_format("r32ui")]] RWTexture2D<uint> u_key : register(u4);
#else
Texture2D<uint> t_key : register(t3);
[[vk::image_format("r32ui")]] RWTexture2D<uint> u_flow : register(u5);
#endif

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int2 p = int2(id.xy);
    const int2 dst = c_sizes.xy, src = c_sizes.zw;
    if (any(p >= dst))
        return;
    const float2 dsz = float2(dst);
    const int2   lo  = TruncI(c_genRect.xy * dsz);
    const int2   hi  = TruncI(c_genRect.zw * dsz) - 1;
    const int2   cp  = clamp(p, lo, hi);
    // Nearest texel of the source at uv = (cp + 0.5) / dstSize: floor(uv * srcSize), exactly, in integers.
    const int2   sp  = clamp(((2 * cp + 1) * src) / (2 * dst), int2(0, 0), src - 1);
    const float2 fl  = t_flow.Load(int3(sp, 0)).xy;
    const bool   valid = fl.x != F16_MAX;
    const float2 f   = ((valid ? fl : float2(0, 0)) * c_proj.x) * c_proj.y;
    const float  d   = abs(max(t_depth.Load(int3(sp, 0)).x, 1e-37));
#if MLFG_INVERTED_DEPTH
    // AMD writes 1 - (1 - |d|); its compiler folds that to |d| (bit-exact against the capture), so |d| it is.
    const uint key = max(asuint(d), 1u);
#else
    const uint key = max(asuint(1.0 - d), 1u);
#endif
    const int2 t = TruncI((0.5 + float2(p)) + f * dsz);
#if MLFG_SPLAT_PASS == 1
    const uint word = valid ? PackF16x2(f.x, f.y) : INVALID2;
#endif
    [unroll] for (int dy = 0; dy < 2; ++dy)
        [unroll] for (int dx = 0; dx < 2; ++dx)
        {
            const int2 q = t + int2(dx, dy);
            if (!Inside(q, dst))
                continue;
#if MLFG_SPLAT_PASS == 0
            InterlockedMax(u_key[q], key);
#else
            if (t_key.Load(int3(q, 0)) == key)
                InterlockedMax(u_flow[q], word);
#endif
        }
}
