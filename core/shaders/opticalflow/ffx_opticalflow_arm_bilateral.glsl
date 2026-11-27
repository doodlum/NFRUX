// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// Arm's joint bilateral filter with its neighbourhoods read from shared memory, the same values
// in the same order (so the same result): the workgroup's 12 x 12 tile of TM1's texels and the median flow, and at the
// last level the hint-warped TP1, each loaded or computed once into shared memory, where each texel's 5x5 filter and
// hint SAD read them 25 times (the hint's warp a bilinear sample and a hint fetch each).
// Ported from Arm's ML Emulation Layer for Vulkan (github.com/arm/ai-ml-emulation-layer-for-vulkan, Apache-2.0).
//
// The median filter (step 3) is taken here too, into shared memory, from the match's flow (a 14 x 14 tile): one dispatch less a level.
//
// FFX_OF_IMAGE=arm (ffx_opticalflow_arm_common.h), per level, step 4 of 4: the layer's bilateral_filter.comp, a 5x5
// filter of the flow weighted by TM1's intensity differences, which the next finer level warps by. At the last level
// (level 2) also the hint (the layer's dense_warp.comp, block_match_of.comp's RAW_SAD and mv_replace.comp): TP1
// warped by the hint flow, the 5x5 SAD of TM1 against that, and where it is lower than the match's the hint is the
// flow; else the filtered flow, in colour pixels (x 4).

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive : enable

#include "ffx_opticalflow_arm_common.h"

#define GROUP 8
#define TILE (GROUP + 2 * ARM_BLOCK_R)  // 12: the group's texels and the 5x5 neighbourhoods' border

layout (local_size_x = GROUP, local_size_y = GROUP, local_size_z = 1) in;

layout(set = 0, binding = 1)        uniform texture2D r_pyramid_tm1;
layout(set = 0, binding = 2)        uniform texture2D r_pyramid_tp1;
layout(set = 0, binding = 3)        uniform texture2D r_arm_sub;    // the match's flow: its 3x3 median (step 3) is taken here
layout(set = 0, binding = 4)        uniform utexture2D r_arm_match;
layout(set = 0, binding = 5)        uniform texture2D r_flow_hint;  // the hint, colour pixels per 4x4 cell
layout(set = 0, binding = 6, rg16f) uniform writeonly image2D rw_arm_flow;
layout(set = 0, binding = 7, rg16f) uniform writeonly image2D rw_flow;  // optical flow resolution

const float16_t kSigmaIntensityDivSqrt = float16_t(5.05129933779);
const float16_t kSpatial               = float16_t(-0.00913);

#define SUB (TILE + 2)  // 14: the medians' 3x3 neighbourhoods
shared float      s_tm1[TILE * TILE];   // TM1's texel, clamped to the level (ArmTexel)
shared f16vec2    s_sub[SUB * SUB];     // the match's flow, clamped to the level
shared f16vec2    s_med[TILE * TILE];   // the median flow, clamped to the level
shared FfxUInt32  s_hint[TILE * TILE];  // the last level: TP1 warped by the hint (zero off the level)

#define SORT(a, b)                     \
    {                                  \
        const f16vec2 lo = min(a, b);  \
        const f16vec2 hi = max(a, b);  \
        a                = lo;         \
        b                = hi;         \
    }

// TP1's level 2 warped by the hint at p (dense_warp.comp), zero off the level as the RAW_SAD reads it.
FfxUInt32 HintWarped(FfxInt32x2 p)
{
    if (any(lessThan(p, FfxInt32x2(0))) || any(greaterThanEqual(p, ubo._LevelDims)))
        return 0u;
    // The hint image is the output's size; past it (the level padded to even) a zero flow.
    const FfxFloat32x2 hint = all(lessThan(p, ubo._HintDims)) ? texelFetch(r_flow_hint, p, 0).xy : FfxFloat32x2(0.0f);
    precise FfxFloat32x2 src = FfxFloat32x2(p) + 0.5f + hint * 0.25f;
    return ArmQuantize(ArmBilinear(r_pyramid_tp1, ubo._LevelRow, ubo._LevelDims, src).x);
}

void main()
{
    const FfxInt32x2 origin = FfxInt32x2(gl_WorkGroupID.xy) * GROUP - ARM_BLOCK_R;
    const bool       last   = ubo._Final != 0;
    const FfxInt32x2 maxPos = ubo._LevelDims - FfxInt32x2(1);
    for (FfxUInt32 i = gl_LocalInvocationIndex; i < SUB * SUB; i += GROUP * GROUP)
        s_sub[i] = f16vec2(texelFetch(r_arm_sub, clamp(origin - 1 + FfxInt32x2(i % SUB, i / SUB), FfxInt32x2(0), maxPos), 0).xy);
    for (FfxUInt32 i = gl_LocalInvocationIndex; i < TILE * TILE; i += GROUP * GROUP)
    {
        const FfxInt32x2 k = origin + FfxInt32x2(i % TILE, i / TILE);
        s_tm1[i]           = ArmTexel(r_pyramid_tm1, ubo._LevelRow, ubo._LevelDims, k);
        if (last)
            s_hint[i] = HintWarped(k);
    }
    barrier();
    // The layer's median_filter.comp (what step 3 wrote and this read back): the 3x3 median of each component at the texel clamped to the level,
    // its neighbours clamped too, by the same sorting network. Each median is one of its texels, so it is that texel's value.
    for (FfxUInt32 i = gl_LocalInvocationIndex; i < TILE * TILE; i += GROUP * GROUP)
    {
        const FfxInt32x2 q = clamp(origin + FfxInt32x2(i % TILE, i / TILE), FfxInt32x2(0), maxPos);
        f16vec2 v[9];
        [[unroll]] for (FfxInt32 j = 0; j < 9; ++j)
        {
            const FfxInt32x2 p = clamp(q + FfxInt32x2(j % 3, j / 3) - 1, FfxInt32x2(0), maxPos) - (origin - 1);
            v[j] = s_sub[p.y * SUB + p.x];
        }
        SORT(v[1], v[2]); SORT(v[4], v[5]); SORT(v[7], v[8]);
        SORT(v[0], v[1]); SORT(v[3], v[4]); SORT(v[6], v[7]);
        SORT(v[1], v[2]); SORT(v[4], v[5]); SORT(v[7], v[8]);
        SORT(v[0], v[3]); SORT(v[5], v[8]); SORT(v[4], v[7]);
        SORT(v[3], v[6]); SORT(v[1], v[4]); SORT(v[2], v[5]);
        SORT(v[4], v[7]); SORT(v[4], v[2]); SORT(v[6], v[4]);
        SORT(v[4], v[2]);
        s_med[i] = v[4];
    }
    barrier();

    const FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, ubo._LevelDims)) || (last && any(greaterThanEqual(pos, ubo._HintDims))))
        return;
    const FfxInt32x2 local = FfxInt32x2(gl_LocalInvocationID.xy);
    precise float16_t centre    = float16_t(s_tm1[(local.y + ARM_BLOCK_R) * TILE + local.x + ARM_BLOCK_R]);
    precise float16_t coeffSum  = float16_t(0.0);
    precise f16vec2   filterSum = f16vec2(0.0);
    [[unroll]] for (FfxInt32 i = 0; i < 25; ++i)
    {
        const FfxInt32   t     = (local.y + i / 5) * TILE + local.x + i % 5;
        precise float16_t diff  = (centre - float16_t(s_tm1[t])) * kSigmaIntensityDivSqrt;
        precise float16_t coeff = clamp(float16_t(1.0) - abs(kSpatial - diff * diff), float16_t(0.0), float16_t(1.0));
        coeffSum += coeff;
        filterSum += coeff * s_med[t];
    }
    precise f16vec2 filtered = filterSum / coeffSum;
    if (!last)
    {
        imageStore(rw_arm_flow, pos, FfxFloat32x4(FfxFloat32x2(filtered), 0.0f, 0.0f));
        return;
    }
    filtered *= f16vec2(4.0);
    FfxUInt32 hintCost = 0u;
    [[unroll]] for (FfxInt32 i = 0; i < 25; ++i)
    {
        // TM1 zero off the level (ArmU8Zero): inside it the clamped texel is the texel.
        const FfxInt32x2 b  = pos + FfxInt32x2(i % 5, i / 5) - ARM_BLOCK_R;
        const FfxInt32   t  = (local.y + i / 5) * TILE + local.x + i % 5;
        const bool       inside = all(greaterThanEqual(b, FfxInt32x2(0))) && all(lessThan(b, ubo._LevelDims));
        const FfxUInt32  tm = inside ? ArmU8(s_tm1[t]) : 0u;
        hintCost += FfxUInt32(abs(FfxInt32(s_hint[t]) - FfxInt32(tm)));
    }
    const FfxUInt32 matchCost = texelFetch(r_arm_match, pos, 0).x;
    const f16vec2   hint      = f16vec2(texelFetch(r_flow_hint, pos, 0).xy);
    imageStore(rw_flow, pos, FfxFloat32x4(FfxFloat32x2(hintCost < matchCost ? hint : filtered), 0.0f, 0.0f));
}
