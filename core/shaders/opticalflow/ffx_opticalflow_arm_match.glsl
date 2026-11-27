// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// Arm's block match restructured for the workgroup, the same
// integer costs and the same spiral order (so the same result): TM1's and the search's texels loaded once a
// workgroup into shared memory, and per offset each block's sum of absolute differences as a separable 5 x 5 box
// sum of the texels' differences (rows, then columns) instead of 25 per texel. A row-sum place reads its 11 search
// texels once for a row of seven offsets; the sums are of 8-bit values held as floats (exact: integers below 2^24);
// the Lucas-Kanade step reads the same texels from shared memory instead of fetching them per thread.
// Ported from Arm's ML Emulation Layer for Vulkan (github.com/arm/ai-ml-emulation-layer-for-vulkan, Apache-2.0).
//
// FFX_OF_IMAGE=arm (ffx_opticalflow_arm_common.h), per level, step 2 of 4: the layer's block_match_of.comp and
// subpixel_me.comp in one pass. Per texel of the level, TM1's 5x5 block against the warped TP1 at the 49 offsets of
// +-3 in the Motion Engine's spiral order (equal costs to the earlier, zero off the level); then a Lucas-Kanade step
// in half precision on the block's border ring (24 texels) around the match, dropped where 1 texel or more, and the
// coarser level's flow added. Also keeps the match's cost, which the last level compares with the hint's.

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

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 1)        uniform texture2D r_pyramid_tm1;  // template
layout(set = 0, binding = 2)        uniform texture2D r_arm_warp;     // search
layout(set = 0, binding = 3)        uniform texture2D r_arm_flow;     // the coarser level's filtered flow
layout(set = 0, binding = 4, rg16f) uniform writeonly image2D rw_arm_sub;
layout(set = 0, binding = 5, r32ui) uniform writeonly uimage2D rw_arm_match;  // the match's cost

#define SR 3
#define SW (2 * SR + 1)
// Each window offset's place in the spiral: the earliest of equal costs wins.
const float kRank[SW * SW] = {43, 42, 41, 40, 39, 38, 37, 44, 21, 20, 19, 18, 17, 36, 45, 22, 7, 6, 5, 16, 35, 46, 23, 8, 0, 4, 15, 34, 47, 24, 1, 2, 3, 14, 33, 48, 9, 10, 11, 12, 13, 32, 25, 26, 27, 28, 29, 30, 31};

// The workgroup's TM1 texels (its blocks, 12 x 12), the search's (blocks and +-SR, 18 x 18), and the row sums; and for the
// Lucas-Kanade step the same texels as floats: TM1 clamped to the level around the ring and its gradients (+-3, 14 x 14),
// the search zero off the level (18 x 18). Each is the texel the earlier per-thread fetches read.
const FfxInt32 kT = 8 + 2 * ARM_BLOCK_R, kW = 8 + 2 * ARM_BLOCK_R + 2 * SR, kF = 8 + 2 * (ARM_BLOCK_R + 1);
shared float     sT[kT * kT];  // the 8-bit values, as floats: every sum below is of integers under 2^24, so exact
shared float     sW[kW * kW];
shared float     sH[SW * kT * 8];
shared float     sTf[kF * kF];
shared float     sWf[kW * kW];

void main()
{
    const FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);

    const FfxInt32x2 base = FfxInt32x2(gl_WorkGroupID.xy) * 8;
    const FfxInt32x2 lid  = FfxInt32x2(gl_LocalInvocationID.xy);
    const FfxInt32   li   = lid.y * 8 + lid.x;
    for (FfxInt32 k = li; k < kT * kT; k += 64)
        sT[k] = float(ArmU8Zero(r_pyramid_tm1, ubo._LevelRow, ubo._LevelDims, base - ARM_BLOCK_R + FfxInt32x2(k % kT, k / kT)));
    for (FfxInt32 k = li; k < kF * kF; k += 64)
        sTf[k] = ArmTexel(r_pyramid_tm1, ubo._LevelRow, ubo._LevelDims, base - ARM_BLOCK_R - 1 + FfxInt32x2(k % kF, k / kF));
    for (FfxInt32 k = li; k < kW * kW; k += 64)
    {
        const FfxInt32x2 p   = base - ARM_BLOCK_R - SR + FfxInt32x2(k % kW, k / kW);
        const bool       off = any(lessThan(p, FfxInt32x2(0))) || any(greaterThanEqual(p, ubo._LevelDims));
        const float      v   = off ? 0.0f : texelFetch(r_arm_warp, p, 0).x;
        sWf[k] = v;
        sW[k]  = off ? 0.0f : float(ArmU8(v));
    }
    barrier();
    // A thread's tile rows (two of the 12 x 8 row-sum places: li and li + 64): their TM1 texels kept for every row of offsets.
    // A thread's row-sum runs (two of the 12 rows x 7 offsets: li and li + 64): their TM1 row kept for every row of offsets.
    const FfxInt32 y0 = li % kT, x0 = li / kT, y1 = (li + 64) % kT, x1 = (li + 64) / kT;
    const bool     two = li + 64 < kT * SW;
    float t0[kT], t1[kT];
    [[unroll]] for (FfxInt32 i = 0; i < kT; ++i)
    {
        t0[i] = sT[y0 * kT + i];
        t1[i] = two ? sT[y1 * kT + i] : 0.0f;
    }
    float      best    = 3.0e38f;
    FfxInt32x2 intFlow = FfxInt32x2(0);
    // A row of seven offsets at a time: all their row sums (5 differences, for the tile's 12 rows and the group's 8 columns), then the block sums.
    // A run is one tile row at one offset: its 12 differences, then the 8 row sums sliding along it (exact: integers).
    for (FfxInt32 sy = 0; sy < SW; ++sy)
    {
        {
            float d[kT];
            [[unroll]] for (FfxInt32 i = 0; i < kT; ++i)
                d[i] = abs(sW[(y0 + sy) * kW + x0 + i] - t0[i]);
            float h = d[0] + d[1] + d[2] + d[3] + d[4];
            sH[x0 * kT * 8 + y0 * 8] = h;
            [[unroll]] for (FfxInt32 x = 1; x < 8; ++x)
            {
                h += d[x + 4] - d[x - 1];
                sH[x0 * kT * 8 + y0 * 8 + x] = h;
            }
        }
        if (two)
        {
            float d[kT];
            [[unroll]] for (FfxInt32 i = 0; i < kT; ++i)
                d[i] = abs(sW[(y1 + sy) * kW + x1 + i] - t1[i]);
            float h = d[0] + d[1] + d[2] + d[3] + d[4];
            sH[x1 * kT * 8 + y1 * 8] = h;
            [[unroll]] for (FfxInt32 x = 1; x < 8; ++x)
            {
                h += d[x + 4] - d[x - 1];
                sH[x1 * kT * 8 + y1 * 8 + x] = h;
            }
        }
        barrier();
        [[unroll]] for (FfxInt32 sx = 0; sx < SW; ++sx)
        {
            float d = 0.0f;
            [[unroll]] for (FfxInt32 j = 0; j < 5; ++j)
                d += sH[(sx * kT + lid.y + j) * 8 + lid.x];
            const float key = d * 64.0f + kRank[sy * SW + sx];  // the cost is at most 25 x 255: (cost << 6) | rank, exact
            if (key < best)
                best = key, intFlow = FfxInt32x2(sx - SR, sy - SR);
        }
        barrier();
    }
    const FfxUInt32 cost = FfxUInt32(best) >> 6;
    if (any(greaterThanEqual(pos, ubo._LevelDims)))
        return;
    imageStore(rw_arm_match, pos, FfxUInt32x4(cost, 0u, 0u, 0u));

    // subpixel_me.comp
    precise float16_t a = float16_t(0.0), b = float16_t(0.0), d = float16_t(0.0), p = float16_t(0.0), q = float16_t(0.0);
    [[unroll]] for (FfxInt32 i = 0; i < 25; ++i)
    {
        const FfxInt32x2 o = FfxInt32x2(i % 5, i / 5) - ARM_BLOCK_R;
        if (abs(o.x) != ARM_BLOCK_R && abs(o.y) != ARM_BLOCK_R)
            continue;  // the ring only
        const FfxInt32x2   bpos = pos + o;
        const FfxInt32     tf   = (lid.y + o.y + ARM_BLOCK_R + 1) * kF + lid.x + o.x + ARM_BLOCK_R + 1;  // bpos in sTf
        const FfxInt32x2   ws   = lid + o + intFlow + ARM_BLOCK_R + SR;                                // bpos + intFlow in sWf
        precise float16_t f    = float16_t(sTf[tf]);
        precise float16_t dfx  = (float16_t(sTf[tf + 1]) - float16_t(sTf[tf - 1])) * float16_t(0.5);
        precise float16_t dfy  = (float16_t(sTf[tf + kF]) - float16_t(sTf[tf - kF])) * float16_t(0.5);
        const float16_t   in_  = any(lessThan(bpos, FfxInt32x2(0))) || any(greaterThanEqual(bpos, ubo._LevelDims)) ? float16_t(0.0) : float16_t(1.0);
        f *= in_, dfx *= in_, dfy *= in_;
        a += dfx * dfx;
        b += dfx * dfy;
        d += dfy * dfy;
        precise float16_t z = float16_t(sWf[ws.y * kW + ws.x]) - f;
        p += z * dfx;
        q += z * dfy;
    }
    precise FfxFloat32   detA = FfxFloat32(a) * FfxFloat32(d) - FfxFloat32(b) * FfxFloat32(b);
    precise FfxFloat32x2 num  = FfxFloat32x2(FfxFloat32(d) * FfxFloat32(p) - FfxFloat32(b) * FfxFloat32(q),
                                             FfxFloat32(a) * FfxFloat32(q) - FfxFloat32(b) * FfxFloat32(p));
    FfxFloat32x2 sub = FfxFloat32x2(0.0f);
    if (detA > 1e-7f)
    {
        sub = num / detA;
        sub = mix(sub, FfxFloat32x2(0.0f), greaterThanEqual(abs(sub), FfxFloat32x2(1.0f)));
    }
    f16vec2 flow = f16vec2(intFlow) - f16vec2(sub);
    if (ubo._CoarseDims.x > 0)  // the upscaled flow mv_process_and_warp.comp keeps
        flow += f16vec2(ArmBilinear(r_arm_flow, 0, ubo._CoarseDims, (FfxFloat32x2(pos) + 0.5f) * 0.5f).xy * 2.0f);
    imageStore(rw_arm_sub, pos, FfxFloat32x4(FfxFloat32x2(flow), 0.0f, 0.0f));
}
