// FSR FG4 prepass "pipe191": motion-vector dilation + depth downscale (AMD FidelityFX SDK v2.4.0, MIT; spec:
// tools/ffx_extract/mlfg_io.py pipe191_dilate). Built twice: MLFG_INVERTED_DEPTH 0 / 1 (the Inv variant).
//
// Reads   t1 mv     render-res motion vectors (RG16F or any float2 texture; FFX convention mv(p1) = p0 - p1, in pixels
//                   times motionVectorScale), renderSize
//         t2 depth  render-res depth (R32F), renderSize
// Writes  u3 dilated mv, outSize (1/3 of the display in AMD's plan), RG16F, UV units, values fp16-exact
//         u4 depth, outSize, R32F: the depth of the nearest render pixel (not the dilated one)
// For each output texel g:
//   src = ((g - colorRectMin*outSize + o) * renderSize + renderSize/2) / trunc(colorRect size * outSize)   (integers)
//   pick the nearest-to-camera depth among src and its 8 neighbours (neighbours clamped, first wins ties, order centre,
//   (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)); mv = mv[picked] * scale * colorRect size / renderSize.
//   Outside the generation rect: depth = 0 (inverted) / 1, mv = 0.
// AMD's lens-distortion field (r_distortion_field) is not ported: AMD's host leaves it unbound and it reads as 0.
// No timestep here.
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

#ifndef MLFG_INVERTED_DEPTH
#define MLFG_INVERTED_DEPTH 0
#endif

cbuffer Constants : register(b0)
{
    int4   c_sizes;      // renderSize.xy, outSize.xy (AMD dispatchSize)
    float4 c_mvScale;    // motionVectorScale.xy, 0, 0
    float4 c_genRect;    // generationRectMinUV.xy, generationRectMaxUV.xy
    float4 c_colorRect;  // colorRectMinUV.xy, colorRectMaxUV.xy
};
Texture2D<float4> t_mv : register(t1);
Texture2D<float4> t_depth : register(t2);
[[vk::image_format("rg16f")]] RWTexture2D<float4> u_mv : register(u3);
[[vk::image_format("r32f")]] RWTexture2D<float4> u_depth : register(u4);

float4 Fetch(Texture2D<float4> t, int2 p, int2 size) { return Inside(p, size) ? t.Load(int3(p, 0)) : float4(0, 0, 0, 0); }

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int2 g = int2(id.xy);
    const int2 rs = c_sizes.xy, outSize = c_sizes.zw;
    if (any(g >= outSize))
        return;
    const float2 dsz = float2(outSize);
    const float2 gf  = float2(g);
    const float2 uv  = FDiv(gf + 0.5, dsz);
    const bool   outside = any(uv > c_genRect.zw) || any(uv < c_genRect.xy);
    const int2   base = TruncI(gf - c_colorRect.xy * dsz);
    const float2 rect = c_colorRect.zw - c_colorRect.xy;
    const int2   rsz  = max(TruncI(rect * dsz), int2(1, 1));
    const int2   half = rs / 2;
    int2         best = ((base * rs) + half) / rsz;
    float        bestD = Fetch(t_depth, best, rs).x;
    const float  dOut  = bestD;
    const int2   offs[8] = {int2(-1, -1), int2(-1, 0), int2(-1, 1), int2(0, -1), int2(0, 1), int2(1, -1), int2(1, 0), int2(1, 1)};
    [unroll] for (int i = 0; i < 8; ++i)
    {
        const int2  c = min(max(((base + offs[i]) * rs + half) / rsz, int2(0, 0)), rs - 1);
        const float d = Fetch(t_depth, c, rs).x;
#if MLFG_INVERTED_DEPTH
        const bool take = d > bestD;
#else
        const bool take = d < bestD;
#endif
        if (take)
        {
            bestD = d;
            best  = c;
        }
    }
    float2 m = Fetch(t_mv, best, rs).xy;
    m = FDiv((m * c_mvScale.xy) * rect, float2(rs));
#if MLFG_INVERTED_DEPTH
    const float outsideDepth = 0.0;
#else
    const float outsideDepth = 1.0;
#endif
    u_depth[g] = float4(outside ? outsideDepth : dOut, 0, 0, 0);
    u_mv[g]    = float4(outside ? float2(0, 0) : F16Round(m), 0, 0);
}
