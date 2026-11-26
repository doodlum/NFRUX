// NFRUX's DLSS-G rules over an FSR FG4 frame (kMlfgComposite's output): the no-warp mask, the game's UI and the BGRA8
// byte order, as the NFRU interpolator's composite applies them (core/gpu/frameinterpolation/
// ffx_frameinterpolation_50_postprocess_pass.h, docs/DESIGN.md D8). Built per output format (MLFG_OUT_*).
//
// Reads   t1 gen      the generated frame (kMlfgComposite* into a texture of the output's format), size
//         t2 cur      the frame interpolated (current colour, HUD-less when the game has one)
//         t3 ui       the frame with its UI (uiKind 0: HUD-less difference test; 2/3 with alpha; 4/5 alpha only)
//         t4 alpha    UI opacity: red (UI alpha image) or alpha (a UI layer: kinds 3 / 5)
//         t5 nowarp   no-warp mask (red > 0.5)
// Writes  u6 out      size, the output's format; alpha 0 (as kMlfgComposite: the same bytes with and without this pass)
//   no-warp pixel: cur; then kind 2/3: frame - (1 - a) * (HUD-less - generated) (opaque UI where that colour cannot be
//   right); kind 4/5: the frame where the opacity is above 0.4 in the pixel's 2x2 block; kind 0: the frame where a
//   channel differs from the HUD-less by more than 0.07 in the block. c_size.w bit 0: a UI source is given, bit 1:
//   the no-warp mask, bit 2: store blue first (BGRA8 bytes in an RGBA8 image).
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

cbuffer Constants : register(b0)
{
    int4 c_size;  // width, height, uiKind, flags
};
Texture2D<float4> t_gen : register(t1);
Texture2D<float4> t_cur : register(t2);
Texture2D<float4> t_ui : register(t3);
Texture2D<float4> t_alpha : register(t4);
Texture2D<float4> t_nowarp : register(t5);
#if defined(MLFG_OUT_RGBA8)
[[vk::image_format("rgba8")]] RWTexture2D<float4> u_out : register(u6);
#elif defined(MLFG_OUT_RGB10A2)
[[vk::image_format("rgb10a2")]] RWTexture2D<float4> u_out : register(u6);
#else
[[vk::image_format("rgba16f")]] RWTexture2D<float4> u_out : register(u6);
#endif

// As kMlfgComposite: the code k stored as (k + 0.25) / max, which every implementation's conversion stores as k.
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

float Opacity(int2 p, uint kind) { const float4 v = t_alpha.Load(int3(p, 0)); return (kind == 3u || kind == 5u) ? v.a : v.r; }

groupshared uint s_ui[64];
bool UiPixel(int2 p, uint kind)
{
    if (any(p >= c_size.xy) || (c_size.w & 1) == 0 || kind == 2u || kind == 3u)
        return false;
    if (kind == 4u || kind == 5u)
        return Opacity(p, kind) > 0.4;
    const float3 d = abs(t_ui.Load(int3(p, 0)).rgb - t_cur.Load(int3(p, 0)).rgb);
    return max(d.r, max(d.g, d.b)) > 0.07;
}

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    const int2 g    = int2(id.xy);
    const uint kind = uint(c_size.z);
    s_ui[lid.y * 8u + lid.x] = UiPixel(g, kind) ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    if (any(g >= c_size.xy))
        return;
    const uint i     = (lid.y & ~1u) * 8u + (lid.x & ~1u);
    const bool block = (s_ui[i] | s_ui[i + 1u] | s_ui[i + 8u] | s_ui[i + 9u]) != 0u;
    float3     o     = t_gen.Load(int3(g, 0)).rgb;
    if ((c_size.w & 2) != 0 && t_nowarp.Load(int3(g, 0)).r > 0.5)
        o = t_cur.Load(int3(g, 0)).rgb;
    if ((c_size.w & 1) != 0)
    {
        const float3 ui = t_ui.Load(int3(g, 0)).rgb;
        if (kind == 2u || kind == 3u)
        {
            const float  a   = Opacity(g, kind);
            const float3 cur = t_cur.Load(int3(g, 0)).rgb;
            const float3 uiColour = ui - (1.0 - a) * cur;
            const bool   bad = any(uiColour != uiColour) || any(abs(uiColour) > 3.0e38) || any(uiColour <= -1.0) || any(uiColour < ui - 2.0 * cur);
            o = bad ? ui : uiColour + (1.0 - a) * o;
        }
        else if (block)
            o = ui;
    }
    float4 q = Quantise(o);
    if ((c_size.w & 4) != 0)
        q = q.bgra;
    u_out[g] = q;
}
