// FSR FG4 prepass "pipe193": forward motion of the previous frame, "mv_0_to_1" (AMD FidelityFX SDK v2.4.0, MIT;
// spec: tools/ffx_extract/mlfg_io.py pipe193_forward_mv).
//
// Reads   t1 mv0     frame 0's dilated mv (RG16F, UV units; last frame's kMlfgDilate output), mvSize
//         t2 depth0  frame 0's downscaled depth (R32F; last frame's kMlfgDilate output), depthSize
//         t3 key     splat1to0 key image (R32_UINT, kMlfgSplatKey, proj 1), mvSize
//         t4 flow    splat1to0 flow image (R32_UINT, kMlfgSplatFlow), mvSize
// Writes  u5 forward mv (RG16F, mvSize): uv0 - uv1 at the frame-0 pixel (FFX sign: splatting it with flowIsUv
//         negates it back to a 0 -> 1 displacement); 65504 = unknown. Values fp16-exact.
// The frame-0 clip position (ndc from the texel's top-left uv, y up, z = depth) reprojected with prevPsToPrev2Ps
// (frame 0 -> frame -1) is compared with uv0 + mv0: within 2 display pixels the pixel moves with the camera and the
// output is the camera-only motion from prevPstoPs (frame 0 -> frame 1); otherwise the splat of the current frame's
// vectors onto frame 0 (its flow, or 65504 where nothing landed). Matrices are AMD's row-vector form,
// clip' = [x y z 1] * M, M given as four rows. Products are summed left to right.
// No timestep here (the 0.25 * 2 is the ndc -> uv half, not the midpoint).
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

cbuffer Constants : register(b0)
{
    float4 c_prevPstoPs[4];       // rows of frame-0 clip -> frame-1 clip
    float4 c_prevPsToPrev2Ps[4];  // rows of frame-0 clip -> frame-(-1) clip
    int4   c_sizes;               // displaySize.xy, mvSize.xy
    int4   c_depthSize;           // depthSize.xy, 0, 0
};
Texture2D<float4> t_mv0 : register(t1);
Texture2D<float4> t_depth0 : register(t2);
Texture2D<uint>   t_key : register(t3);
Texture2D<uint>   t_flow : register(t4);
[[vk::image_format("rg16f")]] RWTexture2D<float4> u_out : register(u5);

float4 RowTimes(float4 v, float4 m[4])  // v * M, M as rows, summed left to right
{
    return ((m[0] * v.x + m[1] * v.y) + m[2] * v.z) + m[3] * v.w;
}

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int2 g = int2(id.xy);
    const int2 msi = c_sizes.zw;
    if (any(g >= msi))
        return;
    const float2 disp = float2(c_sizes.xy);
    const float2 msz  = float2(msi);
    const float2 dsz  = float2(c_depthSize.xy);
    const float2 p86  = float2(g) * FDiv(disp, msz);
    const float2 uvc  = p86 * FDiv(float2(1, 1), disp) + FDiv(float2(0.5, 0.5), msz);
    const int2   di   = TruncI(uvc * dsz);
    const float  z    = max(Inside(di, c_depthSize.xy) ? t_depth0.Load(int3(di, 0)).x : 0.0, 1e-37);
    const float2 uv   = FDiv(p86, disp);
    const float2 ndc  = uv * 2.0 - 1.0;
    const float4 clip = float4(ndc.x, -ndc.y, z, 1.0);
    float4       q    = RowTimes(clip, c_prevPsToPrev2Ps);
    q.xy              = FDiv(q.xy, float2(q.w, q.w));
    const float2 prev2 = float2(q.x, -q.y) * 0.5 + 0.5;
    const int2   mi   = TruncI(uvc * msz);
    const float2 m    = Inside(mi, msi) ? t_mv0.Load(int3(mi, 0)).xy : float2(0, 0);
    const float2 err  = ((uv - m * -1.0) - prev2) * disp;
    float4       r    = RowTimes(clip, c_prevPstoPs);
    r.xy              = FDiv(r.xy, float2(r.w, r.w));
    const float2 cam  = -(FDiv(((r.xy - clip.xy) * float2(0.25, -0.25)) * disp, disp) * 2.0);
    const uint   key  = t_key.Load(int3(g, 0));
    const float2 fl   = UnpackF16x2(t_flow.Load(int3(g, 0)));
    const bool   hit  = key > 0u && fl.x != F16_MAX;
    const float2 fb   = hit ? fl : float2(F16_MAX, F16_MAX);
    const bool   useCam = !(max(abs(err.x), abs(err.y)) > 2.0);
    u_out[g] = float4(F16Round(useCam ? cam : fb), 0, 0);
}
