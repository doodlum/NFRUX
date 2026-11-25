/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_FFX_FRAMEINTERPOLATION_20_PREPROCESS_PASS_H
#define GPU_FFX_FRAMEINTERPOLATION_20_PREPROCESS_PASS_H

#include "ffx_frameinterpolation_callbacks_glsl.h"
#include "quant.h"

float SingleTapDepthClip(float current_depth, float previous_depth)
{
    const float fCurrentDepthViewSpace     = ConvertFromDeviceDepthToViewSpace(current_depth);
    const float fPrevNearestDepthViewSpace = ConvertFromDeviceDepthToViewSpace(previous_depth);

    const float fDepthDiff = fCurrentDepthViewSpace - fPrevNearestDepthViewSpace;
    if (fDepthDiff <= 0)
        return 0;

    const float fDepthThreshold          = max(fCurrentDepthViewSpace, fPrevNearestDepthViewSpace);
    const float fRequiredDepthSeparation = DepthClipScale() * fDepthThreshold;
    const float fDepth                   = pow(ffxSaturate(fRequiredDepthSeparation / fDepthDiff), DepthClipPower());

    return 1.0 - fDepth;
}

// Transform tp1 depth values to tm1 frame's screen space.
// NOTE: the SDK assumes depth and the reprojection matrices share the SAME convention.
// So the device depth passed here is already in the matrix convention: pass `depth` directly (NOT
// 1-depth) and return reproj.z/w directly
float reproject_depth_fwd(float2 uv, float depth)
{
    float2 uvT    = float2(uv.x, 1.0f - uv.y);
    float4 clip   = float4(2.0f * uvT - 1.0f, depth, 1.0f);
    float4 reproj = FFX_TRANSFORM_VECTOR(GetMotionTP1ToTM1(), clip);
    return reproj.z / reproj.w;
}

// Transform tm1 depth values to tp1 frame's screen space
float reproject_depth_bwd(float2 uv, float depth)
{
    float2 uvT    = float2(uv.x, 1.0f - uv.y);
    float4 clip   = float4(2.0f * uvT - 1.0f, depth, 1.0f);
    float4 reproj = FFX_TRANSFORM_VECTOR(GetMotionTM1ToTP1(), clip);
    return reproj.z / reproj.w;
}

FFX_MIN16_F2 SingleTapCalculateDisocclusionMask(float depth_m1_warp_t_mv, float depth_p1_warp_t_mv, float2 uv_m1_mv, float2 uv_p1_mv)
{
    // Reproject/transform depth to respective opposite depth screen space (depth already in matrix convention)
    float depth_m1_warp_t_mv_transform_p1 = reproject_depth_bwd(uv_m1_mv, depth_m1_warp_t_mv);
    float depth_p1_warp_t_mv_transform_m1 = reproject_depth_fwd(uv_p1_mv, depth_p1_warp_t_mv);

    float dis_mask_m1 = SingleTapDepthClip(depth_p1_warp_t_mv, depth_m1_warp_t_mv_transform_p1);
    float dis_mask_p1 = SingleTapDepthClip(depth_m1_warp_t_mv, depth_p1_warp_t_mv_transform_m1);

    return FFX_MIN16_F2(dis_mask_m1, dis_mask_p1);
}

uint32_t hash32_fast(uint32_t v)
{
    v = (v ^ 61u) ^ (v >> 16);
    v *= 9u;
    v ^= v >> 4;
    v *= 0x27d4eb2du;
    v ^= v >> 15;
    return v;
}

half hash_to_half(uint32_t x, uint32_t y)
{
    uint32_t v = x + 0x9e3779b9u * y + RandomSeed();
    uint32_t h = hash32_fast(v);
    return half(uintBitsToFloat((h >> 9) | 0x3f800000u) - 1.0f);
}

void WriteToInputTensor(int32_t2     outputPixel,
                        FFX_MIN16_F3 rgb_m1_warp_t_mv,
                        FFX_MIN16_F3 rgb_p1_warp_t_mv,
                        FFX_MIN16_F3 rgb_m1_warp_t_flow,
                        FFX_MIN16_F3 rgb_p1_warp_t_flow,
                        FFX_MIN16_F2 dis_mask,
                        float2       depth,
                        bool         colourBytes,
                        uint32_t     m1FlowBytes,
                        uint32_t     p1FlowBytes,
                        FFX_MIN16_F2 mv_t_f30_m1)
{
    InputTensorElement_t te;

    te.rgbM1MV_rP1MV = PackInt8(Quantize(FFX_MIN16_F4(rgb_m1_warp_t_mv.rgb, rgb_p1_warp_t_mv.r), QuantParamsSINT().xy));
    if (colourBytes)  // the flow warps' colours as their quantised bytes (rgb in the low three)
    {
        const FfxInt32x2 gb = Quantize(FFX_MIN16_F2(rgb_p1_warp_t_mv.gb), QuantParamsSINT().xy);
        te.gbP1MV_rgM1OF = (PackInt8(FfxInt32x4(gb, 0, 0)) & 0xffffu) | ((m1FlowBytes & 0xffffu) << 16);
        te.bM1OF_rgbP1OF = ((m1FlowBytes >> 16) & 0xffu) | ((p1FlowBytes & 0xffffffu) << 8);
    }
    else
    {
        te.gbP1MV_rgM1OF = PackInt8(Quantize(FFX_MIN16_F4(rgb_p1_warp_t_mv.gb, rgb_m1_warp_t_flow.rg), QuantParamsSINT().xy));
        te.bM1OF_rgbP1OF = PackInt8(Quantize(FFX_MIN16_F4(rgb_m1_warp_t_flow.b, rgb_p1_warp_t_flow.rgb), QuantParamsSINT().xy));
    }
    te.depth_disMask = PackInt8(Quantize(FFX_MIN16_F4(depth, dis_mask), QuantParamsSINT().xy));
    // The multi-frame graph's channels 16-19, as the SDK fork writes them (docs/NFRU/NFRU.md there).
    const float t         = Timestep();
    const float mv_pixels = length(float2(mv_t_f30_m1) * float2(MvDepthLaneSize()));
    te.timestep_pad = PackInt8(Quantize(FFX_MIN16_F4(t, ffxSaturate(t * mv_pixels / 64.0f), ffxSaturate((1.0f - t) * mv_pixels / 64.0f), 0.0f),
                                        QuantParamsSINT().xy));

    StoreInTensor(outputPixel, te);
}

// read the raw packed warped MV/depth (QMVD) for the t->m1 motion, detecting holes (raw == 0) and
// dynamic-OOB markers (raw == 0x7FFFFFFF).
FFX_MIN16_F2 LoadMotionQMVD(int32_t2 pixel, out bool dynamic_oob_hole)
{
    FfxUInt32 packed = LoadWarpedMotionQDataTP1(pixel);
    dynamic_oob_hole = (packed == 0x7FFFFFFFu);
    float2 motion    = dqFloatMv(packed);
    motion += float2(0.001f) * step(0.5f, float(packed));
    motion *= float2(1.0f) / float2(MvDepthLaneSize());
    return FFX_MIN16_F2(motion) * FFX_MIN16_F(!dynamic_oob_hole);
}

// In addition to unpacking the raw QMVD motion, detect hole edges. A hole edge is a pixel that has at
// least one (but not all) zero-motion (no-scatter) neighbour in its 8-neighbourhood.
// The edge mask suppresses 3x3 depth dilation across hole boundaries and gates
// the inpaint seed.
FFX_MIN16_F Count4(bvec4 b) { return FFX_MIN16_F(uint32_t(b.x) + uint32_t(b.y) + uint32_t(b.z) + uint32_t(b.w)); }
bvec4 And4(bvec4 a, bvec4 b) { return bvec4(a.x && b.x, a.y && b.y, a.z && b.z, a.w && b.w); }
FFX_MIN16_F2 LoadMotionAndHoleEdgeDetect(int32_t2 pixel, out FFX_MIN16_F hole_edge_mask, out bool dynamic_oob_hole)
{
    // The eight around it: two gathers (at the pixel's opposite corners) and the two corners they miss. Pixels off the lane do not count.
    const int32_t2 size   = WarpedMotionQDataTP1Size();
    const bvec4    side   = bvec4(greaterThan(pixel, int32_t2(0)), lessThan(pixel + 1, size));  // left, up, right, down
    const bvec4    corner = bvec4(side.x && side.y, side.z && side.w, side.x && side.w, side.z && side.y);
    const uvec4    ga     = textureGather(usampler2D(r_warped_motion_qdata_tp1, s_PointClamp), float2(pixel) / float2(size));
    const uvec4    gd     = textureGather(usampler2D(r_warped_motion_qdata_tp1, s_PointClamp), float2(pixel + 1) / float2(size));
    const uint32_t g0     = LoadWarpedMotionQDataTP1(clamp(pixel + int32_t2(-1, 1), int32_t2(0), size - 1));
    const uint32_t g1     = LoadWarpedMotionQDataTP1(clamp(pixel + int32_t2(1, -1), int32_t2(0), size - 1));
    hole_edge_mask = Count4(And4(side, equal(uvec4(ga.x, ga.z, gd.z, gd.x), uvec4(0u)))) + Count4(And4(corner, equal(uvec4(ga.w, gd.y, g0, g1), uvec4(0u))));
    const FFX_MIN16_F valid_sample_count = Count4(side) + Count4(corner);
    hole_edge_mask *= FFX_MIN16_F(hole_edge_mask != valid_sample_count);

    return LoadMotionQMVD(pixel, dynamic_oob_hole);
}

// 3x3 min-depth dilation (selects the nearest surface within the neighbourhood). Skipped when masked
// (double-disocclusion / hole edge). Returns the selected (dilated) sample position so the disocclusion
// reprojection uses the representative surface.
float LoadDilatedDepthTm1(int32_t2 pos, int32_t2 depth_size, FFX_MIN16_F mask, out int32_t2 selected_pos)
{
    // the 8 neighbours' loads issued together (coordinates clamped), then Arm's ordered strict selection
    // over the on-screen ones: Arm's result, without a dependent load and wait per tap (22 full waits -> 8).
    const int32_t2 offsets[8] = {int32_t2(+1, +0), int32_t2(+0, +1), int32_t2(+0, -1), int32_t2(-1, +0),
                                 int32_t2(-1, +1), int32_t2(+1, +1), int32_t2(-1, -1), int32_t2(+1, -1)};
    float best_depth = LoadInputDepthTm1Pixel(pos);
    selected_pos     = pos;
    if (mask > FFX_MIN16_F(0.0f))
        return best_depth;
    const int32_t2 lo = clamp(pos - 1, int32_t2(0), depth_size - 1), hi = clamp(pos + 1, int32_t2(0), depth_size - 1);
    const bool     vxm = pos.x - 1 >= 0 && pos.x - 1 < depth_size.x, vxp = pos.x + 1 >= 0 && pos.x + 1 < depth_size.x;
    const bool     vx0 = pos.x >= 0 && pos.x < depth_size.x;
    const bool     vym = pos.y - 1 >= 0 && pos.y - 1 < depth_size.y, vyp = pos.y + 1 >= 0 && pos.y + 1 < depth_size.y;
    const bool     vy0 = pos.y >= 0 && pos.y < depth_size.y;
    // The eight around it: two gathers (at the pixel's opposite corners) and the two corners they miss.
    const float4 ga = textureGather(sampler2D(r_depth_tm1, s_PointClamp), float2(pos) / float2(depth_size));
    const float4 gd = textureGather(sampler2D(r_depth_tm1, s_PointClamp), float2(pos + 1) / float2(depth_size));
    const float  sd[8] = {gd.z, gd.x, ga.z, ga.x, LoadInputDepthTm1Pixel(int32_t2(lo.x, hi.y)), gd.y, ga.w, LoadInputDepthTm1Pixel(int32_t2(hi.x, lo.y))};
    const bool   v[8]  = {vxp && vy0, vx0 && vyp, vx0 && vym, vxm && vy0, vxm && vyp, vxp && vyp, vxm && vym, vxp && vym};
    for (int32_t i = 0; i < 8; ++i)
    {
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
        if (v[i] && sd[i] > best_depth)
#else
        if (v[i] && sd[i] < best_depth)
#endif
        {
            best_depth   = sd[i];
            selected_pos = pos + offsets[i];
        }
    }
    return best_depth;
}

float LoadDilatedDepthTp1(int32_t2 pos, int32_t2 depth_size, FFX_MIN16_F mask, out int32_t2 selected_pos)
{
    // the 8 neighbours' loads issued together (coordinates clamped), then Arm's ordered strict selection
    // over the on-screen ones: Arm's result, without a dependent load and wait per tap (22 full waits -> 8).
    const int32_t2 offsets[8] = {int32_t2(+1, +0), int32_t2(+0, +1), int32_t2(+0, -1), int32_t2(-1, +0),
                                 int32_t2(-1, +1), int32_t2(+1, +1), int32_t2(-1, -1), int32_t2(+1, -1)};
    float best_depth = LoadDepthTp1FeedbackPixel(pos);
    selected_pos     = pos;
    if (mask > FFX_MIN16_F(0.0f))
        return best_depth;
    const int32_t2 lo = clamp(pos - 1, int32_t2(0), depth_size - 1), hi = clamp(pos + 1, int32_t2(0), depth_size - 1);
    const bool     vxm = pos.x - 1 >= 0 && pos.x - 1 < depth_size.x, vxp = pos.x + 1 >= 0 && pos.x + 1 < depth_size.x;
    const bool     vx0 = pos.x >= 0 && pos.x < depth_size.x;
    const bool     vym = pos.y - 1 >= 0 && pos.y - 1 < depth_size.y, vyp = pos.y + 1 >= 0 && pos.y + 1 < depth_size.y;
    const bool     vy0 = pos.y >= 0 && pos.y < depth_size.y;
    // The eight around it: two gathers (at the pixel's opposite corners) and the two corners they miss.
    const float4 ga = textureGather(sampler2D(r_depth_tm1_next, s_PointClamp), float2(pos) / float2(depth_size));
    const float4 gd = textureGather(sampler2D(r_depth_tm1_next, s_PointClamp), float2(pos + 1) / float2(depth_size));
    const float  sd[8] = {gd.z, gd.x, ga.z, ga.x, LoadDepthTp1FeedbackPixel(int32_t2(lo.x, hi.y)), gd.y, ga.w, LoadDepthTp1FeedbackPixel(int32_t2(hi.x, lo.y))};
    const bool   v[8]  = {vxp && vy0, vx0 && vyp, vx0 && vym, vxm && vy0, vxm && vyp, vxp && vyp, vxm && vym, vxp && vym};
    for (int32_t i = 0; i < 8; ++i)
    {
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
        if (v[i] && sd[i] > best_depth)
#else
        if (v[i] && sd[i] < best_depth)
#endif
        {
            best_depth   = sd[i];
            selected_pos = pos + offsets[i];
        }
    }
    return best_depth;
}

FfxInt32 ReflectPadIndex(FfxInt32 index, FfxInt32 size)
{
    return index < size ? index : max(0, 2 * size - index - 2);
}

FfxInt32x2 ReflectPadCoord(FfxInt32x2 coord)
{
    return FfxInt32x2(ReflectPadIndex(coord.x, MvDepthLaneSize().x), ReflectPadIndex(coord.y, MvDepthLaneSize().y));
}

FFX_MIN16_F3 LoadColourPreprocessPrev(float2 warp_uv)
{
    FfxInt32x2 colour_size = PreprocessColourM1Size();
    return LoadPreprocessColourM1Pixel(FfxInt32x2(floor(warp_uv * float2(colour_size))));
}

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_BYTES_M1)
// The lane's colour as R32_UINT (its rgba8 bytes), at the texel LoadColourPreprocessPrev / Cur take.
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_BYTES_M1) uniform utexture2D r_preprocess_bytes_m1;
layout(set = 0, binding = FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_BYTES_P1) uniform utexture2D r_preprocess_bytes_p1;
uint32_t LoadPreprocessBytesM1(float2 warp_uv)
{
    const FfxInt32x2 colour_size = PreprocessColourM1Size();
    return texelFetch(r_preprocess_bytes_m1, ClampTexelCoord(FfxInt32x2(floor(warp_uv * float2(colour_size))), r_preprocess_bytes_m1), 0).x;
}
uint32_t LoadPreprocessBytesP1(float2 warp_uv)
{
    const FfxInt32x2 colour_size = PreprocessColourP1Size();
    return texelFetch(r_preprocess_bytes_p1, ClampTexelCoord(FfxInt32x2(floor(warp_uv * float2(colour_size))), r_preprocess_bytes_p1), 0).x;
}
#endif

FFX_MIN16_F3 LoadColourPreprocessCur(float2 warp_uv)
{
    FfxInt32x2 colour_size = PreprocessColourP1Size();
    return LoadPreprocessColourP1Pixel(FfxInt32x2(floor(warp_uv * float2(colour_size))));
}

void preprocess(int32_t2 input_pixel)
{
    //-------------------------------------------------------------------------
    // 1) Dispatch guard
    //-------------------------------------------------------------------------
    if (any(greaterThanEqual(input_pixel, InputTensorSize())))
    {
        return;
    }

    int32_t2 tensor_pixel        = input_pixel;
    int32_t2 output_pixel        = ReflectPadCoord(tensor_pixel);
    bool     padded_tensor_pixel = any(notEqual(tensor_pixel, output_pixel));

    // The tensor/preprocess lane is the 540p mvDepthLane (padded). Sample everything via uv in
    // preprocess-lane space; the flow lane (270p) is sampled with the same uv.
    float2 invPreprocess = MvDepthLaneSizeRcp();
    float2 uv            = (float2(output_pixel) + 0.5f) * invPreprocess;

    int32_t2 iMvCoord = clamp(output_pixel, int32_t2(0), WarpedMotionQDataTP1Size() - int32_t2(1));

    //-------------------------------------------------------------------------
    // 2) Motion and OF sampling. Reprojection setup
    //-------------------------------------------------------------------------
    FFX_MIN16_F2 flow_t_f30_p1    = SampleWarpedFilledFlowTM1(uv);
    bool         dynamic_oob_hole = false;
    FFX_MIN16_F  hole_edge_mask   = FFX_MIN16_F(0.0f);
    FFX_MIN16_F2 mv_t_f30_m1      = LoadMotionAndHoleEdgeDetect(iMvCoord, hole_edge_mask, dynamic_oob_hole);

    // Calculate warped game mv positions
    float  timestep = Timestep();
    float2 uv_m1_mv = uv + float2(mv_t_f30_m1 * timestep);
    float2 uv_p1_mv = uv - float2(mv_t_f30_m1 * (1.0f - timestep));

    // Calculate warped optical flow positions
    float2 uv_m1_flow = uv - float2(flow_t_f30_p1 * timestep);
    float2 uv_p1_flow = uv + float2(flow_t_f30_p1 * (1.0f - timestep));

    // Calculate OOB areas
    bool uv_p1_mv_oob   = (uv_p1_mv.x <= 0.0 || uv_p1_mv.x >= 1.0) || (uv_p1_mv.y <= 0.0 || uv_p1_mv.y >= 1.0);
    bool uv_m1_mv_oob   = (uv_m1_mv.x <= 0.0 || uv_m1_mv.x >= 1.0) || (uv_m1_mv.y <= 0.0 || uv_m1_mv.y >= 1.0);
    bool uv_m1_flow_oob = (uv_m1_flow.x <= 0.0 || uv_m1_flow.x >= 1.0) || (uv_m1_flow.y <= 0.0 || uv_m1_flow.y >= 1.0);
    bool uv_p1_flow_oob = (uv_p1_flow.x <= 0.0 || uv_p1_flow.x >= 1.0) || (uv_p1_flow.y <= 0.0 || uv_p1_flow.y >= 1.0);

    //-------------------------------------------------------------------------
    // 3) Colour motion and OF warping
    //-------------------------------------------------------------------------

    bool mv_hole         = (mv_t_f30_m1.x == FFX_MIN16_F(0.0f) && mv_t_f30_m1.y == FFX_MIN16_F(0.0f));
    bool double_oob_hole = mv_hole || (uv_m1_mv_oob && uv_p1_mv_oob);
    bool invalid_mv_pos  = double_oob_hole || dynamic_oob_hole;
    uv_p1_mv_oob         = uv_p1_mv_oob || invalid_mv_pos;
    uv_m1_mv_oob         = uv_m1_mv_oob || invalid_mv_pos;

    FFX_MIN16_F3 rgb_m1_warp_t_mv   = FFX_MIN16_F3(0.0f);
    FFX_MIN16_F3 rgb_p1_warp_t_mv   = FFX_MIN16_F3(0.0f);
    FFX_MIN16_F3 rgb_m1_warp_t_flow = FFX_MIN16_F3(0.0f);
    FFX_MIN16_F3 rgb_p1_warp_t_flow = FFX_MIN16_F3(0.0f);

    if (!uv_m1_mv_oob)
        rgb_m1_warp_t_mv = LoadColourPreprocessPrev(uv_m1_mv);
    if (!uv_p1_mv_oob)
        rgb_p1_warp_t_mv = LoadColourPreprocessCur(uv_p1_mv);
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_BYTES_M1)
    // the lane's colour 8-bit unorm (_Options 0x800): the flow warps' colours only feed the network, whose
    // quantisation (scale 255, zero point -128) takes a channel n / 255 to n - 128 exactly: their bytes, top bits flipped.
    const bool colourBytes = true;  // kFiPreprocess8, the rgba8 lane's (used with _Options 0x800)
#else
    const bool colourBytes = false;
#endif
    uint32_t m1FlowBytes = 0u, p1FlowBytes = 0u;
    if (!uv_m1_flow_oob)
    {
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_BYTES_M1)
        if (colourBytes)
            m1FlowBytes = LoadPreprocessBytesM1(uv_m1_flow) ^ 0x80808080u;
        else
#endif
            rgb_m1_warp_t_flow = LoadColourPreprocessPrev(uv_m1_flow);
    }
    if (!uv_p1_flow_oob)
    {
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREPROCESS_BYTES_M1)
        if (colourBytes)
            p1FlowBytes = LoadPreprocessBytesP1(uv_p1_flow) ^ 0x80808080u;
        else
#endif
            rgb_p1_warp_t_flow = LoadColourPreprocessCur(uv_p1_flow);
    }

    uint32_t seed_base = uint32_t(output_pixel.x) * 10000u + uint32_t(output_pixel.y);
#if !(DISABLE_RANDOM_OOB)
    if (uv_m1_mv_oob)
    {
        rgb_m1_warp_t_mv = FFX_MIN16_F3(hash_to_half(seed_base, 0u), hash_to_half(seed_base, 1u), hash_to_half(seed_base, 2u));
    }
    if (uv_p1_mv_oob)
    {
        rgb_p1_warp_t_mv = FFX_MIN16_F3(hash_to_half(seed_base, 3u), hash_to_half(seed_base, 4u), hash_to_half(seed_base, 5u));
    }
    if (uv_m1_flow_oob)
    {
        rgb_m1_warp_t_flow = FFX_MIN16_F3(hash_to_half(seed_base, 6u), hash_to_half(seed_base, 7u), hash_to_half(seed_base, 8u));
        m1FlowBytes        = PackInt8(Quantize(FFX_MIN16_F4(rgb_m1_warp_t_flow, 0.0f), QuantParamsSINT().xy));
    }
    if (uv_p1_flow_oob)
    {
        rgb_p1_warp_t_flow = FFX_MIN16_F3(hash_to_half(seed_base, 9u), hash_to_half(seed_base, 10u), hash_to_half(seed_base, 11u));
        p1FlowBytes        = PackInt8(Quantize(FFX_MIN16_F4(rgb_p1_warp_t_flow, 0.0f), QuantParamsSINT().xy));
    }
#else
    if (uv_m1_flow_oob)
        m1FlowBytes = 0x80808080u;  // colour 0
    if (uv_p1_flow_oob)
        p1FlowBytes = 0x80808080u;
#endif

    //-------------------------------------------------------------------------
    // 4) Depth warp and disocclusion mask calculation
    //-------------------------------------------------------------------------
    // Double disocclusion mask:
    // 1) If hole in scattered mvs
    // 2) p1/m1 warps go oob
    // NOTE: Do not include 0x7FFFFFF values here, since they indicate dynamic oob holes that the network must learn to handle rather than inpainting.
    half        dd_mask        = half(double_oob_hole);
    int32_t2    depth_tm1_size = int32_t2(DepthTm1Size());
    int32_t2    depth_tp1_size = int32_t2(DepthTp1FeedbackSize());
    FFX_MIN16_F depth_dil_mask = dd_mask + hole_edge_mask;

#if FFX_ARM_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
    float depth_m1_warp_t_mv = 1.0f;
    float depth_p1_warp_t_mv = 1.0f;
#else
    float depth_m1_warp_t_mv = 0.0f;
    float depth_p1_warp_t_mv = 0.0f;
#endif
    float2 uv_m1_depth = uv_m1_mv;
    float2 uv_p1_depth = uv_p1_mv;

    if (!uv_m1_mv_oob)
    {
        int32_t2 sel;
        depth_m1_warp_t_mv = LoadDilatedDepthTm1(int32_t2(floor(uv_m1_mv * float2(depth_tm1_size))), depth_tm1_size, depth_dil_mask, sel);
        uv_m1_depth        = (float2(sel) + 0.5f) / float2(depth_tm1_size);
    }
    if (!uv_p1_mv_oob)
    {
        int32_t2 sel;
        depth_p1_warp_t_mv = LoadDilatedDepthTp1(int32_t2(floor(uv_p1_mv * float2(depth_tp1_size))), depth_tp1_size, depth_dil_mask, sel);
        uv_p1_depth        = (float2(sel) + 0.5f) / float2(depth_tp1_size);
    }

    // Normalize Depth
    float  depth_m1_warp_t_mv_norm = NormalizeDepth(depth_m1_warp_t_mv);
    float  depth_p1_warp_t_mv_norm = NormalizeDepth(depth_p1_warp_t_mv);
    float2 depth                   = float2(depth_m1_warp_t_mv_norm, depth_p1_warp_t_mv_norm);

    // Calculate disocclusion mask (using the dilated-selected depth positions)
    FFX_MIN16_F2 dis_mask = SingleTapCalculateDisocclusionMask(depth_m1_warp_t_mv, depth_p1_warp_t_mv, uv_m1_depth, uv_p1_depth);

    dis_mask.x = ffxSaturate(dis_mask.x + dd_mask + half(dynamic_oob_hole));
    dis_mask.y = ffxSaturate(dis_mask.y + dd_mask + half(dynamic_oob_hole));

    //-------------------------------------------------------------------------
    // 5) Inpainting initialization
    //-------------------------------------------------------------------------
    // Inpaint seed: a background-colour guess for hole regions, written on the preprocess lane and
    // later spread into holes by the inpaint mip chain. Valid at single-disocclusion edges (one side
    // occluded so the other warp supplies a plausible background); skipped on double-disocclusion / OOB.
    {
        FFX_MIN16_F t_seed = FFX_MIN16_F(0.5f);
        t_seed += FFX_MIN16_F(0.5f) * (FFX_MIN16_F(1.0f) - dis_mask.x);
        t_seed -= FFX_MIN16_F(0.5f) * (FFX_MIN16_F(1.0f) - dis_mask.y);

        // edge_valid: a true hole edge (>=1 zero-motion neighbour, both MV warps in-bounds), matching
        // Coverage additionally drops where both sides are disoccluded.
        FfxFloat32   edge_valid   = (hole_edge_mask > FFX_MIN16_F(0.0f) && !(uv_m1_mv_oob || uv_p1_mv_oob)) ? 1.0f : 0.0f;
        FfxFloat32   coverage     = (1.0f - FfxFloat32((dis_mask.x + dis_mask.y) >= FFX_MIN16_F(2.0f))) * edge_valid;
        FFX_MIN16_F3 blended_seed = ffxLerp(rgb_p1_warp_t_mv, rgb_m1_warp_t_mv, t_seed);
        FfxFloat32   seed_valid   = (coverage > 0.0f) ? 1.0f : 0.0f;

        if (!padded_tensor_pixel)
        {
            StoreInpaintSeed(output_pixel, FfxFloat32x4(FfxFloat32x3(blended_seed) * seed_valid, seed_valid));
        }
    }

    //-------------------------------------------------------------------------
    // 6) Write network tensor
    //-------------------------------------------------------------------------
    WriteToInputTensor(tensor_pixel, rgb_m1_warp_t_mv, rgb_p1_warp_t_mv, rgb_m1_warp_t_flow, rgb_p1_warp_t_flow, dis_mask, depth, colourBytes,
                       m1FlowBytes, p1FlowBytes, mv_t_f30_m1);
}

#endif