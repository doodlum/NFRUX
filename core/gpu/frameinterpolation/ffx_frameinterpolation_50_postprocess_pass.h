/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_FFX_FRAMEINTERPOLATION_40_POSTPROCESS_PASS_H
#define GPU_FFX_FRAMEINTERPOLATION_40_POSTPROCESS_PASS_H

#define HALF_MIN 0.00006103515625f

#include "ffx_frameinterpolation_callbacks_glsl.h"
#include "quant.h"

// methods

FFX_MIN16_F max4(FFX_MIN16_F4 x)
{
    return max(max(max(x.x, x.y), x.z), x.w);
}

FFX_MIN16_F2 LoadMotionQMVD(int32_t2 pixel, out bool dynamic_oob_hole)
{
    FfxUInt32 packed = LoadWarpedMotionQDataTP1(pixel);
    dynamic_oob_hole = (packed == 0x7FFFFFFFu);
    float2 motion    = dqFloatMv(packed);
    motion += float2(0.001f) * step(0.5f, float(packed));
    motion *= float2(1.0f) / float2(MvDepthLaneSize());
    return FFX_MIN16_F2(motion) * FFX_MIN16_F(!dynamic_oob_hole);
}

// prior: the multi-frame network is trained with log(prior) added to its logits, [1 - t, t, 1 - t, t] (the SDK's nfru-mfg branch).
FFX_MIN16_F4 Stable_Softmax(FFX_MIN16_F4 tensor, FFX_MIN16_F4 prior)
{
    // Subtract the max value to help reduce overflow issues
    FFX_MIN16_F4 stable_ten     = tensor - max4(tensor);
    FFX_MIN16_F4 e_ten          = exp2(stable_ten * 1.44269504088896340736f) * prior;
    e_ten                      /= max4(e_ten);  // the largest is 1, as when the prior is in the logits
    FFX_MIN16_F  sum            = ((e_ten.x + e_ten.y) + e_ten.z) + e_ten.w;
    FFX_MIN16_F4 smax_activated = e_ten / (sum + HALF_MIN);
    return smax_activated;
}

float2 CroppedNetworkParamsUV(float2 uv)
{
    int32_t2 logical_dims   = max(int32_t2(1), MvDepthLaneSize() / 2);
    float2   logical_dims_f = float2(logical_dims);
    float2   params_pixel   = clamp(uv * logical_dims_f - 0.5f, float2(0.0f), logical_dims_f - 1.0f);
    return (params_pixel + 0.5f) * ParamsLaneSizeRcp();
}

// DLSS-G's UI tests decide per block of this
// many pixels square (it has 2 and 4): a pixel is UI when any pixel of its block is.
#ifndef FFX_FRAMEINTERPOLATION_UI_BLOCK
#define FFX_FRAMEINTERPOLATION_UI_BLOCK 2
#endif

// FFX_FRAMEINTERPOLATION_QUAD_UI (the compute pass): each quad of lanes is a 2x2 pixel block (the UI test's block),
// so the test loads each pixel once and ORs over the quad, where it loaded the block's four per pixel. Lanes off
// the display take its last pixel and store nothing, so that every lane of a quad is active.
#if FFX_FRAMEINTERPOLATION_QUAD_UI
#if FFX_FRAMEINTERPOLATION_UI_BLOCK != 2
#error the quad UI test takes 2x2 blocks
#endif
bool QuadAny(bool b)
{
    b = b || subgroupQuadSwapHorizontal(b);
    return b || subgroupQuadSwapVertical(b);
}
#endif

// The UI test by 2x2 blocks: every lane of the group tests its own pixel once, the block takes its four (the edge's missing pixels repeat others).
shared uint32_t s_ui[FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH * FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT];
bool UiPixel(int32_t2 p)
{
    if (any(greaterThanEqual(p, DisplaySize())) || UiKind() == 2u || UiKind() == 3u)
        return false;
    if (UiKind() == 4u || UiKind() == 5u)
        return LoadUiAlpha(p, UiKind() == 5u) > FFX_MIN16_F(0.4f);
    const FFX_MIN16_F3 d = abs(LoadUiSource(p).rgb - LoadCurrentBackbuffer(p));
    return max(d.r, max(d.g, d.b)) > FFX_MIN16_F(0.07f);
}
bool UiBlock()
{
    const uint32_t w = uint32_t(FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH), i = (gl_LocalInvocationID.y & ~1u) * w + (gl_LocalInvocationID.x & ~1u);
    return (s_ui[i] | s_ui[i + 1u] | s_ui[i + w] | s_ui[i + w + 1u]) != 0u;
}
void postprocess(int32_t2 output_pixel)
{
    //-------------------------------------------------------------------------
    // 1) Dispatch guard
    //-------------------------------------------------------------------------
    int32_t2 displaySize = DisplaySize();
#if FFX_FRAMEINTERPOLATION_QUAD_UI
    const bool store = all(lessThan(output_pixel, displaySize));
    output_pixel     = min(output_pixel, displaySize - 1);
#else
    const bool store = true;
    if (any(greaterThanEqual(output_pixel, displaySize)))
        return;
#endif

    //-------------------------------------------------------------------------
    // 2) Motion and OF sampling. Reprojection setup
    //-------------------------------------------------------------------------
    float2 uv = (float2(output_pixel.xy) + 0.5f) * DisplaySizeRcp();

    float2   mvDepthLanef = float2(MvDepthLaneSize());
    int32_t2 iMvCoord     = clamp(int32_t2(floor(uv * mvDepthLanef)), int32_t2(0), WarpedMotionQDataTP1Size() - int32_t2(1));

    // Read Output parameters - bilinear upsampling is done here. The graph params/gate are on the
    // params lane (padded preprocess lane / 2), so crop display UV to the logical half-resolution
    // content before sampling the physical padded image.
    float2       tensor_uv = CroppedNetworkParamsUV(uv);
    FFX_MIN16_F4 params    = Dequantize(FFX_MIN16_F4(SampleOutParamsTensor(tensor_uv)), QuantParamsSNORM().zw);

    // Read Vectors warped to "t" position - sampling nearest.
    FFX_MIN16_F2 flow_t_f30_p1    = SampleWarpedFilledFlowTM1(uv);
    bool         dynamic_oob_hole = false;  // Not used
    FFX_MIN16_F2 mv_t_f30_m1      = LoadMotionQMVD(iMvCoord, dynamic_oob_hole);

    // Calculate warped game mv positions
    float  timestep   = Timestep();
    float2 uv_m1_mv   = uv + float2(mv_t_f30_m1 * timestep);
    float2 uv_p1_mv   = uv - float2(mv_t_f30_m1 * (1.0f - timestep));
    float2 uv_m1_flow = uv - float2(flow_t_f30_p1 * timestep);
    float2 uv_p1_flow = uv + float2(flow_t_f30_p1 * (1.0f - timestep));

    bool uv_p1_mv_oob = (uv_p1_mv.x <= 0.0f || uv_p1_mv.x >= 1.0f) || (uv_p1_mv.y <= 0.0f || uv_p1_mv.y >= 1.0f);
    bool uv_m1_mv_oob = (uv_m1_mv.x <= 0.0f || uv_m1_mv.x >= 1.0f) || (uv_m1_mv.y <= 0.0f || uv_m1_mv.y >= 1.0f);

    //-------------------------------------------------------------------------
    // 3) Colour motion and OF warping
    //-------------------------------------------------------------------------
    FFX_MIN16_F3 rgb_m1_warp_t_mv   = SamplePreviousBackbuffer(uv_m1_mv);
    FFX_MIN16_F3 rgb_p1_warp_t_mv   = SampleCurrentBackbuffer(uv_p1_mv);
    FFX_MIN16_F3 rgb_m1_warp_t_flow = SamplePreviousBackbuffer(uv_m1_flow);
    FFX_MIN16_F3 rgb_p1_warp_t_flow = SampleCurrentBackbuffer(uv_p1_flow);

    //-------------------------------------------------------------------------
    // 4) Colour blending
    //-------------------------------------------------------------------------
    params = Stable_Softmax(params, FFX_MIN16_F4(1.0f - timestep, timestep, 1.0f - timestep, timestep));

    FFX_MIN16_F3 rgb_t = FFX_MIN16_F3(0.f);
    rgb_t += rgb_m1_warp_t_mv * params.x;
    rgb_t += rgb_p1_warp_t_mv * params.y;
    rgb_t += rgb_m1_warp_t_flow * params.z;
    rgb_t += rgb_p1_warp_t_flow * params.w;

    //-------------------------------------------------------------------------
    // 5) Inpaint holes
    //-------------------------------------------------------------------------
    // gated inpaint: only blend the resolved inpaint colour into actual holes (zero MV or both MV
    // warps OOB, excluding dynamic-OOB markers that the network handles itself)
    bool hole_to_inpaint =
        ((mv_t_f30_m1.x == FFX_MIN16_F(0.0f) && mv_t_f30_m1.y == FFX_MIN16_F(0.0f)) || (uv_m1_mv_oob && uv_p1_mv_oob)) && (!dynamic_oob_hole);
    if (hole_to_inpaint)
    {
        FFX_MIN16_F  gate    = ffxSaturate(Dequantize(FFX_MIN16_F(SampleOutParamsGateTensor(tensor_uv)), QuantParamsGateSNORM().zw));
        FFX_MIN16_F4 inpaint = FFX_MIN16_F4(SampleInpaintResult(uv));
        rgb_t                = ffxLerp(rgb_t, inpaint.rgb, inpaint.a * gate);
    }

    // the game's no-warp mask and its UI over the interpolated frame (as the v1 composite).
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_NO_WARP_MASK
    // Pixels the game marks not to warp (DLSS-G's no-warp mask): the current frame's, any UI composited over them.
    if (HasNoWarpMask() && LoadNoWarp(output_pixel))
        rgb_t = LoadCurrentBackbuffer(output_pixel);
#endif

    // Write alpha composited output
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_UI_SOURCE
    {
        const FFX_MIN16_F4 ui = LoadUiSource(output_pixel);
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_UI_ALPHA
        if (UiKind() == 2u || UiKind() == 3u)
        {
            // The frame with UI and the UI's opacity (DLSS-G's UI alpha, or a UI layer's alpha): the UI colour is
            // frame - (1 - a) x HUD-less, so over the interpolated scene the output is frame - (1 - a) x (HUD-less -
            // interpolated): where nothing moves, the frame exactly. Where that UI colour cannot be right (not
            // finite, -1 or below, or below frame - 2 x HUD-less) the pixel is taken as opaque UI, as DLSS-G does.
            const FFX_MIN16_F  a   = LoadUiAlpha(output_pixel, UiKind() == 3u);
            const FFX_MIN16_F3 cur = LoadCurrentBackbuffer(output_pixel);
            const FFX_MIN16_F3 uiColor = ui.rgb - (FFX_MIN16_F(1.0f) - a) * cur;
            if (any(isnan(uiColor)) || any(isinf(uiColor)) || any(lessThanEqual(uiColor, FFX_MIN16_F3(-1.0f))) ||
                any(lessThan(uiColor, ui.rgb - FFX_MIN16_F(2.0f) * cur)))
                rgb_t = ui.rgb;
            else
                rgb_t = uiColor + (FFX_MIN16_F(1.0f) - a) * rgb_t;  // not clamped, as DLSS-G (scRGB has negatives)
        }
        else if (UiKind() == 4u || UiKind() == 5u)
        {
            // No HUD-less frame, the frame interpolated with its UI, and the UI's opacity (red, or alpha for a UI
            // layer): the frame as it is where the opacity is above 0.4 anywhere in the block, as DLSS-G does.
#if FFX_FRAMEINTERPOLATION_QUAD_UI
            const bool isUi = QuadAny(LoadUiAlpha(output_pixel, UiKind() == 5u) > FFX_MIN16_F(0.4f));
#else
            const bool isUi = UiBlock();
#endif
            if (isUi)
                rgb_t = ui.rgb;
        }
        else
#endif
        {
            // The frame with UI: where it differs from the one interpolated (HUD-less), its pixel as it is, not warped.
            // DLSS-G's test: a channel off by more than 0.07 anywhere in the pixel's block, so grain or sharpening the
            // game adds after its HUD-less copy is not taken for UI, and text's soft edges go with its strokes.
#if FFX_FRAMEINTERPOLATION_QUAD_UI
            const FFX_MIN16_F3 d    = abs(ui.rgb - LoadCurrentBackbuffer(output_pixel));
            const bool         isUi = QuadAny(max(d.r, max(d.g, d.b)) > FFX_MIN16_F(0.07f));
#else
            const bool isUi = UiBlock();
#endif
            if (isUi)
                rgb_t = ui.rgb;
        }
    }
#endif

    if (!store)
        return;

    //-----------------------------------------------------------------------
    // 6) Output final colour
    //-------------------------------------------------------------------------
    StoreFrameinterpolationOutput(output_pixel, FFX_MIN16_F4(rgb_t, 1.0f));

    // When the SDK owns the previous-source ping-pong buffer, persist the current
    // backbuffer pixel so it can be sampled as colorTm1 on the next frame without
    // an extra copy job (see PREVIOUS_INTERPOLATION_SOURCE_BACKUP).
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_COLOR
    StorePreviousInterpolationSource(output_pixel, FFX_MIN16_F4(LoadCurrentBackbuffer(output_pixel), 1.0f));
#endif
}

#endif