// SPDX-License-Identifier: MIT
// DeckyUpscaler: the 03 downsample (the colour at the preprocess lane) and the Arm optical flow engine's pyramid level 1
// (ffx_opticalflow_arm_pyramid.glsl's FromColour: Y = (R + 2G + B) / 4 over a 2x2 box) in one pass, so the frame's
// colour is read from memory once for both: each workgroup takes a 16 x 16 tile of the lane and the level 1 texels over
// the same part of the frame. Each value as the two passes compute it.
#version 460
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive : enable

#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OF_COLOUR_P1               0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1            1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_PREPROCESS_COLOUR_P1       3
#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION          6

#include "frameinterpolation/ffx_frameinterpolation_03_downsample_colour_pass.h"

// The pyramid's (ffx_opticalflow_image_common.h's OpticalFlowImageConstants, level 1's), its colour and its image.
layout(set = 0, binding = 8, std140) uniform Uniforms
{
    FfxInt32x2   _LevelDims;
    FfxInt32x2   _SrcDims;
    FfxInt32x2   _CellDims;
    FfxInt32x2   _CoarseDims;
    FfxInt32x2   _HintDims;
    FfxFloat32x2 _OutScale;
    FfxInt32     _LevelRow;
    FfxInt32     _FlowRow;
    FfxInt32     _CoarseRow;
    FfxInt32     _Level;
    FfxUInt32    _LambdaHint;
    FfxUInt32    _LambdaSeed;
    FfxInt32     _LevelCount;
    FfxUInt32    _SettledCost;
    FfxInt32     _Final;
    FfxInt32     _Fsr3;
    FfxInt32     _Unorm8;
    FfxInt32     _Pad2;
    FfxInt32x4   _Levels[5];
}
ubo;
layout(set = 0, binding = 4) uniform highp texture2D r_color_tp1;  // the same image as binding 0
layout(set = 0, binding = 2, r8) uniform image2D rw_pyramid_tp1;

FfxFloat32x3 Rgb(FfxInt32x2 p)
{
    return texelFetch(r_color_tp1, clamp(p, FfxInt32x2(0), ubo._SrcDims - FfxInt32x2(1)), 0).rgb;
}

FfxUInt32 FromColour(FfxInt32x2 base)
{
    if (ubo._Unorm8 != 0)
    {
        FfxUInt32x3 sum = FfxUInt32x3(0u);
        [[unroll]] for (FfxInt32 i = 0; i < 4; ++i)
            sum += FfxUInt32x3(floor(Rgb(base + FfxInt32x2(i & 1, i >> 1)) * 255.0f + 0.5f));
        return (sum.r + 2u * sum.g + sum.b + 8u) / 16u;
    }
    FfxUInt32x3 sum = FfxUInt32x3(0u);
    [[unroll]] for (FfxInt32 i = 0; i < 4; ++i)
        sum += FfxUInt32x3(floor(clamp(Rgb(base + FfxInt32x2(i & 1, i >> 1)), 0.0f, 1.0f) * 65535.0f + 0.5f));
    const FfxUInt32 weighted = sum.r + 2u * sum.g + sum.b;
    return (weighted * 255u + 524280u) / 1048560u;
}

// Level 1's first texel over lane tile t (16 lane pixels a tile): ceil(t x 16 x level / lane), at most the level.
FfxInt32x2 PyramidStart(FfxInt32x2 tile, FfxInt32x2 lane)
{
    return min(ubo._LevelDims, (tile * 16 * ubo._LevelDims + lane - 1) / lane);
}

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main()
{
    downsample_colour(int32_t2(gl_GlobalInvocationID.xy));

    const FfxInt32x2 lane = MvDepthLaneSize();
    const FfxInt32x2 tile = FfxInt32x2(gl_WorkGroupID.xy);
    const FfxInt32x2 p0   = PyramidStart(tile, lane);
    // The last tiles of a row or column take the rest of the level.
    const FfxInt32x2 lastTile = (lane + 15) / 16 - 1;
    const FfxInt32x2 p1 = FfxInt32x2(tile.x == lastTile.x ? ubo._LevelDims.x : PyramidStart(tile + FfxInt32x2(1, 0), lane).x,
                                     tile.y == lastTile.y ? ubo._LevelDims.y : PyramidStart(tile + FfxInt32x2(0, 1), lane).y);
    for (FfxInt32 y = p0.y + FfxInt32(gl_LocalInvocationID.y); y < p1.y; y += 16)
        for (FfxInt32 x = p0.x + FfxInt32(gl_LocalInvocationID.x); x < p1.x; x += 16)
        {
            const FfxUInt32 v = FromColour(FfxInt32x2(x, y) * 2);
            imageStore(rw_pyramid_tp1, FfxInt32x2(x, ubo._LevelRow + y), FfxFloat32x4(FfxFloat32(v) / 255.0f, 0.0f, 0.0f, 0.0f));
        }
}
