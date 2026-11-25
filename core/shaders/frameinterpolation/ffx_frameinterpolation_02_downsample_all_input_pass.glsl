// SPDX-License-Identifier: MIT
// Arm's 02 downsample (ffx_frameinterpolation_02_downsample_all_input_pass_fs.glsl) as a compute pass:
// the colour, depth and motion at the preprocess lane (depth/motion at 540 rows or more), stored as UAVs.
#version 460
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_GOOGLE_include_directive : enable

#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OF_COLOUR_P1               0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1            1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1           2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_PREPROCESS_COLOUR_P1       3
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DEPTH_TM1_NEXT             4
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_REDUCED_MOTION_P1          5
#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION          6
// The pass header compiles its body only for the render-target variant's names.
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT    0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_REDUCED_MOTION_P1 0

#include "frameinterpolation/ffx_frameinterpolation_02_downsample_all_input_pass.h"

layout(local_size_x = FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH, local_size_y = FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT, local_size_z = 1) in;
void main()
{
    downsample_all_input(int32_t2(gl_GlobalInvocationID.xy));
}
