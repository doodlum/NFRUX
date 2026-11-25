// SPDX-License-Identifier: MIT
// Arm's 01 downsample (ffx_frameinterpolation_01_downsample_of_colour_pass_fs.glsl) as a compute pass: the optical flow's
// colour, 1080 rows at most.
#version 460
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_GOOGLE_include_directive : enable

#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE 0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OF_COLOUR_P1                 1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION            2

#include "frameinterpolation/ffx_frameinterpolation_01_downsample_of_colour_pass.h"
layout(local_size_x = FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH, local_size_y = FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT, local_size_z = 1) in;
void main()
{
    downsample_of_colour(int32_t2(gl_GlobalInvocationID.xy));
}
