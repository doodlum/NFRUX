// The frame to interpolate out of a host image: its region, and with a distortion field (DLSSG.BidirectionalDistortionField)
// taken to the undistorted space depth and motion vectors are in: at an undistorted position u the field's BA is the offset to
// the distorted one. SPDX-License-Identifier: MIT
#version 450
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : enable
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, std140) uniform Prepare { ivec2 base, size, fieldBase, fieldSize; vec2 scale, bias; ivec4 distort; } u;
layout(set = 0, binding = 1) uniform texture2D src;
layout(set = 0, binding = 2) uniform texture2D field;
layout(set = 0, binding = 3, rgba16f) uniform writeonly image2D dst;
#include "bilinear.h"
void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, u.size)))
        return;
    vec3 c;
    if (u.distort.x != 0)
    {
        const vec2 uv = (vec2(p) + 0.5) / vec2(u.size);
        const vec2 ba = Bilinear(field, u.fieldBase, u.fieldSize, uv).ba * u.scale + u.bias;
        c = Bilinear(src, u.base, u.size, uv + ba).rgb;
    }
    else
        c = texelFetch(src, u.base + p, 0).rgb;
    imageStore(dst, p, vec4(c, 1.0));
}
