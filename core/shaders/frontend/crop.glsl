// A region of a host image (depth, motion vectors) into an image of exactly its size. SPDX-License-Identifier: MIT
#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, std140) uniform Crop { ivec2 base; ivec2 size; } u;
layout(set = 0, binding = 1) uniform texture2D src;
layout(set = 0, binding = 2, OUTPUT_IMG_FORMAT) uniform writeonly image2D dst;
void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(p, u.size)))
        imageStore(dst, p, texelFetch(src, u.base + p, 0));
}