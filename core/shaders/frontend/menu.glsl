// Full-screen menu detection (DLSSG.MenuDetectionEnabled): counts, on a sparse grid, the pixels where the game's UI is mostly
// opaque. SPDX-License-Identifier: MIT
#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, std140) uniform Menu { ivec2 base, size; ivec4 k; } u;  // k: grid spacing, 1 = opacity in red
layout(set = 0, binding = 1) uniform texture2D ui;
layout(set = 0, binding = 2, r32ui) uniform uimage2D counts;  // covered, total; from ~0 (the first adds make 0)
void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy) * u.k.x + u.k.x / 2;
    if (any(greaterThanEqual(p, u.size)))
        return;
    const vec4 v = texelFetch(ui, u.base + p, 0);
    if ((u.k.y != 0 ? v.r : v.a) >= 0.5)
        imageAtomicAdd(counts, ivec2(0, 0), 1u);
    imageAtomicAdd(counts, ivec2(1, 0), 1u);
}
