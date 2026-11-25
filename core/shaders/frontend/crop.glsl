// A region of a host image (depth, motion vectors) into an image of NFRUX's, nearest texel when the sizes differ (dynamic resolution). For motion vectors
// without the camera's motion: that of the texel's depth added, in the vectors' units. SPDX-License-Identifier: MIT
#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, std140) uniform Crop { ivec2 base, size, outSize, depthBase, depthSize; int camera, pad; vec2 toUnits, pad2; mat4 clipToPrevClip; } u;
layout(set = 0, binding = 1) uniform texture2D src;
layout(set = 0, binding = 2, OUTPUT_IMG_FORMAT) uniform writeonly image2D dst;
layout(set = 0, binding = 3) uniform texture2D depth;
void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, u.outSize)))
        return;
    const ivec2 s = (p * u.size + u.size / 2) / u.outSize;
    vec4 v = texelFetch(src, u.base + s, 0);
    if (u.camera != 0)
    {
        const ivec2 d  = (p * u.depthSize + u.depthSize / 2) / u.outSize;
        const vec2  uv = (vec2(s) + 0.5) / vec2(u.size);
        const vec4  c  = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, texelFetch(depth, u.depthBase + d, 0).r, 1.0);
        const vec4  q  = ((u.clipToPrevClip[0] * c.x + u.clipToPrevClip[1] * c.y) + u.clipToPrevClip[2] * c.z) + u.clipToPrevClip[3];
        if (q.w > 1e-6)
            v.xy += (vec2(q.x / q.w * 0.5 + 0.5, 0.5 - q.y / q.w * 0.5) - uv) * u.toUnits;
    }
    imageStore(dst, p, v);
}