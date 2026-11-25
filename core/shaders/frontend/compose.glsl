// An interpolated frame put back where the game's frame is: distorted again (out(d) = interpolated(d + RG(d))), the no-warp
// mask and the UI over it, inside the generated region; the frame as it is around it. The UI rules are the postprocess pass's.
// SPDX-License-Identifier: MIT
#version 450
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : enable
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, std140) uniform Compose
{
    ivec2 outSize, origin, size, baseOrigin, uiOrigin, noWarpOrigin, fieldBase, fieldSize;
    vec2  scale, bias;  // field value to UV offset
    int   distort, uiKind, noWarp, swapRB;  // uiKind: -1 none, then the postprocess pass's
} u;
layout(set = 0, binding = 1) uniform texture2D interp;
layout(set = 0, binding = 2) uniform texture2D field;
layout(set = 0, binding = 3) uniform texture2D baseImage;  // what was interpolated, where the game draws it
layout(set = 0, binding = 4) uniform texture2D frameImage;
layout(set = 0, binding = 5) uniform texture2D uiImage;
layout(set = 0, binding = 6) uniform texture2D noWarpImage;
layout(set = 0, binding = 7, OUTPUT_IMG_FORMAT) uniform writeonly image2D dst;
#include "bilinear.h"
#define UI_BLOCK 2
float Opacity(ivec2 p) { const vec4 v = texelFetch(uiImage, u.uiOrigin + p, 0); return u.uiKind == 3 || u.uiKind == 5 ? v.a : v.r; }
void main()
{
    const ivec2 o = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(o, u.outSize)))
        return;
    const vec3  frame = texelFetch(frameImage, o, 0).rgb;
    const ivec2 p     = o - u.origin;
    vec3        c     = frame;
    if (all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, u.size)))
    {
        const vec3 base = texelFetch(baseImage, u.baseOrigin + p, 0).rgb;
        if (u.distort != 0)
        {
            const vec2 uv = (vec2(p) + 0.5) / vec2(u.size);
            c = Bilinear(interp, ivec2(0), u.size, uv + (Bilinear(field, u.fieldBase, u.fieldSize, uv).rg * u.scale + u.bias)).rgb;
        }
        else
            c = texelFetch(interp, p, 0).rgb;
        if (u.noWarp != 0 && texelFetch(noWarpImage, u.noWarpOrigin + p, 0).r > 0.5)
            c = base;
        const ivec2 block = p & ~ivec2(UI_BLOCK - 1);
        if (u.uiKind == 2 || u.uiKind == 3)
        {
            const float a  = Opacity(p);
            const vec3  ui = frame - (1.0 - a) * base;
            if (any(isnan(ui)) || any(isinf(ui)) || any(lessThanEqual(ui, vec3(-1.0))) || any(lessThan(ui, frame - 2.0 * base)))
                c = frame;
            else
                c = ui + (1.0 - a) * c;
        }
        else if (u.uiKind >= 0)
        {
            bool isUi = false;
            for (int i = 0; i < UI_BLOCK * UI_BLOCK; ++i)
            {
                const ivec2 q = min(block + ivec2(i % UI_BLOCK, i / UI_BLOCK), u.size - 1);
                if (u.uiKind >= 4)
                    isUi = isUi || Opacity(q) > 0.4;
                else
                {
                    const vec3 d = abs(texelFetch(frameImage, u.origin + q, 0).rgb - texelFetch(baseImage, u.baseOrigin + q, 0).rgb);
                    isUi = isUi || max(d.r, max(d.g, d.b)) > 0.07;
                }
            }
            if (isUi)
                c = frame;
        }
    }
    imageStore(dst, o, vec4(u.swapRB != 0 ? c.bgr : c, 1.0));
}
