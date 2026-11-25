// Bilinear filtering inside a region of an image, its edges clamped. By hand: the same on both APIs, and the region's
// neighbours never leak in. SPDX-License-Identifier: MIT
vec4 Bilinear(texture2D t, ivec2 base, ivec2 size, vec2 uv)
{
    const vec2  x  = uv * vec2(size) - 0.5;
    const vec2  f  = floor(x);
    const vec2  w  = x - f;
    const ivec2 i0 = clamp(ivec2(f), ivec2(0), size - 1), i1 = clamp(ivec2(f) + 1, ivec2(0), size - 1);
    const vec4  a  = texelFetch(t, base + i0, 0), b = texelFetch(t, base + ivec2(i1.x, i0.y), 0);
    const vec4  c  = texelFetch(t, base + ivec2(i0.x, i1.y), 0), d = texelFetch(t, base + i1, 0);
    const vec4  m  = a + (b - a) * w.x, n = c + (d - c) * w.x;
    return m + (n - m) * w.y;
}
