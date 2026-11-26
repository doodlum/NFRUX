// One FasterNet block of FSR FG4's stage 2 in one pass, the generic tier (core/mlfg; no 16-bit types): y = q8(project(
// q8(relu(expand(cat(q8(pconv3x3(x[:P])), x[P:]))))) + x), C = MLFG_C (48 / 64), P = MLFG_P (16 / 32). AMD FidelityFX SDK
// v2.4.0, MIT; spec tools/ffx_extract/mlfg_net.py ('amd'). A group of 64 threads makes 8 x 8 pixels, a thread its pixel:
// the pconv patch, then the 64 pixels' expand outputs (fp8 values as fp16 pairs, in the patch's memory), and their
// cat(pconv, x[P:]) are in groupshared memory (32 KB). Each step's work items are a row of the tile (8 pixels) by 8
// outputs, spread over the 64 threads (8 x 8 sums a weight load serves). fp32 sums in a fixed order (the same on every API);
// the order differs from the three separate kMlfgConv layers. Weights as kMlfgConv's ([tap][cin][coPad] fp32, then the bias).
//
// Reads   t1 x (pitch C), t3 weights; writes u5 y (pitch C). c_size: width, height; c_w: weight offsets (floats) of pconv,
// expand, project; c_pad: their coPad.
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

cbuffer Constants : register(b0)
{
    int4 c_size;  // width, height, 0, 0
    int4 c_w;     // weight offsets (floats): pconv, expand, project, 0
    int4 c_pad;   // coPad: pconv, expand, project, 0
};
ByteAddressBuffer   t_x : register(t1);
ByteAddressBuffer   t_w : register(t3);
RWByteAddressBuffer u_y : register(u5);

#define C MLFG_C
#define P MLFG_P
#define E (2 * C)

float F8Round(float x)  // as mlfg_conv.hlsl (torch's float8_e4m3fn cast)
{
    const uint u = asuint(x);
    const uint s = u & 0x80000000u, a = u & 0x7FFFFFFFu;
    if (a >= 0x7F800000u)
        return asfloat(0x7FC00000u);
    const uint e = a >> 23;
    if (e >= 121u)
    {
        uint       q   = a >> 20;
        const uint rem = a & 0xFFFFFu;
        if (rem > 0x80000u || (rem == 0x80000u && (q & 1u) != 0u))
            q += 1u;
        const uint r = q << 20;
        return r > 0x43E00000u ? asfloat(0x7FC00000u) : asfloat(s | r);
    }
    if (e == 0u)
        return asfloat(s);
    const uint m = (a & 0x7FFFFFu) | 0x800000u, shift = 141u - e;
    if (shift >= 26u)
        return asfloat(s);
    uint       q   = m >> shift;
    const uint rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (q & 1u) != 0u))
        q += 1u;
    return asfloat(s | asuint(float(q) * 0.001953125));
}
float W(uint i) { return asfloat(t_w.Load(i << 2)); }
float4 W4(uint i) { return asfloat(t_w.Load4(i << 2)); }

// 32 KB: [0, 4096) the pconv patch (floats), later the expand outputs (fp16 pairs); [4096, 8192) cat(pconv, x[P:]).
#define HB 4096u
groupshared uint s_m[8192];

[numthreads(64, 1, 1)] void main(uint3 gid : SV_GroupID, uint li : SV_GroupIndex)
{
    const int2 tile = int2(gid.xy) * 8;
    const int2 me   = tile + int2(li % 8u, li / 8u);
    const int2 size = c_size.xy;
    [loop] for (uint i = li; i < 100u * uint(P / 2); i += 64u)
    {
        const uint p = i / uint(P / 2), c = (i % uint(P / 2)) * 2u;
        const int2 q = tile - 1 + int2(p % 10u, p / 10u);
        const float2 v = Inside(q, size) ? UnpackF16x2(t_x.Load(((uint(q.y) * uint(size.x) + uint(q.x)) * uint(C) + c) << 1)) : float2(0, 0);
        s_m[p * uint(P) + c] = asuint(v.x), s_m[p * uint(P) + c + 1u] = asuint(v.y);
    }
    const bool inside = Inside(me, size);
    const uint pix = inside ? uint(me.y) * uint(size.x) + uint(me.x) : 0u;
    const uint hb  = HB + li * uint(C);
    [loop] for (uint c = uint(P); c < uint(C); c += 2u)
    {
        const float2 v = inside ? UnpackF16x2(t_x.Load((pix * uint(C) + c) << 1)) : float2(0, 0);
        s_m[hb + c] = asuint(v.x), s_m[hb + c + 1u] = asuint(v.y);
    }
    GroupMemoryBarrierWithGroupSync();
    // pconv: items (row, 8 outputs) into cat(...)[:P].
    [loop] for (uint it = li; it < 8u * uint(P / 8); it += 64u)
    {
        const uint row = it % 8u, o = (it / 8u) * 8u;
        float a[8][8];
        [unroll] for (uint q = 0u; q < 8u; ++q) [unroll] for (uint k = 0u; k < 8u; ++k) a[q][k] = 0.0;
        [loop] for (uint t = 0u; t < 9u; ++t)
        {
            const uint xb = ((row + t / 3u) * 10u + t % 3u) * uint(P);
            [loop] for (uint ci = 0u; ci < uint(P); ++ci)
            {
                const uint   wa = uint(c_w.x) + (t * uint(P) + ci) * uint(c_pad.x) + o;
                const float4 w0 = W4(wa), w1 = W4(wa + 4u);
                [unroll] for (uint q = 0u; q < 8u; ++q)
                {
                    const float xv = asfloat(s_m[xb + q * uint(P) + ci]);
                    a[q][0] += xv * w0.x; a[q][1] += xv * w0.y; a[q][2] += xv * w0.z; a[q][3] += xv * w0.w;
                    a[q][4] += xv * w1.x; a[q][5] += xv * w1.y; a[q][6] += xv * w1.z; a[q][7] += xv * w1.w;
                }
            }
        }
        const uint ba = uint(c_w.x) + 9u * uint(P) * uint(c_pad.x) + o;
        [unroll] for (uint q = 0u; q < 8u; ++q)
            [unroll] for (uint k = 0u; k < 8u; ++k)
                s_m[HB + (row * 8u + q) * uint(C) + o + k] = asuint(F8Round(a[q][k] + W(ba + k)));
    }
    GroupMemoryBarrierWithGroupSync();  // the patch is read: its memory takes the expand outputs
    // expand (ReLU, fp8): items (row, 8 outputs), as fp16 pairs [pixel][E / 2].
    const uint eb = uint(c_w.y) + uint(C) * uint(c_pad.y);
    [loop] for (uint it2 = li; it2 < 8u * uint(E / 8); it2 += 64u)
    {
        const uint row = it2 % 8u, e0 = (it2 / 8u) * 8u;
        float a[8][8];
        [unroll] for (uint q = 0u; q < 8u; ++q) [unroll] for (uint k = 0u; k < 8u; ++k) a[q][k] = 0.0;
        [loop] for (uint ci = 0u; ci < uint(C); ++ci)
        {
            const uint   wa = uint(c_w.y) + ci * uint(c_pad.y) + e0;
            const float4 w0 = W4(wa), w1 = W4(wa + 4u);
            [unroll] for (uint q = 0u; q < 8u; ++q)
            {
                const float xv = asfloat(s_m[HB + (row * 8u + q) * uint(C) + ci]);
                a[q][0] += xv * w0.x; a[q][1] += xv * w0.y; a[q][2] += xv * w0.z; a[q][3] += xv * w0.w;
                a[q][4] += xv * w1.x; a[q][5] += xv * w1.y; a[q][6] += xv * w1.z; a[q][7] += xv * w1.w;
            }
        }
        [unroll] for (uint q = 0u; q < 8u; ++q)
            [unroll] for (uint k = 0u; k < 8u; k += 2u)
                s_m[(row * 8u + q) * uint(E / 2) + (e0 + k) / 2u] = PackF16x2(F8Round(max(a[q][k] + W(eb + e0 + k), 0.0)), F8Round(max(a[q][k + 1u] + W(eb + e0 + k + 1u), 0.0)));
    }
    GroupMemoryBarrierWithGroupSync();
    // project + x: items (row, 8 outputs).
    const uint jb = uint(c_w.z) + uint(E) * uint(c_pad.z);
    [loop] for (uint it3 = li; it3 < 8u * uint(C / 8); it3 += 64u)
    {
        const uint row = it3 % 8u, j0 = (it3 / 8u) * 8u;
        float a[8][8];
        [unroll] for (uint q = 0u; q < 8u; ++q) [unroll] for (uint k = 0u; k < 8u; ++k) a[q][k] = 0.0;
        [loop] for (uint e = 0u; e < uint(E); e += 2u)
        {
            const uint   wa = uint(c_w.z) + e * uint(c_pad.z) + j0, wb = wa + uint(c_pad.z);
            const float4 w0 = W4(wa), w1 = W4(wa + 4u), v0 = W4(wb), v1 = W4(wb + 4u);
            [unroll] for (uint q = 0u; q < 8u; ++q)
            {
                const float2 hv = UnpackF16x2(s_m[(row * 8u + q) * uint(E / 2) + e / 2u]);
                a[q][0] += hv.x * w0.x; a[q][1] += hv.x * w0.y; a[q][2] += hv.x * w0.z; a[q][3] += hv.x * w0.w;
                a[q][4] += hv.x * w1.x; a[q][5] += hv.x * w1.y; a[q][6] += hv.x * w1.z; a[q][7] += hv.x * w1.w;
                a[q][0] += hv.y * v0.x; a[q][1] += hv.y * v0.y; a[q][2] += hv.y * v0.z; a[q][3] += hv.y * v0.w;
                a[q][4] += hv.y * v1.x; a[q][5] += hv.y * v1.y; a[q][6] += hv.y * v1.z; a[q][7] += hv.y * v1.w;
            }
        }
        [unroll] for (uint q = 0u; q < 8u; ++q)
        {
            const int2 op = tile + int2(q, row);
            if (!Inside(op, size))
                continue;
            const uint pp = uint(op.y) * uint(size.x) + uint(op.x);
            [unroll] for (uint k = 0u; k < 8u; k += 2u)
            {
                const float2 xr = UnpackF16x2(t_x.Load((pp * uint(C) + j0 + k) << 1));
                const float  r0 = F8Round((a[q][k] + W(jb + j0 + k)) + xr.x), r1 = F8Round((a[q][k + 1u] + W(jb + j0 + k + 1u)) + xr.y);
                u_y.Store((pp * uint(C) + j0 + k) << 1, PackF16x2(r0, r1));
            }
        }
    }
}