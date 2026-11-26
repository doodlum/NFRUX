// FSR FG4 network layers on any GPU (D3D12 FL12 / Vulkan 1.1, no 16-bit types): NFRUX's "generic" tier of the ML
// runtime (core/mlfg), and the reference the faster tiers are compared with. AMD FidelityFX SDK v2.4.0, MIT; spec:
// tools/ffx_extract/mlfg_net.py, precision 'amd' (fp16 activations, fp32 sums, fp8 e4m3 stores in stage 2).
// One shader, four builds: MLFG_CONV_K / MLFG_CONV_S = 3/1 (padding 1), 2/2, 1/1, and MLFG_CONV_T (transposed 2x2/2).
//
// Reads   t1 src0    NHWC fp16 tensor (mlfg_common.hlsli layout), c_in.xy = its size, pitch c_in.z, first channel c_in2.x
//         t2 src1    input channels c >= split come from here (channel c - split + c_in2.z, pitch c_in2.y): the
//                    FasterNet concatenation cat(pconv(x[:P]), x[P:]); bind src0 again when unused
//         t3 weights fp32 [tap][cin][coPad] (tap = ky * K + kx; transposed: dy * 2 + dx), then the bias [coPad]
//         t4 add     added before the activation (residual, skip), c_add.x = pitch (0: none), channel c_add.y
// Writes  u5 out     NHWC fp16, c_out.xy its size, pitch c_out.z, first channel c_out.w; channels [cout, writeEnd) 0
// out = round(act(sum + bias + add)), sum over taps and input channels in fp32 (strict: no fused multiply-add, the same
// order on every GPU and API), act none / LeakyReLU 0.2 / ReLU, round fp16 or fp8 e4m3 (stored as its fp16 value).
// Zero padding at the tensor's edges (convolutions over the whole allocated tensor, as mlfg_net). A group is 8 x 8
// output pixels by 8 output channels (group z); input channels are staged 8 at a time in groupshared memory.
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

cbuffer Constants : register(b0)
{
    int4 c_in;   // src width, height, src0 pitch, split (channels below it from src0)
    int4 c_in2;  // src0 first channel, src1 pitch, src1 first channel, cin
    int4 c_out;  // out width, height, pitch, first channel
    int4 c_cfg;  // cout, act (0 none, 1 leaky 0.2, 2 relu), round (0 fp16, 1 fp8), write end (even; >= cout)
    int4 c_add;  // add pitch (0: none), add first channel, weight offset (floats), coPad
};
ByteAddressBuffer   t_src0 : register(t1);
ByteAddressBuffer   t_src1 : register(t2);
ByteAddressBuffer   t_w : register(t3);
ByteAddressBuffer   t_add : register(t4);
RWByteAddressBuffer u_out : register(u5);

// fp32 -> fp8 e4m3fn value, as torch's float8_e4m3fn cast: round to nearest even (subnormals 2^-9 apart), what rounds
// above 448 is NaN (no infinity in the format), NaN and infinity NaN.
float F8Round(float x)
{
    const uint u = asuint(x);
    const uint s = u & 0x80000000u, a = u & 0x7FFFFFFFu;
    if (a >= 0x7F800000u)
        return asfloat(0x7FC00000u);
    const uint e = a >> 23;
    if (e >= 121u)  // 2^-6 and up: 3 mantissa bits
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
    // Subnormal: units of 2^-9. a = m24 * 2^(e - 150), so a / 2^-9 = m24 >> (141 - e).
    const uint m = (a & 0x7FFFFFu) | 0x800000u, shift = 141u - e;
    if (shift >= 26u)
        return asfloat(s);  // below 2^-11: 0
    uint       q   = m >> shift;
    const uint rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (q & 1u) != 0u))
        q += 1u;
    return asfloat(s | asuint(float(q) * 0.001953125));
}

#ifdef MLFG_CONV_T
#define TIN 4
#else
#define TIN (7 * MLFG_CONV_S + MLFG_CONV_K)
#define PAD (MLFG_CONV_K == 3 ? 1 : 0)
#endif
#define CI 8
groupshared float s_x[TIN * TIN * CI];

float LoadIn(int2 p, int c)
{
    if (!Inside(p, c_in.xy) || c >= c_in2.w)
        return 0.0;
    const uint base = (uint(p.y) * uint(c_in.x) + uint(p.x));
    if (c < c_in.w)
        return LoadHalf(t_src0, base * uint(c_in.z) + uint(c_in2.x + c));
    return LoadHalf(t_src1, base * uint(c_in2.y) + uint(c - c_in.w + c_in2.z));
}

[numthreads(8, 8, 1)] void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID, uint li : SV_GroupIndex)
{
    const int2 o   = int2(gid.xy * 8u + lid.xy);
    const int  co0 = int(gid.z) * 8;
#ifdef MLFG_CONV_T
    const int2 origin = int2(gid.xy * 4u);
    const int  tap    = (o.y & 1) * 2 + (o.x & 1);
    const int2 ip     = (int2(lid.xy) >> 1);
#else
    const int2 origin = int2(gid.xy * 8u) * MLFG_CONV_S - PAD;
#endif
    const int  cin  = c_in2.w;
    const uint coP  = uint(c_add.w);
    const uint wOff = uint(c_add.z);
    float acc[8];
    [unroll] for (uint j = 0u; j < 8u; ++j)
        acc[j] = 0.0;
    [loop] for (int c0 = 0; c0 < cin; c0 += CI)
    {
        GroupMemoryBarrierWithGroupSync();
        [loop] for (uint i = li; i < uint(TIN * TIN * CI); i += 64u)
        {
            const uint p = i / uint(CI), c = i % uint(CI);
            s_x[i] = LoadIn(origin + int2(p % uint(TIN), p / uint(TIN)), c0 + int(c));
        }
        GroupMemoryBarrierWithGroupSync();
        const int n = min(CI, cin - c0);
#ifdef MLFG_CONV_T
        {
            const uint xb = (uint(ip.y) * uint(TIN) + uint(ip.x)) * uint(CI);
            [loop] for (int ci = 0; ci < n; ++ci)
            {
                const float x  = s_x[xb + uint(ci)];
                const uint  wa = (wOff + (uint(tap) * uint(cin) + uint(c0 + ci)) * coP + uint(co0)) * 4u;
                const float4 w0 = asfloat(t_w.Load4(wa)), w1 = asfloat(t_w.Load4(wa + 16u));
                acc[0] = acc[0] + x * w0.x; acc[1] = acc[1] + x * w0.y; acc[2] = acc[2] + x * w0.z; acc[3] = acc[3] + x * w0.w;
                acc[4] = acc[4] + x * w1.x; acc[5] = acc[5] + x * w1.y; acc[6] = acc[6] + x * w1.z; acc[7] = acc[7] + x * w1.w;
            }
        }
#else
        [loop] for (int t = 0; t < MLFG_CONV_K * MLFG_CONV_K; ++t)
        {
            const int2 q  = int2(lid.xy) * MLFG_CONV_S + int2(t % MLFG_CONV_K, t / MLFG_CONV_K);
            const uint xb = (uint(q.y) * uint(TIN) + uint(q.x)) * uint(CI);
            float p[8];  // the tap's 8 channels summed first, then added: blocked sums, closer to exact than one long chain
            [unroll] for (uint j = 0u; j < 8u; ++j)
                p[j] = 0.0;
            [loop] for (int ci = 0; ci < n; ++ci)
            {
                const float x  = s_x[xb + uint(ci)];
                const uint  wa = (wOff + (uint(t) * uint(cin) + uint(c0 + ci)) * coP + uint(co0)) * 4u;
                const float4 w0 = asfloat(t_w.Load4(wa)), w1 = asfloat(t_w.Load4(wa + 16u));
                p[0] = p[0] + x * w0.x; p[1] = p[1] + x * w0.y; p[2] = p[2] + x * w0.z; p[3] = p[3] + x * w0.w;
                p[4] = p[4] + x * w1.x; p[5] = p[5] + x * w1.y; p[6] = p[6] + x * w1.z; p[7] = p[7] + x * w1.w;
            }
            [unroll] for (uint m = 0u; m < 8u; ++m)
                acc[m] = acc[m] + p[m];
        }
#endif
    }
    if (any(o >= c_out.xy))
        return;
    const int  taps = MLFG_CONV_K * MLFG_CONV_K;
    const uint ba   = (wOff + uint(taps * cin) * coP + uint(co0)) * 4u;
    const float4 b0 = asfloat(t_w.Load4(ba)), b1 = asfloat(t_w.Load4(ba + 16u));
    const float bias[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    const uint pix = uint(o.y) * uint(c_out.x) + uint(o.x);
    float v[8];
    [unroll] for (uint k = 0u; k < 8u; ++k)
    {
        const int c = co0 + int(k);
        float     y = acc[k] + bias[k];
        if (c_add.x != 0 && c < c_cfg.x)
            y = y + LoadHalf(t_add, pix * uint(c_add.x) + uint(c_add.y + c));
        if (c_cfg.y == 1)
            y = y > 0.0 ? y : y * 0.2;
        else if (c_cfg.y == 2)
            y = max(y, 0.0);
        y    = c_cfg.z == 1 ? F8Round(y) : F16Round(y);
        v[k] = c < c_cfg.x ? y : 0.0;
    }
    const uint ob = pix * uint(c_out.z) + uint(c_out.w + co0);
    [unroll] for (uint m = 0u; m < 8u; m += 2u)
        if (co0 + int(m) < c_cfg.w)
            StoreHalf2(u_out, ob + m, v[m], v[m + 1u]);
}
