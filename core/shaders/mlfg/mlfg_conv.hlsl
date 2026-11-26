// FSR FG4 network layers on any GPU (D3D12 FL12 / Vulkan 1.1, no 16-bit types): NFRUX's "generic" tier of the ML
// runtime (core/mlfg), and the reference the faster tiers are compared with. AMD FidelityFX SDK v2.4.0, MIT; spec:
// tools/ffx_extract/mlfg_net.py, precision 'amd' (fp16 activations, fp32 sums, fp8 e4m3 stores in stage 2).
// Builds: MLFG_CONV_K / MLFG_CONV_S = 3/1 (padding 1), 2/2, 1/1, or MLFG_CONV_T (transposed 2x2/2); a group of 64
// threads makes MLFG_TW x MLFG_TH output pixels by 8 output channels (group z), the input channels split in MLFG_KS
// slices (small layers: more groups in flight) summed in a fixed order at the end.
//
// Reads   t1 src0    NHWC fp16 tensor (mlfg_common.hlsli layout), c_in.xy = its size, pitch c_in.z, first channel c_in2.x
//         t2 src1    input channels c >= split come from here (channel c - split + c_in2.z, pitch c_in2.y): the
//                    FasterNet concatenation cat(pconv(x[:P]), x[P:]); bind src0 again when unused
//         t3 weights fp32 [tap][cin][coPad] (tap = ky * K + kx; transposed: dy * 2 + dx), then the bias [coPad]
//         t4 add     added before the activation (residual, skip), c_add.x = pitch (0: none), channel c_add.y
// Writes  u5 out     NHWC fp16, c_out.xy its size, pitch c_out.z, first channel c_out.w; channels [cout, writeEnd) 0
// out = round(act(sum + bias + add)), sum over taps and input channels in fp32 (strict: no fused multiply-add, the same
// order on every GPU and API: per 8-channel chunk and tap a partial sum, partials added in order), act none /
// LeakyReLU 0.2 / ReLU, round fp16 or fp8 e4m3 (stored as its fp16 value). Zero padding at the tensor's edges.
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

#define TW MLFG_TW
#define TH MLFG_TH
#define KS MLFG_KS
#define TPS (64 / KS)              // threads a slice
#define PX (TW * TH / TPS)         // pixels a thread (consecutive in x)
#ifdef MLFG_CONV_T
#define TINW (TW / 2)
#define TINH (TH / 2)
#define TAPS 4
#else
#define TINW ((TW - 1) * MLFG_CONV_S + MLFG_CONV_K)
#define TINH ((TH - 1) * MLFG_CONV_S + MLFG_CONV_K)
#define PAD (MLFG_CONV_K == 3 ? 1 : 0)
#define TAPS (MLFG_CONV_K * MLFG_CONV_K)
#endif
#define CI 8
#define PS (CI * KS + 1)  // a staged pixel's floats: one more than its channels, so a warp's reads spread over the banks
groupshared float s_x[TINW * TINH * PS];
#if KS > 1
groupshared float s_red[(KS - 1) * TPS * PX * 8];
#endif

float LoadIn(int2 p, int c)
{
    if (!Inside(p, c_in.xy) || c >= c_in2.w)
        return 0.0;
    const uint base = (uint(p.y) * uint(c_in.x) + uint(p.x));
    if (c < c_in.w)
        return LoadHalf(t_src0, base * uint(c_in.z) + uint(c_in2.x + c));
    return LoadHalf(t_src1, base * uint(c_in2.y) + uint(c - c_in.w + c_in2.z));
}

[numthreads(64, 1, 1)] void main(uint3 gid : SV_GroupID, uint li : SV_GroupIndex)
{
    const uint slice = li / uint(TPS), lt = li % uint(TPS);
    const int2 tile  = int2(gid.xy) * int2(TW, TH);
    // The thread's pixels: lt, lt + TPS, ... of the tile in row order (a warp reads neighbouring pixels: no bank conflicts).
    // TW divides TPS, so every pixel of a thread has the same x, and the same y parity: one tap in the transposed conv.
    const int2 first = int2(lt % uint(TW), lt / uint(TW));
#define PIX(q) (first + int2(0, int(q) * (TPS / TW)))
    const int  co0   = int(gid.z) * 8;
#ifdef MLFG_CONV_T
    const int2 origin = tile / 2;
#else
    const int2 origin = tile * MLFG_CONV_S - PAD;
#endif
    const int  cin  = c_in2.w;
    const uint coP  = uint(c_add.w);
    const uint wOff = uint(c_add.z);
    float acc[PX][8];
    [unroll] for (uint q = 0u; q < uint(PX); ++q)
        [unroll] for (uint j = 0u; j < 8u; ++j)
            acc[q][j] = 0.0;
    [loop] for (int cb = 0; cb < cin; cb += CI * KS)
    {
        GroupMemoryBarrierWithGroupSync();
        [loop] for (uint i = li; i < uint(TINW * TINH * CI * KS); i += 64u)
        {
            const uint p = i / uint(CI * KS), c = i % uint(CI * KS);
            s_x[p * uint(PS) + c] = LoadIn(origin + int2(p % uint(TINW), p / uint(TINW)), cb + int(c));
        }
        GroupMemoryBarrierWithGroupSync();
        const int c0 = cb + int(slice) * CI;
        const int n  = min(CI, cin - c0);  // <= 0: a slice past the last channel adds zeros
#ifdef MLFG_CONV_T
        {
            // One tap a pixel, by its position's parity, the same for all of the thread's pixels.
            const uint t = uint((first.y & 1) * 2 + (first.x & 1));
            uint xb[PX];
            [unroll] for (uint q = 0u; q < uint(PX); ++q)
            {
                const int2 s = PIX(q) >> 1;
                xb[q] = (uint(s.y) * uint(TINW) + uint(s.x)) * uint(PS) + slice * uint(CI);
            }
            float p[PX][8];
            [unroll] for (uint q1 = 0u; q1 < uint(PX); ++q1)
                [unroll] for (uint j = 0u; j < 8u; ++j)
                    p[q1][j] = 0.0;
            [loop] for (int ci = 0; ci < n; ++ci)
            {
                const uint   wa = (wOff + (t * uint(cin) + uint(c0 + ci)) * coP + uint(co0)) * 4u;
                const float4 w0 = asfloat(t_w.Load4(wa)), w1 = asfloat(t_w.Load4(wa + 16u));
                [unroll] for (uint q = 0u; q < uint(PX); ++q)
                {
                    const float x = s_x[xb[q] + uint(ci)];
                    p[q][0] = p[q][0] + x * w0.x; p[q][1] = p[q][1] + x * w0.y; p[q][2] = p[q][2] + x * w0.z; p[q][3] = p[q][3] + x * w0.w;
                    p[q][4] = p[q][4] + x * w1.x; p[q][5] = p[q][5] + x * w1.y; p[q][6] = p[q][6] + x * w1.z; p[q][7] = p[q][7] + x * w1.w;
                }
            }
            [unroll] for (uint q2 = 0u; q2 < uint(PX); ++q2)
                [unroll] for (uint j = 0u; j < 8u; ++j)
                    acc[q2][j] = acc[q2][j] + p[q2][j];
        }
#else
        [loop] for (int t = 0; t < TAPS; ++t)
        {
            float p[PX][8];
            [unroll] for (uint q = 0u; q < uint(PX); ++q)
                [unroll] for (uint j = 0u; j < 8u; ++j)
                    p[q][j] = 0.0;
            uint xb[PX];
            [unroll] for (uint q1 = 0u; q1 < uint(PX); ++q1)
            {
                const int2 s = PIX(q1) * MLFG_CONV_S + int2(t % MLFG_CONV_K, t / MLFG_CONV_K);
                xb[q1] = (uint(s.y) * uint(TINW) + uint(s.x)) * uint(PS) + slice * uint(CI);
            }
            [loop] for (int ci = 0; ci < n; ++ci)
            {
                const uint   wa = (wOff + (uint(t) * uint(cin) + uint(c0 + ci)) * coP + uint(co0)) * 4u;
                const float4 w0 = asfloat(t_w.Load4(wa)), w1 = asfloat(t_w.Load4(wa + 16u));
                [unroll] for (uint q = 0u; q < uint(PX); ++q)
                {
                    const float x = s_x[xb[q] + uint(ci)];
                    p[q][0] = p[q][0] + x * w0.x; p[q][1] = p[q][1] + x * w0.y; p[q][2] = p[q][2] + x * w0.z; p[q][3] = p[q][3] + x * w0.w;
                    p[q][4] = p[q][4] + x * w1.x; p[q][5] = p[q][5] + x * w1.y; p[q][6] = p[q][6] + x * w1.z; p[q][7] = p[q][7] + x * w1.w;
                }
            }
            [unroll] for (uint q2 = 0u; q2 < uint(PX); ++q2)
                [unroll] for (uint j = 0u; j < 8u; ++j)
                    acc[q2][j] = acc[q2][j] + p[q2][j];
        }
#endif
    }
#if KS > 1
    // The slices' sums, added in slice order by slice 0.
    if (slice != 0u)
    {
        [unroll] for (uint q = 0u; q < uint(PX); ++q)
            [unroll] for (uint j = 0u; j < 8u; ++j)
                s_red[(((slice - 1u) * uint(TPS) + lt) * uint(PX) + q) * 8u + j] = acc[q][j];
    }
    GroupMemoryBarrierWithGroupSync();
    if (slice != 0u)
        return;
    [unroll] for (uint s = 1u; s < uint(KS); ++s)
        [unroll] for (uint q = 0u; q < uint(PX); ++q)
            [unroll] for (uint j = 0u; j < 8u; ++j)
                acc[q][j] = acc[q][j] + s_red[(((s - 1u) * uint(TPS) + lt) * uint(PX) + q) * 8u + j];
#endif
    const uint   ba = (wOff + uint(TAPS * cin) * coP + uint(co0)) * 4u;
    const float4 b0 = asfloat(t_w.Load4(ba)), b1 = asfloat(t_w.Load4(ba + 16u));
    const float  bias[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    [unroll] for (uint q = 0u; q < uint(PX); ++q)
    {
        const int2 o = tile + PIX(q);
        if (any(o >= c_out.xy))
            continue;
        const uint pix = uint(o.y) * uint(c_out.x) + uint(o.x);
        float      v[8];
        [unroll] for (uint k = 0u; k < 8u; ++k)
        {
            const int c = co0 + int(k);
            float     y = acc[q][k] + bias[k];
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
}
