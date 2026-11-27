// FSR FG4 network layers in int8 (NFRUX's int8 tier of the ML runtime, core/mlfg, AmdPrecision=int8): any D3D12 FL12 /
// Vulkan 1.1 GPU. AMD FidelityFX SDK v2.4.0, MIT; spec: tools/ffx_extract/mlfg_int8.py (the integer semantics of an
// export_amd int8 export). Builds as mlfg_conv.hlsl's (MLFG_CONV_K / _S / _T, MLFG_TW x MLFG_TH pixels by 8 output
// channels a group of 64 threads, input channels in MLFG_KS slices), MLFG_DOT4 = 1: SM 6.4 / SPV_KHR_integer_dot_product
// packed dot products (dot4add_i8packed), 0: the same sums in 32-bit integer code. Integer sums are exact, so the two
// and the slice split give the same bits; the fp32 epilogue is strict (no fused multiply-add) like every kMlfg pass.
//
// Reads   t1 src     int8 NHWC, 4 channels a 32-bit word (channel c in byte c % 4), c_in.z = pitch in channels (a
//                    multiple of 4), first channel c_in.w (a multiple of 4); or (mode 1) an fp16 NHWC tensor
//                    (mlfg_common.hlsli layout) quantised per channel while staged: x = Q(v * rf[c])
//         t3 params  per layer, from word c_off.x: weights [tap][cin / 4][coPad] words (4 input channels of one output,
//                    channel 4i + k in byte k), then fp32 m[coPad] (at c_off.z), b[coPad], rf[cin rounded up to 4]
//         t4 res     int8 residual / skip (mode 8): pitch c_res.x, first channel c_res.y, scale c_q2.x
// Writes  u5 out     int8 (pitch / first channel in channels, multiples of 4) or fp16 (modes 2 and 4; channels
//                    [cout, writeEnd) written as 0)
// Epilogue per output o: y = fp32(fp32(float(acc) * m[o]) + b[o]) [+ fp32(R * s_res)]; mode 2: fp16(y); else with an
// activation (c_cfg.z 1 leaky 0.2 / 2 relu): p = Q(y * inv_pre) * s_pre, a = act(p), O = Q(a * inv_out); without:
// O = Q(y * inv_out); mode 4: fp16(O * s_out). Q(v) = clamp(round-to-nearest-even(v), -127, 127).
// Output channel groups at and above c_res.z copy the source requantised (the FasterNet expand input's x[P:] part):
// O = Q(fp32(X * c_q2.y) * inv_out), same channel index.
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

cbuffer Constants : register(b0)
{
    int4   c_in;   // src width, height, pitch, first channel
    int4   c_out;  // out width, height, pitch, first channel
    int4   c_cfg;  // cout, cin, act, mode (1 fp16 source, 2 fp16 output, 4 fp16 dequantised output, 8 residual)
    int4   c_off;  // weight offset (words), coPad, m offset (words), write end
    int4   c_res;  // residual pitch, residual first channel, copy from (output channel), 0
    float4 c_q;    // inv_pre, s_pre, inv_out, s_out
    float4 c_q2;   // s_res, s_in (copy), 0, 0
};
ByteAddressBuffer   t_src : register(t1);
ByteAddressBuffer   t_w : register(t3);
ByteAddressBuffer   t_res : register(t4);
RWByteAddressBuffer u_out : register(u5);

#if MLFG_DOT4
#define DOT4(a, b, c) dot4add_i8packed(a, b, c)
#else
int DOT4(uint a, uint b, int c)
{
    [unroll] for (uint k = 0u; k < 32u; k += 8u)
        c += (int(a << (24u - k)) >> 24) * (int(b << (24u - k)) >> 24);
    return c;
}
#endif

// Q: clamp, then round to nearest even by the 1.5 * 2^23 add (exact for |v| <= 127; precise: no folding).
int Q8(float v)
{
    precise const float c = min(max(v, -127.0), 127.0);
    precise const float r = (c + 12582912.0) - 12582912.0;
    return int(r);
}
int  S8(uint w, uint k) { return int(w << (24u - 8u * k)) >> 24; }
uint Pack8(int a, int b, int c, int d) { return (uint(a) & 0xFFu) | ((uint(b) & 0xFFu) << 8) | ((uint(c) & 0xFFu) << 16) | (uint(d) << 24); }
float Wf(uint i) { return asfloat(t_w.Load(i << 2)); }

#define TW MLFG_TW
#define TH MLFG_TH
#define KS MLFG_KS
#define TPS (64 / KS)
#define PX (TW * TH / TPS)
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
#define CW 4              // words (16 channels) a slice and chunk
#define PS (CW * KS + 1)  // a staged pixel's words, one more than its data (bank spread)
groupshared uint s_x[TINW * TINH * PS];
#if KS > 1
groupshared int s_red[(KS - 1) * TPS * PX * 8];
#endif

// The staged word w (channels 4w .. 4w + 3 from the chunk's first) of input pixel p.
uint LoadWord(int2 p, int c)
{
    if (!Inside(p, c_in.xy) || c >= c_cfg.y)
        return 0u;
    const uint pix = uint(p.y) * uint(c_in.x) + uint(p.x);
    if ((c_cfg.w & 1) == 0)
        return t_src.Load(((pix * uint(c_in.z) + uint(c_in.w + c)) >> 2) << 2);
    int q[4];
    [unroll] for (uint k = 0u; k < 4u; ++k)
    {
        const int ck = c + int(k);
        q[k] = ck < c_cfg.y ? Q8(LoadHalf(t_src, pix * uint(c_in.z) + uint(c_in.w + ck)) * Wf(uint(c_off.z + 2 * c_off.y + ck))) : 0;
    }
    return Pack8(q[0], q[1], q[2], q[3]);
}

void StoreOut(uint pix, int co0, int o[8], float y[8])
{
    if ((c_cfg.w & 6) != 0)
    {
        float v[8];
        [unroll] for (uint k = 0u; k < 8u; ++k)
            v[k] = co0 + int(k) >= c_cfg.x ? 0.0 : ((c_cfg.w & 2) != 0 ? F16Round(y[k]) : F16Round(float(o[k]) * c_q.w));
        const uint ob = pix * uint(c_out.z) + uint(c_out.w + co0);
        [unroll] for (uint m = 0u; m < 8u; m += 2u)
            if (co0 + int(m) < c_off.w)
                StoreHalf2(u_out, ob + m, v[m], v[m + 1u]);
        return;
    }
    const uint ob = (pix * uint(c_out.z) + uint(c_out.w + co0)) << 0;
    u_out.Store(ob, Pack8(o[0], o[1], o[2], o[3]));
    u_out.Store(ob + 4u, Pack8(o[4], o[5], o[6], o[7]));
}

[numthreads(64, 1, 1)] void main(uint3 gid : SV_GroupID, uint li : SV_GroupIndex)
{
    const uint slice = li / uint(TPS), lt = li % uint(TPS);
    const int2 tile  = int2(gid.xy) * int2(TW, TH);
    const int2 first = int2(lt % uint(TW), lt / uint(TW));
#define PIX(q) (first + int2(0, int(q) * (TPS / TW)))
    const int co0 = int(gid.z) * 8;
    if (co0 >= c_res.z)
    {
        // Copy: the source's channels co0 .. co0 + 7 requantised to the output's scale (one thread a pixel).
        if (slice != 0u)
            return;
        [unroll] for (uint q = 0u; q < uint(PX); ++q)
        {
            const int2 o = tile + PIX(q);
            if (any(o >= c_out.xy))
                continue;
            const uint pix = uint(o.y) * uint(c_out.x) + uint(o.x);
            const uint a   = pix * uint(c_in.z) + uint(c_in.w + co0);
            const uint w0 = t_src.Load(a), w1 = t_src.Load(a + 4u);
            int   r[8];
            float y[8];
            [unroll] for (uint k = 0u; k < 8u; ++k)
            {
                precise const float v = float(S8(k < 4u ? w0 : w1, k & 3u)) * c_q2.y;
                r[k] = Q8(v * c_q.z);
                y[k] = 0.0;
            }
            StoreOut(pix, co0, r, y);
        }
        return;
    }
#ifdef MLFG_CONV_T
    const int2 origin = tile / 2;
#else
    const int2 origin = tile * MLFG_CONV_S - PAD;
#endif
    const int  cin  = c_cfg.y;
    const uint cin4 = uint(cin + 3) >> 2;
    const uint coP  = uint(c_off.y);
    const uint wOff = uint(c_off.x);
    int acc[PX][8];
    [unroll] for (uint q = 0u; q < uint(PX); ++q)
        [unroll] for (uint j = 0u; j < 8u; ++j)
            acc[q][j] = 0;
    [loop] for (int cb = 0; cb < cin; cb += 4 * CW * KS)
    {
        GroupMemoryBarrierWithGroupSync();
        [loop] for (uint i = li; i < uint(TINW * TINH * CW * KS); i += 64u)
        {
            const uint p = i / uint(CW * KS), w = i % uint(CW * KS);
            s_x[p * uint(PS) + w] = LoadWord(origin + int2(p % uint(TINW), p / uint(TINW)), cb + 4 * int(w));
        }
        GroupMemoryBarrierWithGroupSync();
        const int  c0 = cb + int(slice) * 4 * CW;
        const uint w0i = uint(c0) >> 2;
        const int  n  = min(CW, int(cin4) - int(w0i));  // words of this slice (<= 0: past the last channel)
#ifdef MLFG_CONV_T
        {
            const uint t = uint((first.y & 1) * 2 + (first.x & 1));
            uint xb[PX];
            [unroll] for (uint q = 0u; q < uint(PX); ++q)
            {
                const int2 s = PIX(q) >> 1;
                xb[q] = (uint(s.y) * uint(TINW) + uint(s.x)) * uint(PS) + slice * uint(CW);
            }
            [loop] for (int wi = 0; wi < n; ++wi)
            {
                const uint  wa = (wOff + (t * cin4 + w0i + uint(wi)) * coP + uint(co0)) << 2;
                const uint4 a0 = t_w.Load4(wa), a1 = t_w.Load4(wa + 16u);
                [unroll] for (uint q = 0u; q < uint(PX); ++q)
                {
                    const uint x = s_x[xb[q] + uint(wi)];
                    acc[q][0] = DOT4(x, a0.x, acc[q][0]); acc[q][1] = DOT4(x, a0.y, acc[q][1]); acc[q][2] = DOT4(x, a0.z, acc[q][2]); acc[q][3] = DOT4(x, a0.w, acc[q][3]);
                    acc[q][4] = DOT4(x, a1.x, acc[q][4]); acc[q][5] = DOT4(x, a1.y, acc[q][5]); acc[q][6] = DOT4(x, a1.z, acc[q][6]); acc[q][7] = DOT4(x, a1.w, acc[q][7]);
                }
            }
        }
#else
        [loop] for (int t = 0; t < TAPS; ++t)
        {
            uint xb[PX];
            [unroll] for (uint q1 = 0u; q1 < uint(PX); ++q1)
            {
                const int2 s = PIX(q1) * MLFG_CONV_S + int2(t % MLFG_CONV_K, t / MLFG_CONV_K);
                xb[q1] = (uint(s.y) * uint(TINW) + uint(s.x)) * uint(PS) + slice * uint(CW);
            }
            [loop] for (int wi = 0; wi < n; ++wi)
            {
                const uint  wa = (wOff + (uint(t) * cin4 + w0i + uint(wi)) * coP + uint(co0)) << 2;
                const uint4 a0 = t_w.Load4(wa), a1 = t_w.Load4(wa + 16u);
                [unroll] for (uint q = 0u; q < uint(PX); ++q)
                {
                    const uint x = s_x[xb[q] + uint(wi)];
                    acc[q][0] = DOT4(x, a0.x, acc[q][0]); acc[q][1] = DOT4(x, a0.y, acc[q][1]); acc[q][2] = DOT4(x, a0.z, acc[q][2]); acc[q][3] = DOT4(x, a0.w, acc[q][3]);
                    acc[q][4] = DOT4(x, a1.x, acc[q][4]); acc[q][5] = DOT4(x, a1.y, acc[q][5]); acc[q][6] = DOT4(x, a1.z, acc[q][6]); acc[q][7] = DOT4(x, a1.w, acc[q][7]);
                }
            }
        }
#endif
    }
#if KS > 1
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
                acc[q][j] += s_red[(((s - 1u) * uint(TPS) + lt) * uint(PX) + q) * 8u + j];
#endif
    float mo[8], bo[8];
    [unroll] for (uint k = 0u; k < 8u; ++k)
    {
        mo[k] = Wf(uint(c_off.z + co0) + k);
        bo[k] = Wf(uint(c_off.z + int(coP) + co0) + k);
    }
    [unroll] for (uint q = 0u; q < uint(PX); ++q)
    {
        const int2 o = tile + PIX(q);
        if (any(o >= c_out.xy))
            continue;
        const uint pix = uint(o.y) * uint(c_out.x) + uint(o.x);
        uint rw0 = 0u, rw1 = 0u;
        if ((c_cfg.w & 8) != 0)
        {
            const uint ra = pix * uint(c_res.x) + uint(c_res.y + co0);
            rw0 = t_res.Load(ra), rw1 = t_res.Load(ra + 4u);
        }
        int   r[8];
        float y[8];
        [unroll] for (uint k = 0u; k < 8u; ++k)
        {
            precise float v = float(acc[q][k]) * mo[k];
            v = v + bo[k];
            if ((c_cfg.w & 8) != 0)
            {
                precise const float rv = float(S8(k < 4u ? rw0 : rw1, k & 3u)) * c_q2.x;
                v = v + rv;
            }
            y[k] = v;
            if (c_cfg.z != 0)
            {
                precise float p = float(Q8(v * c_q.x)) * c_q.y;
                if (c_cfg.z == 1)
                    p = p > 0.0 ? p : p * 0.2;
                else
                    p = max(p, 0.0);
                r[k] = Q8(p * c_q.z);
            }
            else
                r[k] = Q8(v * c_q.z);
        }
        StoreOut(pix, co0, r, y);
    }
}
