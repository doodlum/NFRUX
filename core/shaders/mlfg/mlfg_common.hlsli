// Shared helpers of the FSR Frame Generation 4 port (AMD FidelityFX SDK v2.4.0, MIT; see docs/FFX_MLFG.md and
// tools/ffx_extract/mlfg_io.py / mlfg_glue.py, which are the specification these shaders follow).
//
// Numeric rules (docs/DESIGN.md D1, A2a): the passes compute the same bits on D3D12 and Vulkan and the same bits as the
// PyTorch transcription at t = 0.5 ("glsl" rounding mode of mlfg_glue.py: every fp16 rounding point of AMD's shader).
//   * No 16-bit types (build_shaders.py refuses them): fp16 values are carried in fp32 registers, rounded with F16Round
//     (round to nearest even, subnormals kept), and stored packed two to a 32-bit word with PackF16x2.
//   * Conversions are integer code, not f32tof16 / PackHalf2x16, whose rounding mode is not specified by either API.
//   * Division is FDiv: correctly rounded (IEEE) in integer code; neither API specifies its OpFDiv / fdiv to the ulp.
//   * exp / log / pow are polynomial code (FExp, FLog, FPow): the same bits everywhere (about 1 ulp from the libm).
//   * Every pass whose name starts kMlfg is built strict (NoContraction, -Gis) in every build (build_shaders.py ALWAYS),
//     and mix / fma / dot are written out (a + (b - a) * t, as mlfg_io / mlfg_glue).
//   * Texture filtering is shader arithmetic on Load()ed texels (exact fp32 weights), never a sampler.
//   * No wave operations, no group-size assumptions.
//
// Tensor layout shared with the ML runtime: every tensor is its own buffer, NHWC fp16, element (x, y, c) at
// half index (y * width + x) * pitch + c (byte = 2 * that), pitch = channel pitch (a multiple of 2; 16 / 32 where the
// network reads 16-channel windows). Channels from C to pitch-1 are written as 0. Two halves share a 32-bit word,
// channel c in the low 16 bits when c is even.
// SPDX-License-Identifier: MIT
#ifndef MLFG_COMMON_HLSLI
#define MLFG_COMMON_HLSLI

#define F16_MAX 65504.0
#define INVALID2 0x7BFF7BFFu  // packHalf2x16(65504, 65504): the "no flow" word of a splat texel

// ------------------------------------------------------------------------------------------------ fp16 conversion
// fp32 -> fp16 bits, round to nearest, ties to even; overflow to inf, NaN kept (quiet), subnormal results rounded.
uint F32ToF16(float f)
{
    const uint u = asuint(f);
    const uint s = (u >> 16) & 0x8000u;
    const uint e = (u >> 23) & 0xFFu;
    uint       m = u & 0x7FFFFFu;
    if (e == 0xFFu)
        return s | 0x7C00u | (m != 0u ? 0x200u : 0u);
    const int E = int(e) - 112;  // fp16 biased exponent
    if (E >= 31)
        return s | 0x7C00u;
    if (E <= 0)
    {
        if (E < -10)
            return s;  // below 2^-25 (2^-25 itself is a tie that rounds to the even 0)
        m |= 0x800000u;
        const uint shift = uint(14 - E);
        uint       q     = m >> shift;
        const uint rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (q & 1u) != 0u))
            q += 1u;
        return s | q;  // 0x400 after a carry is the smallest normal, as it should be
    }
    uint       q   = (uint(E) << 10) | (m >> 13);
    const uint rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u) != 0u))
        q += 1u;  // a carry into the exponent is right, up to inf
    return s | q;
}

// fp16 bits (low 16) -> fp32, exact.
float F16ToF32(uint h)
{
    const uint s = (h & 0x8000u) << 16;
    const uint e = (h >> 10) & 0x1Fu;
    const uint m = h & 0x3FFu;
    if (e == 0u)
        return asfloat(s | asuint(float(m) * 5.9604644775390625e-8));  // m * 2^-24, exact
    if (e == 31u)
        return asfloat(s | 0x7F800000u | (m << 13));
    return asfloat(s | ((e + 112u) << 23) | (m << 13));
}

// The same results faster: where the result is an fp16 normal ([2^-14, 65520) in magnitude, so no overflow and no
// subnormal), round to nearest even on 10 mantissa bits in the fp32 encoding itself (a carry moves into the exponent, as
// it should); elsewhere the full conversion.
float F16Round(float x)
{
    const uint u = asuint(x), a = u & 0x7FFFFFFFu;
    if (a >= 0x38800000u && a < 0x477FF000u)
        return asfloat((u + 0xFFFu + ((u >> 13) & 1u)) & 0xFFFFE000u);
    return F16ToF32(F32ToF16(x));
}
float2 F16Round(float2 x) { return float2(F16Round(x.x), F16Round(x.y)); }
float3 F16Round(float3 x) { return float3(F16Round(x.x), F16Round(x.y), F16Round(x.z)); }
float4 F16Round(float4 x) { return float4(F16Round(x.x), F16Round(x.y), F16Round(x.z), F16Round(x.w)); }

// fp32 -> fp16 bits, F32ToF16's results, with F16Round's fast path for fp16 normals.
uint F32ToF16Fast(float x)
{
    const uint u = asuint(x), a = u & 0x7FFFFFFFu;
    if (a >= 0x38800000u && a < 0x477FF000u)
        return ((u >> 16) & 0x8000u) | ((((a + 0xFFFu + ((a >> 13) & 1u)) & 0x7FFFE000u) - 0x38000000u) >> 13);
    return F32ToF16(x);
}
uint   PackF16x2(float lo, float hi) { return F32ToF16Fast(lo) | (F32ToF16Fast(hi) << 16); }
float2 UnpackF16x2(uint w) { return float2(F16ToF32(w & 0xFFFFu), F16ToF32(w >> 16)); }

// ------------------------------------------------------------------------------------------------ exact division
// a / b in fp32, correctly rounded (round to nearest even), as torch / IEEE: long division of the significands.
float FDiv(float a, float b)
{
    const uint ua = asuint(a), ub = asuint(b);
    const uint sign = (ua ^ ub) & 0x80000000u;
    uint       ea = (ua >> 23) & 0xFFu, eb = (ub >> 23) & 0xFFu;
    uint       ma = ua & 0x7FFFFFu, mb = ub & 0x7FFFFFu;
    const bool nanA = ea == 0xFFu && ma != 0u, nanB = eb == 0xFFu && mb != 0u;
    const bool infA = ea == 0xFFu && ma == 0u, infB = eb == 0xFFu && mb == 0u;
    const bool zeroA = ea == 0u && ma == 0u, zeroB = eb == 0u && mb == 0u;
    if (nanA || nanB || (infA && infB) || (zeroA && zeroB))
        return asfloat(0x7FC00000u);
    if (infA || zeroB)
        return asfloat(sign | 0x7F800000u);
    if (zeroA || infB)
        return asfloat(sign);
    int expA = int(ea), expB = int(eb);
    if (ea == 0u)
    {
        expA = 1;
        while ((ma & 0x800000u) == 0u) { ma <<= 1; expA -= 1; }
    }
    else
        ma |= 0x800000u;
    if (eb == 0u)
    {
        expB = 1;
        while ((mb & 0x800000u) == 0u) { mb <<= 1; expB -= 1; }
    }
    else
        mb |= 0x800000u;
    int e = expA - expB + 127;
    if (ma < mb)
    {
        ma <<= 1;
        e -= 1;
    }
    // ma / mb in [1, 2): 24 quotient bits, then the remainder decides the rounding.
    uint q = 0u, r = ma;
    [loop] for (int i = 0; i < 24; ++i)
    {
        q <<= 1;
        if (r >= mb)
        {
            r -= mb;
            q |= 1u;
        }
        r <<= 1;
    }
    // r = 2 * remainder: compare with mb for the half.
    if (e >= 1)
    {
        if (r > mb || (r == mb && (q & 1u) != 0u))
            q += 1u;
        if (q == 0x1000000u)
        {
            q >>= 1;
            e += 1;
        }
        if (e >= 255)
            return asfloat(sign | 0x7F800000u);
        return asfloat(sign | (uint(e) << 23) | (q & 0x7FFFFFu));
    }
    // Subnormal result.
    const int shift = 1 - e;
    if (shift > 25)
        return asfloat(sign);
    const uint us = uint(shift);
    uint       qs   = q >> us;
    const uint lost = q & ((1u << us) - 1u), half = 1u << (us - 1u);
    const bool sticky = r != 0u;
    if (lost > half || (lost == half && (sticky || (qs & 1u) != 0u)))
        qs += 1u;
    return asfloat(sign | qs);
}
float2 FDiv(float2 a, float2 b) { return float2(FDiv(a.x, b.x), FDiv(a.y, b.y)); }
float3 FDiv(float3 a, float3 b) { return float3(FDiv(a.x, b.x), FDiv(a.y, b.y), FDiv(a.z, b.z)); }

// ------------------------------------------------------------------------------------------------ exp / log / pow
// Deterministic (only + * and integer code), within about 1 ulp of a correctly rounded result for normal outputs.
float LdExp(float x, int k)  // x * 2^k for x normal and the result normal or a flush to 0 / inf
{
    if (k > 127) { x = x * 1.7014118346046923e38; k -= 127; }  // 2^127
    if (k < -126) { x = x * 1.1754943508222875e-38; k += 126; }  // 2^-126
    k = clamp(k, -126, 127);
    return x * asfloat(uint(k + 127) << 23);
}

float FExp(float x)
{
    if (x != x)
        return x;
    if (x > 88.72284)
        return asfloat(0x7F800000u);
    if (x < -103.97208)
        return 0.0;
    // Cody-Waite: x = k ln2 + r, |r| <= ln2 / 2; ln2 = HI + LO with HI exact in 16 bits so k * HI is exact.
    const float k  = floor(x * 1.44269502 + 0.5);
    const float r0 = x - k * 0.693145751953125;
    const float r  = r0 - k * 1.428606765330187e-06;
    // exp(r) - 1 - r on [-0.347, 0.347], Taylor to degree 7 (error below 2^-27 relative).
    float p = 1.9841270e-04;
    p = p * r + 1.3888889e-03;
    p = p * r + 8.3333338e-03;
    p = p * r + 4.1666668e-02;
    p = p * r + 1.6666667e-01;
    p = p * r + 0.5;
    const float e = (p * (r * r) + r) + 1.0;
    return LdExp(e, int(k));
}

// Natural log for x > 0 (0 -> -inf, x < 0 or NaN -> NaN).
float FLog(float x)
{
    if (x != x || x < 0.0)
        return asfloat(0x7FC00000u);
    if (x == 0.0)
        return asfloat(0xFF800000u);
    if (asuint(x) == 0x7F800000u)
        return x;
    uint u = asuint(x);
    int  k = 0;
    if ((u >> 23) == 0u)  // subnormal: scale up by 2^25
    {
        x = x * 33554432.0;
        u = asuint(x);
        k = -25;
    }
    k += int(u >> 23) - 127;
    float m = asfloat((u & 0x7FFFFFu) | 0x3F800000u);  // [1, 2)
    if (m > 1.41421354)
    {
        m = m * 0.5;
        k += 1;
    }
    // log(m) = 2 atanh(s), s = (m - 1) / (m + 1), |s| <= 0.1716.
    const float s  = FDiv(m - 1.0, m + 1.0);
    const float s2 = s * s;
    float p = 0.11111111;
    p = p * s2 + 0.14285715;
    p = p * s2 + 0.2;
    p = p * s2 + 0.33333334;
    const float lm = (s * s2) * p * 2.0 + s * 2.0;
    const float kf = float(k);
    return (kf * 1.428606765330187e-06 + lm) + kf * 0.693145751953125;
}

// pow for x >= 0 as the transfer functions use it (x = 0 -> 0 for y > 0).
float  FPow(float x, float y) { return x == 0.0 ? (y > 0.0 ? 0.0 : (y == 0.0 ? 1.0 : asfloat(0x7F800000u))) : FExp(y * FLog(x)); }
float3 FPow(float3 x, float y)
{
    float3 r;
    [loop] for (uint k = 0u; k < 3u; ++k)  // a loop, not three inlined copies
        r[k] = FPow(x[k], y);
    return r;
}

// ------------------------------------------------------------------------------------------------ small helpers
float  Mix(float a, float b, float t) { return a + (b - a) * t; }
float2 Mix(float2 a, float2 b, float t) { return a + (b - a) * t; }
float3 Mix(float3 a, float3 b, float t) { return a + (b - a) * t; }
float3 Mix3(float3 a, float3 b, float3 t) { return a + (b - a) * t; }
float  Frac(float x) { return x - floor(x); }

// (p + 0.5) / size as mlfg_io.texel_uv: times the correctly rounded fp32 reciprocal.
float2 TexelUv(int2 p, float2 rcpSize) { return (float2(p) + 0.5) * rcpSize; }

// Clamp-free bounds test; Load()s are guarded: D3D12 returns 0 outside, Vulkan leaves it undefined.
bool Inside(int2 p, int2 size) { return p.x >= 0 && p.y >= 0 && p.x < size.x && p.y < size.y; }

// Float -> int conversion as ivec2(vec2) (toward zero), kept inside int range (both APIs leave overflow undefined).
int2 TruncI(float2 v) { return int2(clamp(v, -1.0e9, 1.0e9)); }

// ------------------------------------------------------------------------------------------------ tensor buffers
// One NHWC fp16 tensor: (width, pitch) as in the header comment; helpers take the half index of channel 0 of a texel.
uint TexelBase(int2 p, uint width, uint pitch) { return (uint(p.y) * width + uint(p.x)) * pitch; }

float LoadHalf(ByteAddressBuffer b, uint halfIndex)
{
    const uint w = b.Load((halfIndex >> 1) << 2);
    return F16ToF32((halfIndex & 1u) != 0u ? (w >> 16) : (w & 0xFFFFu));
}
float LoadHalfRW(RWByteAddressBuffer b, uint halfIndex)
{
    const uint w = b.Load((halfIndex >> 1) << 2);
    return F16ToF32((halfIndex & 1u) != 0u ? (w >> 16) : (w & 0xFFFFu));
}
// The glue passes' bilinear tap (mlfg_glue.bilerp16, "glsl" mode): fp16(mix(fp16(mix(a,b,fx)), fp16(mix(c,d,fx)), fy)).
float Bilerp16(float a, float b, float c, float d, float fx, float fy)
{
    return F16Round(Mix(F16Round(Mix(a, b, fx)), F16Round(Mix(c, d, fx)), fy));
}

// One axis of mlfg_glue.resample (bilinear, align_corners=False, clamp to edge): p = max((o + 0.5) * scale - 0.5, 0),
// i0 = clamp(floor p, 0, S-1), i1 = clamp(i0 + 1, 0, S-1), f = clamp(p - i0, 0, 1).
void ResampleAxis(int o, int S, float scale, out int i0, out int i1, out float f)
{
    const float p = max((float(o) + 0.5) * scale - 0.5, 0.0);
    i0 = clamp(int(floor(p)), 0, S - 1);
    i1 = clamp(i0 + 1, 0, S - 1);
    f  = clamp(p - float(i0), 0.0, 1.0);
}

// Two channels from an even half index.
float2 LoadHalf2(ByteAddressBuffer b, uint halfIndex) { return UnpackF16x2(b.Load(halfIndex << 1)); }
void   StoreHalf2(RWByteAddressBuffer b, uint halfIndex, float lo, float hi) { b.Store(halfIndex << 1, PackF16x2(lo, hi)); }

#endif
