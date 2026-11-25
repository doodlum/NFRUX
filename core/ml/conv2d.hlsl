// One int8 convolution layer of the network (CONV2D + RESCALE, + TABLE) for D3D12: exact integer arithmetic, the same
// results as the Vulkan kernels. Compiled once per layer shape (tools/build_shaders.py), with packed dot products (shader
// model 6.4) and without (6.0).
// SPDX-License-Identifier: MIT
// K, STRIDE, IC0 / IC1 (words per pixel of the two sources), OC, OCB (channels a thread makes), RESIZE (source 0 nearest
// resized), PERCH, TABLE, IMAGE (the output stored as R32_UINT texels), DOT4
cbuffer Layer : register(b0)
{
    int2 gIn;       // w, h of the convolution's input
    int2 gOut;
    int2 gS0;       // source 0's stored w, h
    int2 gPad;      // left, top
    int4 gScale;    // resize: y_n, y_d, x_n, x_d
    int2 gOffset;   // resize: y, x
    int  gOutZp;
    int  gInZp;
    uint gOutStrideWords;
    uint gOutRowWords;
};
ByteAddressBuffer   gSrc0 : register(t0);
ByteAddressBuffer   gSrc1 : register(t1);
ByteAddressBuffer   gWeights : register(t2);  // [ocBlock][tap][icWord][OCB] packed int8
ByteAddressBuffer   gParams : register(t3);   // [bias][multiplier][shift], each OC padded to 4; then the table
#if IMAGE
RWTexture2D<uint>   gDst : register(u0);
#else
RWByteAddressBuffer gDst : register(u0);
#endif

#define ICW (IC0 + IC1)
#define PSTRIDE ((OC + 3) & ~3)
#define COLS ((PX - 1) * STRIDE + K)

int Dot(uint a, uint b, int acc)
{
#if DOT4
    return dot4add_i8packed(a, b, acc);
#else
    const int4 x = int4(a << 24, a << 16, a << 8, a) >> 24, y = int4(b << 24, b << 16, b << 8, b) >> 24;
    return acc + x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
#endif
}

int Nearest(int v, int n, int d, int offset, int size)
{
    const int y = v * d + offset;
    int iy = y / n;
    if (iy * n > y)
        iy--;
    return 2 * (y - iy * n) >= n ? min(iy + 1, size - 1) : max(iy, 0);
}

// Signed 32 x 32 -> 64 from 16-bit halves (no Int64ShaderOps).
void MulWide(int a, int b, out uint hi, out uint lo)
{
    const uint ua = uint(a), ub = uint(b);
    const uint al = ua & 0xffff, ah = ua >> 16, bl = ub & 0xffff, bh = ub >> 16;
    const uint ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    const uint mid = (ll >> 16) + (lh & 0xffff) + (hl & 0xffff);
    lo = (ll & 0xffff) | (mid << 16);
    hi = hh + (lh >> 16) + (hl >> 16) + (mid >> 16);
    if (a < 0) hi -= ub;
    if (b < 0) hi -= ua;
}

int Epilogue(int acc, uint oc)
{
    const uint p   = PERCH ? oc : 0;
    const int  mul = asint(gParams.Load((PSTRIDE + p) * 4));
    const int  shift = asint(gParams.Load((2 * PSTRIDE + p) * 4));
    uint hi, lo;
    MulWide(acc, mul, hi, lo);
    // + (1 << (shift - 1)), then an arithmetic shift of the 64 bits.
    const uint rlo = shift - 1 < 32 ? 1u << (shift - 1) : 0u, rhi = shift - 1 < 32 ? 0u : 1u << (shift - 33);
    const uint sum = lo + rlo;
    hi += rhi + (sum < lo ? 1u : 0u);
    const int r = shift >= 32 ? int(hi) >> (shift - 32) : int((sum >> shift) | (hi << (32 - shift)));
    int v = clamp(r + gOutZp, -128, 127);
#if TABLE
    const uint i = uint(v + 128);
    v = int(gParams.Load((3 * PSTRIDE) * 4 + (i & ~3u)) << (24 - 8 * (i & 3))) >> 24;
#endif
    return v;
}

[numthreads(GX, GY, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
{
    const int ox0 = int(id.x) * PX, oy = int(id.y);
    if (oy >= gOut.y || ox0 >= gOut.x)
        return;
    const uint block = group.z;
    const uint zp    = uint(gInZp & 0xff) * 0x01010101u;
    int acc[PX * OCB];
    {
        [unroll] for (int p = 0; p < PX; ++p)
            [unroll] for (int o = 0; o < OCB; ++o)
            {
                const uint oc  = block * OCB + o;
                acc[p * OCB + o] = oc < OC ? asint(gParams.Load(oc * 4)) : 0;
            }
    }
    [loop] for (int ky = 0; ky < K; ++ky)
    {
        const int vy = oy * STRIDE - gPad.y + ky;
        uint base0[COLS], base1[COLS];
        bool ok[COLS];
        [unroll] for (int j = 0; j < COLS; ++j)
        {
            const int vx = ox0 * STRIDE - gPad.x + j;
            ok[j] = vx >= 0 && vx < gIn.x && vy >= 0 && vy < gIn.y;
            const int cx = clamp(vx, 0, gIn.x - 1), cy = clamp(vy, 0, gIn.y - 1);
#if RESIZE
            const int sx = Nearest(cx, gScale.z, gScale.w, gOffset.y, gS0.x), sy = Nearest(cy, gScale.x, gScale.y, gOffset.x, gS0.y);
#else
            const int sx = cx, sy = cy;
#endif
            base0[j] = uint(sy * gS0.x + sx) * IC0 * 4;
            base1[j] = uint(cy * gIn.x + cx) * IC1 * 4;
        }
        [loop] for (int w = 0; w < ICW; ++w)
        {
            uint x[COLS];
            [unroll] for (int j = 0; j < COLS; ++j)
            {
#if IC1 > 0
                [branch] if (w < IC0) x[j] = gSrc0.Load(base0[j] + w * 4);
                else x[j] = gSrc1.Load(base1[j] + (w - IC0) * 4);
#else
                x[j] = gSrc0.Load(base0[j] + w * 4);
#endif
                x[j] = ok[j] ? x[j] : zp;
            }
            [unroll] for (int kx = 0; kx < K; ++kx)
            {
                const uint wb = ((block * K * K + ky * K + kx) * ICW + w) * OCB * 4;
                [unroll] for (int q = 0; q < OCB; q += 4)
                {
                    const uint4 wv = gWeights.Load4(wb + q * 4);
                    [unroll] for (int p = 0; p < PX; ++p)
                    {
                        const uint xv = x[p * STRIDE + kx];
                        acc[p * OCB + q + 0] = Dot(xv, wv.x, acc[p * OCB + q + 0]);
                        acc[p * OCB + q + 1] = Dot(xv, wv.y, acc[p * OCB + q + 1]);
                        acc[p * OCB + q + 2] = Dot(xv, wv.z, acc[p * OCB + q + 2]);
                        acc[p * OCB + q + 3] = Dot(xv, wv.w, acc[p * OCB + q + 3]);
                    }
                }
            }
        }
    }
    [unroll] for (int p = 0; p < PX; ++p)
    {
        const int ox = ox0 + p;
        if (ox >= gOut.x)
            break;
        [unroll] for (int q = 0; q < OCB; q += 4)
        {
            uint packed = 0;
            [unroll] for (int j = 0; j < 4; ++j)
            {
                const uint oc = block * OCB + q + j;
                if (oc < OC)
                    packed |= uint(Epilogue(acc[p * OCB + q + j], oc) & 0xff) << (8 * j);
            }
#if IMAGE
            if (q == 0) gDst[uint2(ox, oy)] = packed;
#else
            gDst.Store((oy * gOutRowWords + ox * gOutStrideWords + (block * OCB + q) / 4) * 4, packed);
#endif
        }
    }
}
