// The horizontal half of the composite's fp16 bilinear x4 upsample of the five blend weights (FSR FG4 pass 190; AMD
// FidelityFX SDK v2.4.0, MIT; spec: tools/ffx_extract/mlfg_io.py pass190_composite), done once a column per weight row
// instead of twice a full-resolution pixel. The composite (c_splat.w = 1) then reads two of these texels and finishes
// with the vertical mix: F16Round(Mix(F16Round(Mix(a, b, fx)), F16Round(Mix(c, d, fx)), fy)), the same bits as before.
//
// Reads   t1 weights   kMlfgSoftmax5's [TH][pitch][8] fp16 buffer (ch 0-4), live size c_size.zw (TW x TH)
// Writes  u2 rows      [TH][W][8] fp16: ch k = F16Round(Mix(w_k(x0, y), w_k(x1, y), fx)) for k < 5 (x0, x1, fx as the
//                      composite's x axis for output column x), ch 5-7 zero
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

cbuffer Constants : register(b0)
{
    int4 c_size;  // output width W (the colour's), weight row pitch, live width TW, live height TH
};
ByteAddressBuffer   t_weights : register(t1);
RWByteAddressBuffer u_rows : register(u2);

[numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int x = int(id.x), y = int(id.y);
    if (x >= c_size.x || y >= c_size.w)
        return;
    int   x0, x1;
    float fx;
    ResampleAxis(x, c_size.z, 0.25, x0, x1, fx);
    const uint  ww = uint(c_size.y);
    const uint4 wa = t_weights.Load4(TexelBase(int2(x0, y), ww, 8u) << 1), wb = t_weights.Load4(TexelBase(int2(x1, y), ww, 8u) << 1);
    float h[6];
    [unroll] for (uint k = 0u; k < 5u; ++k)
    {
        const uint sh = (k & 1u) * 16u;
        h[k] = F16Round(Mix(F16ToF32((wa[k >> 1] >> sh) & 0xFFFFu), F16ToF32((wb[k >> 1] >> sh) & 0xFFFFu), fx));
    }
    h[5] = 0.0;
    const uint o = TexelBase(int2(x, y), uint(c_size.x), 8u) << 1;
    u_rows.Store4(o, uint4(PackF16x2(h[0], h[1]), PackF16x2(h[2], h[3]), PackF16x2(h[4], h[5]), 0u));
}
