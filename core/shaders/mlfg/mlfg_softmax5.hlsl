// FSR FG4 network pass 188, its activation only: softmax over the 5 blend logits with the running maximum started
// at 0 (AMD FidelityFX SDK v2.4.0, MIT; spec: tools/ffx_extract/mlfg_model.py conv_pass act 'softmax_max0').
// The 3x3 convolution in front of it runs on NFRUX's ML runtime.
//
// Reads   t1 logits   [H][inW][16] fp16, ch 0-4 the convolution's outputs (AMD rounds them to fp16 before the
//                     softmax: a runtime producing fp32 must round them first), tensorSizes[90] (960x540 for AMD's 4K
//                     network; 1/4 of the display is live)
// Writes  u2 weights  [H][outW][8] fp16, ch 0-4 the blend weights, ch 5-7 written 0 (AMD: Tensor4 960x540x5);
//                     read by kMlfgComposite*
//   m = max(0, y0..y4); e_i = exp(y_i - m) (fp32); w_i = e_i / (((e0 + e1) + e2) + e3 + e4), rounded to fp16.
// The five weights sum to 1: A (frame-0 warp, holes from frame 1), B (frame-1 warp, holes from frame 0), frame 0 at
// the flow, frame 1 at the flow, and the plain blend of both frames. No timestep here (see the composite).
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

cbuffer Constants : register(b0)
{
    int4 c_size;  // width, height, input width (row pitch), output width (row pitch)
};
ByteAddressBuffer   t_logits : register(t1);
RWByteAddressBuffer u_weights : register(u2);

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int2 g = int2(id.xy);
    if (any(g >= c_size.xy))
        return;
    const uint ib = TexelBase(g, uint(c_size.z), 16u);
    float      y[5];
    const float2 a = LoadHalf2(t_logits, ib), b = LoadHalf2(t_logits, ib + 2u);
    y[0] = a.x; y[1] = a.y; y[2] = b.x; y[3] = b.y; y[4] = LoadHalf(t_logits, ib + 4u);
    float m = 0.0;
    [unroll] for (uint k = 0u; k < 5u; ++k)
        m = max(m, y[k]);
    float e[5];
    float s = 0.0;
    [unroll] for (uint j = 0u; j < 5u; ++j)
    {
        e[j] = FExp(y[j] - m);
        s    = j == 0u ? e[0] : s + e[j];
    }
    const uint ob = TexelBase(g, uint(c_size.w), 8u);
    StoreHalf2(u_weights, ob, FDiv(e[0], s), FDiv(e[1], s));
    StoreHalf2(u_weights, ob + 2u, FDiv(e[2], s), FDiv(e[3], s));
    StoreHalf2(u_weights, ob + 4u, FDiv(e[4], s), 0.0);
    u_weights.Store((ob + 6u) << 1, 0u);
}
