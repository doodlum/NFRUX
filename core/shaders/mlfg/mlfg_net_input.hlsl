// FSR FG4 network pass 0: the stage-1 network input (AMD FidelityFX SDK v2.4.0, MIT; spec:
// tools/ffx_extract/mlfg_glue.py pass0, decompiled mlfi_4k_pass0).
//
// Reads   t1 rgb0, t2 rgb1  the two colour frames (any float4 texture: UNORM8 / 10-bit / FP16), texSize.
//                           rgb0 = previous frame, rgb1 = current frame.
// Writes  u3 net_in  tensor [tensorH][tensorW][32] fp16 (NHWC, pitch 32; AMD: Tensor4 480x288x18 for the 4K network):
//           ch 0-2 rgb0, ch 3-5 rgb1, ch 6-31 written 0 (the encoder writes features into ch 6-11 / 12-17 later:
//           ch 6-11 = frame-0 features (AMD pass 8), 12-17 = frame-1 features (AMD pass 16)).
//           The network reads 16-channel windows of it: ch 0-15 (AMD pass 2) and ch 2-17 (AMD pass 10); both are
//           contiguous 16-half runs of a texel in this layout (at half offsets 0 and 2). The whole tensor is written
//           (zero outside the live region), so it needs no clear.
// Size rule: tensorW x tensorH = the network's level-0 tensor (tensorSizes[0], 480x288 for AMD's 4K network); the live
// region is min(tensorSizes[0], texSize / 8) and inside the generation rect.
// Each live texel samples each colour once at uv = (x + 0.5) * fp32(1 / (texSize / 8)), through AMD's binding-320
// sampler, which is nearest / clamp (mlfg_sample.hlsli): texel 8x+3 or 8x+4 (the corner between them, decided by the
// fp32 rounding of uv and the 21-bit truncation; mlfg_glue._nearest_index), then rounded to fp16. AMD's shader writes
// uv as a division; it is evaluated as the product by the rounded reciprocal, as the capture GPU did (bit-exact
// against AMD's dump). Transfer function 2 (scRGB) is converted to PQ with AMD's fp16 arithmetic (every fp16
// operation rounded); 0 (sRGB) and 1 (PQ) pass through.
// No timestep here.
// SPDX-License-Identifier: MIT
#include "mlfg_sample.hlsli"

cbuffer Constants : register(b0)
{
    int4   c_tensor;   // tensor width, height (the buffer's extent), live limit (tensorSizes[0].xy)
    int4   c_tex;      // texSize.xy, backBufferTransferFunction, 0
    float4 c_genRect;  // generationRectMinUV.xy, generationRectMaxUV.xy
    float4 c_lum;      // minMaxLuminance.xy, 0, 0
};
Texture2D<float4>   t_rgb0 : register(t1);
Texture2D<float4>   t_rgb1 : register(t2);
RWByteAddressBuffer u_net : register(u3);

// AMD's pass-0 scRGB -> PQ, in its fp16 arithmetic (constants are its fp16 literals).
float3 ToPqF16(float3 c)
{
    const float  lo = c_lum.x * 0.0125, range = (c_lum.y - c_lum.x) * 0.0125;
    const float3 a  = max(F16Round(FDiv(F16Round(c - lo), float3(range, range, range))), 0.0);
    // mat3(c0, c1, c2) * v, the ci being AMD's literal columns.
    const float3 c0 = float3(0.6274039745330810546875, 0.329281985759735107421875, 0.043313600122928619384765625);
    const float3 c1 = float3(0.06909699738025665283203125, 0.91954100131988525390625, 0.0113612003624439239501953125);
    const float3 c2 = float3(0.01639159955084323883056640625, 0.0880132019519805908203125, 0.895595014095306396484375);
    const float3 m  = F16Round((c0 * a.x + c1 * a.y) + c2 * a.z);
    const float3 n  = F16Round(FDiv(F16Round(m * 300.0), float3(10000.0, 10000.0, 10000.0)));
    const float3 y  = F16Round(FPow(n, 0.1593017578125));
    const float3 num = F16Round(0.8359375 + F16Round(y * 18.859375)), den = F16Round(1.0 + F16Round(y * 18.6875));
    return F16Round(FPow(F16Round(FDiv(num, den)), 78.875));
}

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int2 g = int2(id.xy);
    if (any(g >= c_tensor.xy))
        return;
    const int2   tex = c_tex.xy;
    const float2 uv  = (float2(g) + 0.5) * FDiv(float2(1, 1), float2(tex) * 0.125);
    const bool   inRect = !(any(uv > c_genRect.zw) || any(uv < c_genRect.xy));
    const bool   live = all(g < c_tensor.zw) && all(g < tex / 8) && all(g * 8 + 8 <= tex);  // the last: AMD's whole-block test
    float3       v[2] = {float3(0, 0, 0), float3(0, 0, 0)};
    if (live && inRect)
    {
        v[0] = F16Round(SampleG(t_rgb0, uv, tex).rgb);
        v[1] = F16Round(SampleG(t_rgb1, uv, tex).rgb);
        if (c_tex.z == 2)
        {
            [loop] for (uint k = 0u; k < 2u; ++k)
                v[k] = ToPqF16(v[k]);
        }
    }
    const uint base = TexelBase(g, uint(c_tensor.x), 32u);
    StoreHalf2(u_net, base + 0u, v[0].x, v[0].y);
    StoreHalf2(u_net, base + 2u, v[0].z, v[1].x);
    StoreHalf2(u_net, base + 4u, v[1].y, v[1].z);
    [unroll] for (uint c = 6u; c < 32u; c += 2u)
        u_net.Store((base + c) << 1, 0u);
}
