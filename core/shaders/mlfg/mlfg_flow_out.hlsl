// FSR FG4 network passes 149 + 151: final flow residual and the export of the two flows as UV images
// (AMD FidelityFX SDK v2.4.0, MIT; spec: tools/ffx_extract/mlfg_glue.py pass149 / pass151).
//
// Reads   t1 head     f149 [headH][headW][16] fp16 from the network (ch 0-3 = the last flow residual), tensorSizes[69]
//         t2 prev     flow112 [H][W][4] fp16 (kMlfgRefine mode 2's flow), level-0 pixels
// Writes  u3 flowout  [outH][outW][4] fp16 = f149[0:4] + flow112 (fp16 sum), tensorSizes[69] (zero outside)
//         u4 flow0toT  RG16F image, imageSize = tensorSizes[70] (display / 8: the true level-0 size), UV units:
//                      flowout[0:2] / imageSize (fp32 division, fp16 result)
//         u5 flow1toT  RG16F image, the same from flowout[2:4]
// flow0toT / flow1toT (AMD: rw_output_flow_0to05 / _1to05) are the offsets from a pixel of the generated frame at t
// to where it is found in frame 0 / frame 1 (sample frame k at uv + flow_ktoT). Renamed only: the values are the
// network's, for whatever t channel 18 of its stacks held.
// SPDX-License-Identifier: MIT
#include "mlfg_common.hlsli"

cbuffer Constants : register(b0)
{
    int4 c_head;   // head width (row pitch), prev width (row pitch), 0, 0
    int4 c_out;    // flowout width, height (the buffer's extent), live size tensorSizes[69].xy
    int4 c_image;  // image size tensorSizes[70].xy, 0, 0
};
ByteAddressBuffer   t_head : register(t1);
ByteAddressBuffer   t_prev : register(t2);
RWByteAddressBuffer u_flow : register(u3);
[[vk::image_format("rg16f")]] RWTexture2D<float4> u_flow0 : register(u4);
[[vk::image_format("rg16f")]] RWTexture2D<float4> u_flow1 : register(u5);

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    const int2 g = int2(id.xy);
    const bool inOut = all(g < c_out.xy), inImage = all(g < c_image.xy);
    if (!inOut && !inImage)
        return;
    float4 f = float4(0, 0, 0, 0);
    if (all(g < c_out.zw))
    {
        const uint hb = TexelBase(g, uint(c_head.x), 16u), pb = TexelBase(g, uint(c_head.y), 4u);
        f = F16Round(float4(LoadHalf2(t_head, hb), LoadHalf2(t_head, hb + 2u)) + float4(LoadHalf2(t_prev, pb), LoadHalf2(t_prev, pb + 2u)));
    }
    if (inOut)
    {
        const uint ob = TexelBase(g, uint(c_out.x), 4u);
        StoreHalf2(u_flow, ob, f.x, f.y);
        StoreHalf2(u_flow, ob + 2u, f.z, f.w);
    }
    if (inImage)
    {
        const float2 s = float2(c_image.xy);
        u_flow0[g] = float4(F16Round(FDiv(f.xy, s)), 0, 0);
        u_flow1[g] = float4(F16Round(FDiv(f.zw, s)), 0, 0);
    }
}
