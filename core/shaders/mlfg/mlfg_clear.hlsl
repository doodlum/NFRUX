// Zeroes the two R32_UINT images of an FSR FG4 splat (kMlfgSplatKey / kMlfgSplatFlow need key and flow word 0 first,
// as AMD's cleared R64 image). SPDX-License-Identifier: MIT
cbuffer Constants : register(b0)
{
    int4 c_size;  // width, height, 0, 0
};
[[vk::image_format("r32ui")]] RWTexture2D<uint> u_key : register(u1);
[[vk::image_format("r32ui")]] RWTexture2D<uint> u_flow : register(u2);

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)
{
    if (any(int2(id.xy) >= c_size.xy))
        return;
    u_key[id.xy]  = 0u;
    u_flow[id.xy] = 0u;
}
