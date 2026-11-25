// SPDX-License-Identifier: MIT
#include "model.h"
#include "nfru_mfg_v1_int8.h"
namespace nfrux
{
ModelBindings GetModelBindings()
{
    const auto& b = g_nfru_mfg_v1_int8_Info;
    return {b.inputTensorBindings[0], b.outputTensorBindings[0], b.outputTensorBindings[1]};
}
ffx_ml::GraphDesc DescribeModel(uint32_t width, uint32_t height)
{
    const auto&       b = g_nfru_mfg_v1_int8_Info;
    ffx_ml::GraphDesc d;
    d.spirv      = reinterpret_cast<const uint32_t*>(b.graphData);
    d.spirvWords = b.graphDataSize / 4;
    for (uint32_t i = 0; i < b.constantNums; ++i)
    {
        ffx_ml::GraphConstant k;
        k.id       = b.constantIds[i];
        k.vkFormat = b.constantFormats[i];
        k.shape.assign(b.constantShapes[i], b.constantShapes[i] + b.constantShapeSize[i]);
        k.data     = b.constantDatas[i];
        k.dataSize = b.constantDataSize[i];
        d.constants.push_back(k);
    }
    d.inputShapes[b.inputTensorBindings[0]] = {1, int64_t(height), int64_t(width), int64_t(kInputChannels)};
    return d;
}
}  // namespace nfrux