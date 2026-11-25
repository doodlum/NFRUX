// The embedded network and its description for a size. SPDX-License-Identifier: MIT
#pragma once
#include "ml/ffx_vk_ml_runtime.h"
namespace nfrux
{
constexpr uint32_t kInputChannels = 20;
struct ModelBindings { uint32_t input, params, gate; };
ModelBindings     GetModelBindings();
ffx_ml::GraphDesc DescribeModel(uint32_t width, uint32_t height);
}