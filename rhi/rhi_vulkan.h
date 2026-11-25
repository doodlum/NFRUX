// SPDX-License-Identifier: MIT
#pragma once
#include "rhi.h"
#include <vulkan/vulkan.h>
namespace nfrux
{
struct VulkanHost
{
    VkInstance                instance       = VK_NULL_HANDLE;
    VkPhysicalDevice          physicalDevice = VK_NULL_HANDLE;
    VkDevice                  device         = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr gipa           = nullptr;
    PFN_vkGetDeviceProcAddr   gdpa           = nullptr;
    uint32_t                  mlFeatures     = 0;      // ffx_ml::FEATURE_* known to be enabled on the device
    bool                      atomic64       = false;  // shaderBufferInt64Atomics + shaderInt64 known to be enabled
};
std::unique_ptr<Device> CreateVulkanDevice(const VulkanHost& host);
}