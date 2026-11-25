// SPDX-License-Identifier: MIT
#pragma once
#include "rhi.h"
#include <vulkan/vulkan.h>
#include <string>
struct ID3D12Device;
namespace nfrux
{
struct VulkanHost
{
    VkInstance                instance       = VK_NULL_HANDLE;
    VkPhysicalDevice          physicalDevice = VK_NULL_HANDLE;
    VkDevice                  device         = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr gipa           = nullptr;
    PFN_vkGetDeviceProcAddr   gdpa           = nullptr;
    uint32_t                  queueFamily = 0, minorVersion = 2;
    const char* const*        deviceExtensions = nullptr;  // enabled, as far as known
    uint32_t                  deviceExtensionCount = 0;
    uint32_t                  mlFeatures     = 0;      // ffx_ml::FEATURE_* known (or assumed) to be enabled on the device
    std::string               marker;                  // assumed: the crash marker to clear once the network has run
};
std::unique_ptr<Device> CreateNriVulkanDevice(const VulkanHost& host);
void LogD3D12Messages(ID3D12Device* device);  // NFRUX_TEST_D3D12_MESSAGES: the debug layer's messages into the log
std::unique_ptr<Device> CreateNriD3D12Device(ID3D12Device* device);
}  // namespace nfrux
