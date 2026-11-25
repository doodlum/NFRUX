// The network on Vulkan through CUDA kernels (VK_NVX_binary_import, which Streamline enables): the same kernels as cuda_network on D3D12.
// SPDX-License-Identifier: MIT
#pragma once
#include "cuda_network.h"
#include "vulkan_network.h"

namespace nfrux
{
class CudaVulkanGraph
{
public:
    static bool Supported(VkDevice device, PFN_vkGetDeviceProcAddr gdpa);
    static std::unique_ptr<CudaVulkanGraph> Create(const ffx_ml::DeviceContext& dc, const ffx_ml::GraphDesc& desc, std::string* error);
    ~CudaVulkanGraph();
    struct Buffer { VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; uint64_t address = 0; void* map = nullptr; };
    Buffer   MakeBuffer(uint64_t size, bool upload);
    void     Destroy(Buffer& b);
    // Several of one model, a layer at a time. Addresses of each one's input and two outputs; pitch: an output row's bytes.
    static void Record(VkCommandBuffer cmd, CudaVulkanGraph* const* graphs, const uint64_t* inputs, const uint64_t (*outputs)[2], uint32_t count, uint32_t pitch);
    uint32_t outputBinding(uint32_t i) const { return m_plan.outputBindings[i]; }

private:
    uint64_t Address(VkBuffer b) const;
    struct Fn
    {
        PFN_vkCreateCuModuleNVX createModule; PFN_vkCreateCuFunctionNVX createFunction; PFN_vkDestroyCuModuleNVX destroyModule; PFN_vkDestroyCuFunctionNVX destroyFunction;
        PFN_vkCmdCuLaunchKernelNVX launch; PFN_vkCreateBuffer createBuffer; PFN_vkDestroyBuffer destroyBuffer; PFN_vkGetBufferMemoryRequirements requirements;
        PFN_vkAllocateMemory allocate; PFN_vkFreeMemory free; PFN_vkBindBufferMemory bind; PFN_vkMapMemory map; PFN_vkGetBufferDeviceAddress address;
        PFN_vkCmdPipelineBarrier barrier; PFN_vkCmdCopyBuffer copy;
    } m_fn = {};
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties m_memory = {};
    CudaPlan m_plan;
    std::vector<Buffer> m_tensors;
    Buffer m_constants, m_upload;
    VkCuModuleNVX m_module = VK_NULL_HANDLE;
    std::vector<VkCuFunctionNVX> m_functions;
    bool m_uploaded = false;
};
}  // namespace nfrux
