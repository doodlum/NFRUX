// SPDX-License-Identifier: MIT
#include "cuda_vulkan.h"

#include "../log.h"

#include <algorithm>
#include <cstring>

namespace nfrux
{
bool CudaVulkanGraph::Supported(VkDevice device, PFN_vkGetDeviceProcAddr gdpa)
{
    return gdpa(device, "vkCreateCuModuleNVX") && gdpa(device, "vkCmdCuLaunchKernelNVX");
}

std::unique_ptr<CudaVulkanGraph> CudaVulkanGraph::Create(const ffx_ml::DeviceContext& dc, const ffx_ml::GraphDesc& desc, std::string* error)
{
    auto fail = [&](const std::string& m) -> std::unique_ptr<CudaVulkanGraph> { if (error) *error = m; return nullptr; };
    if (!Supported(dc.device, dc.getDeviceProcAddr)) return fail("VK_NVX_binary_import is not enabled");
    VkPhysicalDeviceIDProperties id = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2  pp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(dc.getInstanceProcAddr(dc.instance, "vkGetPhysicalDeviceProperties2"))(dc.physicalDevice, &pp);
    if (!id.deviceLUIDValid) return fail("no LUID");
    LUID luid;
    memcpy(&luid, id.deviceLUID, sizeof(luid));
    auto out = std::unique_ptr<CudaVulkanGraph>(new CudaVulkanGraph());
    if (!MakeCudaPlan(desc, luid, out->m_plan, error)) return nullptr;
    auto& f = out->m_fn;
    auto  get = [&](auto& fn, const char* n) { fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dc.getDeviceProcAddr(dc.device, n)); };
    get(f.createModule, "vkCreateCuModuleNVX"), get(f.createFunction, "vkCreateCuFunctionNVX"), get(f.destroyModule, "vkDestroyCuModuleNVX"), get(f.destroyFunction, "vkDestroyCuFunctionNVX"),
        get(f.launch, "vkCmdCuLaunchKernelNVX"), get(f.createBuffer, "vkCreateBuffer"), get(f.destroyBuffer, "vkDestroyBuffer"), get(f.requirements, "vkGetBufferMemoryRequirements"),
        get(f.allocate, "vkAllocateMemory"), get(f.free, "vkFreeMemory"), get(f.bind, "vkBindBufferMemory"), get(f.map, "vkMapMemory"), get(f.address, "vkGetBufferDeviceAddress"),
        get(f.barrier, "vkCmdPipelineBarrier"), get(f.copy, "vkCmdCopyBuffer");
    if (!f.address) get(f.address, "vkGetBufferDeviceAddressKHR");
    if (!f.address) return fail("no buffer addresses");
    out->m_device = dc.device;
    reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(dc.getInstanceProcAddr(dc.instance, "vkGetPhysicalDeviceMemoryProperties"))(dc.physicalDevice, &out->m_memory);
    for (uint64_t s : out->m_plan.tensors)
        if (!out->m_tensors.emplace_back(out->MakeBuffer(s, false)).buffer) return fail("out of memory");
    out->m_constants = out->MakeBuffer(out->m_plan.constants.size(), false), out->m_upload = out->MakeBuffer(out->m_plan.constants.size(), true);
    if (!out->m_constants.buffer || !out->m_upload.map) return fail("out of memory");
    memcpy(out->m_upload.map, out->m_plan.constants.data(), out->m_plan.constants.size());
    VkCuModuleCreateInfoNVX mi = {VK_STRUCTURE_TYPE_CU_MODULE_CREATE_INFO_NVX, nullptr, out->m_plan.cubin.size(), out->m_plan.cubin.data()};
    if (f.createModule(dc.device, &mi, nullptr, &out->m_module)) return fail("vkCreateCuModuleNVX");
    for (size_t i = 0; i < out->m_plan.layers.size(); ++i)
    {
        const std::string name = "layer" + std::to_string(i);
        VkCuFunctionCreateInfoNVX fi = {VK_STRUCTURE_TYPE_CU_FUNCTION_CREATE_INFO_NVX, nullptr, out->m_module, name.c_str()};
        if (f.createFunction(dc.device, &fi, nullptr, &out->m_functions.emplace_back(VK_NULL_HANDLE))) return fail("vkCreateCuFunctionNVX");
    }
    return out;
}

CudaVulkanGraph::Buffer CudaVulkanGraph::MakeBuffer(uint64_t size, bool upload)
{
    Buffer b;
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = (size + 255) & ~255ull, ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!upload) ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (m_fn.createBuffer(m_device, &ci, nullptr, &b.buffer)) return {};
    VkMemoryRequirements r;
    m_fn.requirements(m_device, b.buffer, &r);
    const VkMemoryPropertyFlags want = upload ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryAllocateFlagsInfo fl = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, upload ? nullptr : &fl, r.size, UINT32_MAX};
    for (uint32_t i = 0; i < m_memory.memoryTypeCount && ai.memoryTypeIndex == UINT32_MAX; ++i)
        if ((r.memoryTypeBits >> i & 1) && (m_memory.memoryTypes[i].propertyFlags & want) == want) ai.memoryTypeIndex = i;
    if (ai.memoryTypeIndex == UINT32_MAX || m_fn.allocate(m_device, &ai, nullptr, &b.memory) || m_fn.bind(m_device, b.buffer, b.memory, 0))
    {
        Destroy(b);
        return {};
    }
    if (upload) m_fn.map(m_device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map);
    else b.address = Address(b.buffer);
    return b;
}
void CudaVulkanGraph::Destroy(Buffer& b)
{
    if (b.buffer) m_fn.destroyBuffer(m_device, b.buffer, nullptr);
    if (b.memory) m_fn.free(m_device, b.memory, nullptr);
    b = {};
}
uint64_t CudaVulkanGraph::Address(VkBuffer b) const
{
    VkBufferDeviceAddressInfo i = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, b};
    return m_fn.address(m_device, &i);
}
CudaVulkanGraph::~CudaVulkanGraph()
{
    if (!m_device) return;
    for (auto f : m_functions) if (f) m_fn.destroyFunction(m_device, f, nullptr);
    if (m_module) m_fn.destroyModule(m_device, m_module, nullptr);
    for (auto& t : m_tensors) Destroy(t);
    Destroy(m_constants), Destroy(m_upload);
}

void CudaVulkanGraph::Record(VkCommandBuffer cmd, CudaVulkanGraph* const* graphs, const uint64_t* inputs, const uint64_t (*outputs)[2], uint32_t count, uint32_t pitch)
{
    auto& f = graphs[0]->m_fn;
    auto barrier = [&](VkPipelineStageFlags from, VkAccessFlags a) {
        VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, a, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        f.barrier(cmd, from, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    };
    bool copied = false;
    for (uint32_t i = 0; i < count; ++i)
        if (!graphs[i]->m_uploaded)
        {
            VkBufferCopy c = {0, 0, graphs[i]->m_plan.constants.size()};
            f.copy(cmd, graphs[i]->m_upload.buffer, graphs[i]->m_constants.buffer, 1, &c);
            graphs[i]->m_uploaded = copied = true;
        }
    barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT | (copied ? VK_ACCESS_TRANSFER_WRITE_BIT : 0));
    std::vector<bool> dirty(graphs[0]->m_tensors.size(), false);
    for (size_t li = 0; li < graphs[0]->m_plan.layers.size(); ++li)
    {
        const auto& l = graphs[0]->m_plan.layers[li];
        bool wait = false;
        for (int32_t t : {l.src[0], l.src[1], l.dst}) wait = wait || (t >= 0 && dirty[t]);
        if (wait) barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT), std::fill(dirty.begin(), dirty.end(), false);
        for (uint32_t i = 0; i < count; ++i)
        {
            CudaVulkanGraph& g = *graphs[i];
            auto va = [&](int32_t t) { return t >= 0 ? g.m_tensors[t].address : t == -1 ? inputs[i] : outputs[i][-2 - t]; };
            const uint64_t a[5] = {va(l.src[0]), va(l.src[1]), g.m_constants.address + l.weights, g.m_constants.address + l.params, va(l.dst)};
            const void*    p[6] = {&a[0], &a[1], &a[2], &a[3], &a[4], &pitch};
            VkCuLaunchInfoNVX k = {VK_STRUCTURE_TYPE_CU_LAUNCH_INFO_NVX, nullptr, g.m_functions[li], l.grid[0], l.grid[1], l.grid[2], 128, 1, 1, 0, 6, p, 0, nullptr};
            f.launch(cmd, &k);
        }
        if (l.dst >= 0) dirty[l.dst] = true;
    }
    barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}
}  // namespace nfrux
