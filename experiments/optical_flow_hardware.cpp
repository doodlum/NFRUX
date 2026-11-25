// A12 spike: NVIDIA's optical flow hardware (VK_NV_optical_flow) on pairs of luma images, offline.
//   optical_flow_hardware W H COUNT DIR     DIR\a<i>.bin, DIR\b<i>.bin (R8) -> DIR\f<i>.bin (int16 x 2, 1/32 pixel, 4x4 grid, a to b)
// SPDX-License-Identifier: MIT
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define VKC(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("%s = %d\n", #x, r_); return 1; } } while (0)
int main(int argc, char** argv)
{
    if (argc < 5) return 2;
    const uint32_t W = atoi(argv[1]), H = atoi(argv[2]), N = atoi(argv[3]), G = 4, FW = (W + G - 1) / G, FH = (H + G - 1) / G;
    const std::string dir = argv[4];
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &ai;
    VkInstance inst;
    VKC(vkCreateInstance(&ii, nullptr, &inst));
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t gfx = 0, of = ~0u;
    for (uint32_t i = 0; i < qn; ++i) if (qf[i].queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV) of = i;
    if (of == ~0u) return puts("no optical flow queue"), 1;
    VkPhysicalDeviceOpticalFlowPropertiesNV op = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES_NV};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &op};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    printf("%s: output grids %x, hint %d, cost %d\n", p2.properties.deviceName, op.supportedOutputGridSizes, op.hintSupported, op.costSupported);
    const float prio = 1;
    VkDeviceQueueCreateInfo qi[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, gfx, 1, &prio}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, of, 1, &prio}};
    VkPhysicalDeviceOpticalFlowFeaturesNV off = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV, nullptr, VK_TRUE};
    VkPhysicalDeviceVulkan13Features v13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &off};
    v13.synchronization2 = VK_TRUE;
    const char* ext[] = {VK_NV_OPTICAL_FLOW_EXTENSION_NAME};
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &v13};
    di.queueCreateInfoCount = 2, di.pQueueCreateInfos = qi, di.enabledExtensionCount = 1, di.ppEnabledExtensionNames = ext;
    VkDevice d;
    VKC(vkCreateDevice(pd, &di, nullptr, &d));
    VkQueue q[2];
    vkGetDeviceQueue(d, gfx, 0, &q[0]), vkGetDeviceQueue(d, of, 0, &q[1]);
    VkCommandBuffer cb[2];
    for (int i = 0; i < 2; ++i)
    {
        VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, i ? of : gfx};
        VkCommandPool pool;
        VKC(vkCreateCommandPool(d, &pi, nullptr, &pool));
        VkCommandBufferAllocateInfo ca = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VKC(vkAllocateCommandBuffers(d, &ca, &cb[i]));
    }
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memory = [&](VkMemoryRequirements r, VkMemoryPropertyFlags want) {
        VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, r.size, 0};
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((r.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) { ma.memoryTypeIndex = i; break; }
        VkDeviceMemory m;
        vkAllocateMemory(d, &ma, nullptr, &m);
        return m;
    };
    auto submit = [&](int i, auto rec) {
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cb[i], &bi), rec(cb[i]), vkEndCommandBuffer(cb[i]);
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1, si.pCommandBuffers = &cb[i];
        vkQueueSubmit(q[i], 1, &si, VK_NULL_HANDLE), vkQueueWaitIdle(q[i]);
    };
    VkImage img[3];
    VkImageView view[3];
    const VkFormat fm[3] = {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R16G16_SFIXED5_NV};
    const uint32_t fam[2] = {gfx, of};
    for (int i = 0; i < 3; ++i)
    {
        VkOpticalFlowImageFormatInfoNV fi = {VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV, nullptr, VkOpticalFlowUsageFlagsNV(i < 2 ? VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV : VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV)};
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &fi};
        ic.imageType = VK_IMAGE_TYPE_2D, ic.format = fm[i], ic.extent = {i < 2 ? W : FW, i < 2 ? H : FH, 1}, ic.mipLevels = ic.arrayLayers = 1, ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ic.sharingMode = VK_SHARING_MODE_CONCURRENT, ic.queueFamilyIndexCount = 2, ic.pQueueFamilyIndices = fam;
        VKC(vkCreateImage(d, &ic, nullptr, &img[i]));
        VkMemoryRequirements r;
        vkGetImageMemoryRequirements(d, img[i], &r);
        vkBindImageMemory(d, img[i], memory(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0);
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img[i], vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = fm[i], vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VKC(vkCreateImageView(d, &vi, nullptr, &view[i]));
    }
    VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bc.size = W * H * 2 + FW * FH * 4, bc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf;
    VKC(vkCreateBuffer(d, &bc, nullptr, &buf));
    VkMemoryRequirements br;
    vkGetBufferMemoryRequirements(d, buf, &br);
    VkDeviceMemory bm = memory(br, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkBindBufferMemory(d, buf, bm, 0);
    uint8_t* map;
    vkMapMemory(d, bm, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&map));
    auto create = reinterpret_cast<PFN_vkCreateOpticalFlowSessionNV>(vkGetDeviceProcAddr(d, "vkCreateOpticalFlowSessionNV"));
    auto bind = reinterpret_cast<PFN_vkBindOpticalFlowSessionImageNV>(vkGetDeviceProcAddr(d, "vkBindOpticalFlowSessionImageNV"));
    auto exec = reinterpret_cast<PFN_vkCmdOpticalFlowExecuteNV>(vkGetDeviceProcAddr(d, "vkCmdOpticalFlowExecuteNV"));
    const char* level = getenv("NVOF_LEVEL");
    VkOpticalFlowSessionCreateInfoNV si = {VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV};
    si.width = W, si.height = H, si.imageFormat = fm[0], si.flowVectorFormat = fm[2], si.outputGridSize = VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV;
    si.performanceLevel = VkOpticalFlowPerformanceLevelNV(level ? atoi(level) : 1);  // 1 slow, 2 medium, 3 fast
    VkOpticalFlowSessionNV s;
    VKC(create(d, &si, nullptr, &s));
    const VkOpticalFlowSessionBindingPointNV pt[3] = {VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV};
    for (int i = 0; i < 3; ++i) VKC(bind(d, s, pt[i], view[i], VK_IMAGE_LAYOUT_GENERAL));
    auto barrier = [&](VkCommandBuffer c, VkImageLayout from) {
        VkImageMemoryBarrier b[3];
        for (int i = 0; i < 3; ++i)
            b[i] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, from, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img[i], {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 3, b);
    };
    submit(0, [&](VkCommandBuffer c) { barrier(c, VK_IMAGE_LAYOUT_UNDEFINED); });
    std::vector<double> ms;
    for (uint32_t i = 0; i < N; ++i)
    {
        for (int k = 0; k < 2; ++k)
        {
            FILE* f = fopen((dir + "\\" + (k ? "b" : "a") + std::to_string(i) + ".bin").c_str(), "rb");
            if (!f || fread(map + size_t(k) * W * H, 1, size_t(W) * H, f) != size_t(W) * H) return printf("pair %u missing\n", i), 1;
            fclose(f);
        }
        submit(0, [&](VkCommandBuffer c) {
            barrier(c, VK_IMAGE_LAYOUT_GENERAL);
            for (int k = 0; k < 2; ++k)
            {
                VkBufferImageCopy r = {VkDeviceSize(k) * W * H, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {W, H, 1}};
                vkCmdCopyBufferToImage(c, buf, img[k], VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            }
            barrier(c, VK_IMAGE_LAYOUT_GENERAL);
        });
        const auto t0 = std::chrono::steady_clock::now();
        submit(1, [&](VkCommandBuffer c) { VkOpticalFlowExecuteInfoNV e = {VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV}; e.flags = VK_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_NV; for (int j = 0; j < 8; ++j) exec(c, s, &e); });
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 8);
        submit(0, [&](VkCommandBuffer c) {
            barrier(c, VK_IMAGE_LAYOUT_GENERAL);
            VkBufferImageCopy r = {VkDeviceSize(2) * W * H, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {FW, FH, 1}};
            vkCmdCopyImageToBuffer(c, img[2], VK_IMAGE_LAYOUT_GENERAL, buf, 1, &r);
            barrier(c, VK_IMAGE_LAYOUT_GENERAL);
        });
        if (FILE* f = fopen((dir + "\\f" + std::to_string(i) + ".bin").c_str(), "wb")) fwrite(map + size_t(W) * H * 2, 4, size_t(FW) * FH, f), fclose(f);
    }
    std::sort(ms.begin(), ms.end());
    printf("%ux%u, %u pairs, level %d: median %.2f ms an execution (wall, 8 back to back)\n", W, H, N, int(si.performanceLevel), ms[ms.size() / 2]);
    return 0;
}