// Runs NFRUX's FSR Frame Generation 4 interpolator (core/mlfg) headlessly on D3D12 or Vulkan over frames from raw files,
// for tests/mlfg_net.py (which compares the outputs with tools/ffx_extract/mlfg_net.py, precision 'amd').
//   mlfg_run d3d12|vulkan DIR [--bench N]
// DIR/frames.txt:
//   size W H RW RH                                     display and render size; colour RGBA8 (W x H), depth R32F (inverted), mv RG16F in render pixels (RW x RH)
//   frame COLOR DEPTH MV RESET COUNT TOTAL m0 .. m15   one rendered frame: files in DIR, view-projection column-major
// Writes DIR/out_<api>_f<frame>_k<k>.bin (RGBA8, the generated frames) and DIR/t_<api>_<tensor>.bin (the NHWC fp16 tensors
// after the last frame). The tier is the runtime's choice or NFRUX_MLFG's. --bench N: the last frame N more times,
// 8 a submission, and the median GPU-inclusive time a frame (the run is short: a warm GPU is the caller's).
// SPDX-License-Identifier: MIT
#include "../core/interpolator.h"
#include "../core/ml/cuda_vulkan.h"
#include "rhi_nri.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

using namespace nfrux;
using Microsoft::WRL::ComPtr;

struct FrameDesc { std::string color, depth, mv; bool reset; uint32_t count, total; float vp[16]; };
static std::string g_dir, g_api;
static uint32_t g_w, g_h, g_rw, g_rh, g_bench = 0;
static std::vector<FrameDesc> g_frames;
struct Data { std::vector<uint8_t> color, depth, mv; };

static bool ReadAll(const std::string& p, std::vector<uint8_t>& d, size_t want)
{
    std::ifstream f(g_dir + "\\" + p, std::ios::binary);
    d.assign(std::istreambuf_iterator<char>(f), {});
    if (d.size() != want) return printf("%s: %zu bytes, want %zu\n", p.c_str(), d.size(), want), false;
    return true;
}
static void WriteAll(const std::string& p, const std::vector<uint8_t>& d)
{
    std::ofstream f(g_dir + "\\" + p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(d.data()), std::streamsize(d.size()));
}

static bool Run(Device& dev, const ExternalImage* im, const std::function<void(const Data&)>& upload, const std::function<void(const std::function<bool(void*)>&)>& submit)
{
    Settings s;
    s.model = Model::Amd, s.depthInverted = true, s.outputFormat = Format::RGBA8Unorm;
    s.displayWidth = g_w, s.displayHeight = g_h, s.renderWidth = g_rw, s.renderHeight = g_rh, s.colorLike = im[0];
    memcpy(s.initialViewProjection, g_frames[0].vp, 64);
    std::string error;
    auto ip = Interpolator::Create(dev, s, &error);
    if (!ip) return printf("create failed: %s\n", error.c_str()), false;
    bool ok = true;
    FrameInputs last;
    for (size_t i = 0; i < g_frames.size(); ++i)
    {
        const FrameDesc& f = g_frames[i];
        Data d;
        if (!ReadAll(f.color, d.color, size_t(g_w) * g_h * 4) || !ReadAll(f.depth, d.depth, size_t(g_rw) * g_rh * 4) || !ReadAll(f.mv, d.mv, size_t(g_rw) * g_rh * 4)) return false;
        upload(d);
        FrameInputs in;
        in.color = im[0], in.depth = im[1], in.motion = im[2], in.reset = f.reset;
        in.mvScale[0] = float(g_w) / float(g_rw), in.mvScale[1] = float(g_h) / float(g_rh);  // render pixels -> display pixels
        memcpy(in.viewProjection, f.vp, 64);
        std::vector<std::pair<std::string, std::shared_ptr<std::vector<uint8_t>>>> got;
        submit([&](void* cmd) {
            if (!dev.BeginFrame(cmd)) return ok = false;
            ok = ip->Dispatch(in, std::max(1u, f.count), std::max(1u, f.total)) && ok;
            for (uint32_t k = 0; k < f.count && !f.reset; ++k)
                got.push_back({"out_" + g_api + "_f" + std::to_string(i) + "_k" + std::to_string(k) + ".bin", std::make_shared<std::vector<uint8_t>>()}), dev.ReadBack(ip->Output(k), got.back().second);
            if (i + 1 == g_frames.size())
                for (auto& t : ip->Tensors())
                    got.push_back({"t_" + g_api + "_" + t.name + ".bin", std::make_shared<std::vector<uint8_t>>()}), dev.ReadBack(*t.buffer, got.back().second);
            dev.EndFrame();
            return ok;
        });
        dev.WaitIdle();
        for (auto& [n, v] : got) WriteAll(n, *v);
        last = in;
    }
    if (g_bench && ok)
    {
        const FrameDesc& f = g_frames.back();
        std::vector<double> ms;
        auto now = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        last.reset = false;
        for (uint32_t r = 0; r < g_bench; ++r)
        {
            double rec = 0;
            const double t0 = now();
            submit([&](void* cmd) {
                const double a = now();
                for (int j = 0; j < 8; ++j) { dev.BeginFrame(cmd); ip->Dispatch(last, std::max(1u, f.count), std::max(1u, f.total)); dev.EndFrame(); }
                rec = now() - a;
                return true;
            });
            if (r >= 2) ms.push_back((now() - t0 - rec) / 8);
        }
        std::sort(ms.begin(), ms.end());
        if (!ms.empty()) printf("bench %s %ux%u, %u generated: %.3f ms a rendered frame (median), %.3f fastest\n", g_api.c_str(), g_w, g_h, f.count, ms[ms.size() / 2], ms[0]);
    }
    ip.reset();
    dev.WaitIdle();
    return ok;
}

static bool RunD3D12()
{
    ComPtr<ID3D12Device> d;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d)))) return false;
    ComPtr<ID3D12CommandQueue> q;
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_COMPUTE};
    d->CreateCommandQueue(&qd, IID_PPV_ARGS(&q));
    ComPtr<ID3D12CommandAllocator> al;
    d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&al));
    ComPtr<ID3D12GraphicsCommandList> cl;
    d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, al.Get(), nullptr, IID_PPV_ARGS(&cl));
    cl->Close();
    ComPtr<ID3D12Fence> fence;
    d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    uint64_t fv = 0;
    auto execute = [&] {
        cl->Close();
        ID3D12CommandList* l = cl.Get();
        q->ExecuteCommandLists(1, &l), q->Signal(fence.Get(), ++fv), fence->SetEventOnCompletion(fv, nullptr);
    };
    const DXGI_FORMAT fm[3] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16_FLOAT};
    ComPtr<ID3D12Resource> tex[3];
    ExternalImage im[3];
    for (int i = 0; i < 3; ++i)
    {
        D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
        const uint32_t w = i ? g_rw : g_w, h = i ? g_rh : g_h;
        D3D12_RESOURCE_DESC r = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, w, h, 1, 1, fm[i], {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
        d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex[i]));
        im[i].state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, im[i].handle = tex[i].Get(), im[i].width = w, im[i].height = h;
    }
    bool uploaded = false;
    auto upload = [&](const Data& f) {
        al->Reset(), cl->Reset(al.Get(), nullptr);
        ComPtr<ID3D12Resource> up[3];
        const void* src[3] = {f.color.data(), f.depth.data(), f.mv.data()};
        for (int i = 0; i < 3; ++i)
        {
            const uint32_t row = im[i].width * 4, pitch = (row + 255) & ~255u, rows = im[i].height;
            D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_UPLOAD};
            D3D12_RESOURCE_DESC r = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, uint64_t(pitch) * rows, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
            d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&up[i]));
            uint8_t* p;
            up[i]->Map(0, nullptr, reinterpret_cast<void**>(&p));
            for (uint32_t y = 0; y < rows; ++y) memcpy(p + size_t(y) * pitch, static_cast<const uint8_t*>(src[i]) + size_t(y) * row, row);
            up[i]->Unmap(0, nullptr);
            D3D12_TEXTURE_COPY_LOCATION ld = {tex[i].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, ls = {up[i].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            ls.PlacedFootprint.Footprint = {fm[i], im[i].width, rows, 1, pitch};
            if (uploaded)
            {
                D3D12_RESOURCE_BARRIER c = {};
                c.Transition = {tex[i].Get(), 0, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST};
                cl->ResourceBarrier(1, &c);
            }
            cl->CopyTextureRegion(&ld, 0, 0, 0, &ls, nullptr);
            D3D12_RESOURCE_BARRIER b = {};
            b.Transition = {tex[i].Get(), 0, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
            cl->ResourceBarrier(1, &b);
        }
        uploaded = true;
        execute();
    };
    auto submit = [&](const std::function<bool(void*)>& record) { al->Reset(), cl->Reset(al.Get(), nullptr), record(cl.Get()), execute(); };
    auto dev = CreateNriD3D12Device(d.Get());
    return dev ? Run(*dev, im, upload, submit) : (printf("device failed\n"), false);
}

#define VKC(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("%s = %d\n", #x, r_); return false; } } while (0)
static bool RunVulkan()
{
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &ai;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    if (getenv("NFRUX_VK_VALIDATION")) ii.enabledLayerCount = 1, ii.ppEnabledLayerNames = &layer;
    VkInstance inst;
    VKC(vkCreateInstance(&ii, nullptr, &inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = pds[0];
    for (VkPhysicalDevice p : pds)
    {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = p; break; }
    }
    std::vector<const char*> ext;
    uint32_t ml = ffx_ml::QueryDeviceFeatures(inst, pd, vkGetInstanceProcAddr, &ext);
    ml |= AddCudaExtension(pd, ext);
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8;
    ml |= AddFloat8Extension(pd, ext, f8);
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, (ml & kFeatureFloat8) ? &f8 : nullptr};
    VkPhysicalDeviceVulkan13Features v13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &cm};
    VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    VkPhysicalDeviceVulkan11Features v11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    f2.features.robustBufferAccess = VK_FALSE;
    uint32_t qn = 0, family = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    for (uint32_t i = 0; i < qn; ++i) if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { family = i; break; }
    const float prio = 1;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, family, 1, &prio};
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    di.queueCreateInfoCount = 1, di.pQueueCreateInfos = &qi, di.enabledExtensionCount = uint32_t(ext.size()), di.ppEnabledExtensionNames = ext.data();
    VkDevice dv;
    VKC(vkCreateDevice(pd, &di, nullptr, &dv));
    VkQueue queue;
    vkGetDeviceQueue(dv, family, 0, &queue);
    VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family};
    VkCommandPool pool;
    VKC(vkCreateCommandPool(dv, &pi, nullptr, &pool));
    VkCommandBufferAllocateInfo ca = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    VKC(vkAllocateCommandBuffers(dv, &ca, &cb));
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    std::vector<VkDeviceMemory> mems;
    auto memory = [&](VkMemoryRequirements r, VkMemoryPropertyFlags want) {
        VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, r.size, 0};
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((r.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) { ma.memoryTypeIndex = i; break; }
        VkDeviceMemory m;
        vkAllocateMemory(dv, &ma, nullptr, &m);
        mems.push_back(m);
        return m;
    };
    auto submit = [&](const std::function<bool(void*)>& record) {
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cb, &bi), record(cb), vkEndCommandBuffer(cb);
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1, si.pCommandBuffers = &cb;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), vkQueueWaitIdle(queue);
    };
    const VkFormat fm[3] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16_SFLOAT};
    VkImage img[3];
    ExternalImage im[3];
    for (int i = 0; i < 3; ++i)
    {
        im[i].width = i ? g_rw : g_w, im[i].height = i ? g_rh : g_h, im[i].format = fm[i], im[i].state = VK_IMAGE_LAYOUT_GENERAL;
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ic.imageType = VK_IMAGE_TYPE_2D, ic.format = fm[i], ic.extent = {im[i].width, im[i].height, 1}, ic.mipLevels = ic.arrayLayers = 1, ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VKC(vkCreateImage(dv, &ic, nullptr, &img[i]));
        VkMemoryRequirements r;
        vkGetImageMemoryRequirements(dv, img[i], &r);
        vkBindImageMemory(dv, img[i], memory(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0);
        im[i].handle = reinterpret_cast<void*>(img[i]);
    }
    VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bc.size = uint64_t(g_w) * g_h * 12, bc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkBuffer up;
    VKC(vkCreateBuffer(dv, &bc, nullptr, &up));
    VkMemoryRequirements ur;
    vkGetBufferMemoryRequirements(dv, up, &ur);
    VkDeviceMemory um = memory(ur, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkBindBufferMemory(dv, up, um, 0);
    uint8_t* map;
    VKC(vkMapMemory(dv, um, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&map)));
    auto upload = [&](const Data& f) {
        const size_t sz = size_t(g_w) * g_h * 4, rz = size_t(g_rw) * g_rh * 4, off[3] = {0, sz, sz + rz};
        memcpy(map, f.color.data(), sz), memcpy(map + off[1], f.depth.data(), rz), memcpy(map + off[2], f.mv.data(), rz);
        submit([&](void*) {
            for (int i = 0; i < 3; ++i)
            {
                VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img[i], {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
                VkBufferImageCopy c = {off[i], 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {im[i].width, im[i].height, 1}};
                vkCmdCopyBufferToImage(cb, up, img[i], VK_IMAGE_LAYOUT_GENERAL, 1, &c);
                b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT, b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
            }
            return true;
        });
    };
    VulkanHost host;
    host.instance = inst, host.physicalDevice = pd, host.device = dv, host.gipa = vkGetInstanceProcAddr, host.gdpa = vkGetDeviceProcAddr;
    host.queueFamily = family, host.minorVersion = 2, host.deviceExtensions = ext.data(), host.deviceExtensionCount = uint32_t(ext.size());
    host.mlFeatures = ml;
    bool ok = false;
    {
        auto dev = CreateNriVulkanDevice(host);
        ok = dev ? Run(*dev, im, upload, submit) : (printf("device failed\n"), false);
    }
    for (auto& i : img) vkDestroyImage(dv, i, nullptr);
    vkDestroyBuffer(dv, up, nullptr);
    for (auto m : mems) vkFreeMemory(dv, m, nullptr);
    vkDestroyCommandPool(dv, pool, nullptr);
    vkDestroyDevice(dv, nullptr);
    vkDestroyInstance(inst, nullptr);
    return ok;
}

int main(int argc, char** argv)
{
    if (argc < 3) return puts("mlfg_run d3d12|vulkan DIR [--bench N]"), 2;
    g_api = argv[1], g_dir = argv[2];
    for (int i = 3; i < argc; ++i)
        if (!strcmp(argv[i], "--bench") && i + 1 < argc) g_bench = uint32_t(atoi(argv[++i]));
    std::ifstream f(g_dir + "\\frames.txt");
    std::string line;
    while (std::getline(f, line))
    {
        std::istringstream s(line);
        std::string k;
        if (!(s >> k) || k[0] == '#') continue;
        if (k == "size") s >> g_w >> g_h >> g_rw >> g_rh;
        else if (k == "frame")
        {
            FrameDesc d;
            int reset;
            s >> d.color >> d.depth >> d.mv >> reset >> d.count >> d.total;
            d.reset = reset != 0;
            for (float& x : d.vp) s >> x;
            g_frames.push_back(d);
        }
    }
    if (!g_w || g_frames.empty()) return puts("no frames"), 2;
    const bool ok = g_api == "vulkan" ? RunVulkan() : RunD3D12();
    printf("%s %s\n", g_api.c_str(), ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
