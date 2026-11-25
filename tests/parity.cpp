// A2 (offline): the same synthetic frames through every backend, every intermediate and output compared.
//   parity [--size WxH] [--render WxH] [--frames N] [--gen N] [--dump DIR] [backend ...]   backends: d3d12 nri12 nrivk
// SPDX-License-Identifier: MIT
#include "../core/interpolator.h"
#include "../core/ml/ffx_vk_ml_runtime.h"
#include "rhi_d3d12.h"
#include "rhi_nri.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

using namespace nfrux;
using Microsoft::WRL::ComPtr;

struct Frame
{
    std::vector<uint8_t>  color;   // RGBA8
    std::vector<float>    depth;
    std::vector<uint16_t> motion;  // RG16F
};
static uint16_t Half(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    const int      e = int((x >> 23) & 255) - 112;
    if (e <= 0) return uint16_t(s);
    return uint16_t(s | (std::min(e, 30) << 10) | ((x >> 13) & 1023));
}
static float Noise(int x, int y) { uint32_t h = uint32_t(x) * 374761393u + uint32_t(y) * 668265263u; h = (h ^ (h >> 13)) * 1274126177u; return float((h ^ (h >> 16)) & 255) / 255.0f; }
static float Smooth(float x, float y)
{
    const int ix = int(floorf(x)), iy = int(floorf(y));
    const float fx = x - ix, fy = y - iy;
    const float a = Noise(ix, iy) * (1 - fx) + Noise(ix + 1, iy) * fx, b = Noise(ix, iy + 1) * (1 - fx) + Noise(ix + 1, iy + 1) * fx;
    return a * (1 - fy) + b * fy;
}
// A textured background panning by (6, 2) pixels a frame and a nearer square moving by (-14, 5) against it.
static Frame Make(uint32_t w, uint32_t h, uint32_t rw, uint32_t rh, float t)
{
    Frame f;
    f.color.resize(size_t(w) * h * 4), f.depth.resize(size_t(rw) * rh), f.motion.resize(size_t(rw) * rh * 2);
    const float bx = 6.0f * t, by = 2.0f * t, sx = w * 0.55f - 14.0f * t, sy = h * 0.35f + 5.0f * t, sw = w * 0.18f;
    auto inside = [&](float x, float y) { return x >= sx && x < sx + sw && y >= sy && y < sy + sw; };
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
        {
            uint8_t* p = &f.color[(size_t(y) * w + x) * 4];
            const bool in = inside(x + 0.5f, y + 0.5f);
            const float u = in ? x - sx : x - bx, v = in ? y - sy : y - by;
            for (int k = 0; k < 3; ++k)
            {
                const float n = 0.6f * Smooth(u / 23.0f + k * 91, v / 23.0f) + 0.4f * Smooth(u / 5.0f + k * 37, v / 5.0f + 11);
                p[k] = uint8_t(std::clamp((in ? 0.25f + 0.75f * n : n * 0.8f) * 255.0f, 0.0f, 255.0f));
            }
            p[3] = 255;
        }
    for (uint32_t y = 0; y < rh; ++y)
        for (uint32_t x = 0; x < rw; ++x)
        {
            const bool in = inside((x + 0.5f) * w / rw, (y + 0.5f) * h / rh);
            f.depth[size_t(y) * rw + x] = in ? 0.90f : 0.99f;
            f.motion[(size_t(y) * rw + x) * 2 + 0] = Half(in ? 14.0f : -6.0f);  // to the previous frame, display pixels
            f.motion[(size_t(y) * rw + x) * 2 + 1] = Half(in ? -5.0f : -2.0f);
        }
    return f;
}

using Results = std::map<std::string, std::vector<uint8_t>>;
struct Opt { uint32_t w = 1280, h = 720, rw = 1280, rh = 720, frames = 8, gen = 2; std::string dump; } g;

static FrameInputs Inputs(const ExternalImage* im, bool first)
{
    FrameInputs in;
    in.color = im[0], in.depth = im[1], in.motion = im[2], in.reset = first;
    in.mvScale[0] = in.mvScale[1] = 1.0f;
    const float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    memcpy(in.viewProjection, id, sizeof(id));
    in.cameraNear = 0.1f, in.cameraFar = 100.0f, in.fovY = 1.0f;
    return in;
}
static Settings MakeSettings(const ExternalImage& color)
{
    Settings s;
    s.displayWidth = g.w, s.displayHeight = g.h, s.renderWidth = g.rw, s.renderHeight = g.rh, s.colorLike = color;
    for (int i = 0; i < 4; ++i) s.initialViewProjection[i * 5] = 1.0f;
    return s;
}
// Runs every frame: upload, record(commands), submit and wait are the API's.
static bool Run(Device& dev, const ExternalImage* im, const std::function<void(const Frame&)>& upload, const std::function<void(const std::function<bool(void*)>&)>& submit, Results& out)
{
    std::string error;
    auto ip = Interpolator::Create(dev, MakeSettings(im[0]), &error);
    if (!ip) return printf("  create failed: %s\n", error.c_str()), false;
    bool ok = true;
    for (uint32_t i = 0; i < g.frames; ++i)
    {
        upload(Make(g.w, g.h, g.rw, g.rh, float(i)));
        std::vector<std::pair<std::string, std::shared_ptr<std::vector<uint8_t>>>> got;
        submit([&](void* cmd) {
            if (!dev.BeginFrame(cmd)) return ok = false;
            ok = ip->Dispatch(Inputs(im, i == 0), g.gen, g.gen) && ok;
            if (i)
            {
                for (auto& [name, t] : ip->Intermediates())
                    if (t && i == g.frames - 1) got.push_back({name, std::make_shared<std::vector<uint8_t>>()}), dev.ReadBack(*t, got.back().second);
                for (uint32_t k = 0; k < g.gen; ++k)
                    got.push_back({"frame" + std::to_string(i) + "_gen" + std::to_string(k), std::make_shared<std::vector<uint8_t>>()}), dev.ReadBack(ip->Output(k), got.back().second);
            }
            dev.EndFrame();
            return ok;
        });
        dev.WaitIdle();
        for (auto& [n, v] : got) out[n] = *v;
    }
    ip.reset();
    return ok;
}

// --- D3D12
static bool RunD3D12(bool nri, Results& out)
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
        const uint32_t w = i ? g.rw : g.w, h = i ? g.rh : g.h;
        D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC r = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, w, h, 1, 1, fm[i], {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
        d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex[i]));
        im[i].handle = tex[i].Get(), im[i].width = w, im[i].height = h;
    }
    auto upload = [&](const Frame& f) {
        al->Reset(), cl->Reset(al.Get(), nullptr);
        ComPtr<ID3D12Resource> up[3];
        const void* src[3] = {f.color.data(), f.depth.data(), f.motion.data()};
        for (int i = 0; i < 3; ++i)
        {
            const uint32_t row = im[i].width * 4, pitch = (row + 255) & ~255u;
            D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_UPLOAD};
            D3D12_RESOURCE_DESC r = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, uint64_t(pitch) * im[i].height, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
            d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&up[i]));
            uint8_t* p;
            up[i]->Map(0, nullptr, reinterpret_cast<void**>(&p));
            for (uint32_t y = 0; y < im[i].height; ++y) memcpy(p + size_t(y) * pitch, static_cast<const uint8_t*>(src[i]) + size_t(y) * row, row);
            up[i]->Unmap(0, nullptr);
            D3D12_TEXTURE_COPY_LOCATION ld = {tex[i].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, ls = {up[i].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            ls.PlacedFootprint.Footprint = {fm[i], im[i].width, im[i].height, 1, pitch};
            cl->CopyTextureRegion(&ld, 0, 0, 0, &ls, nullptr);
            D3D12_RESOURCE_BARRIER b = {};
            b.Transition = {tex[i].Get(), 0, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON};
            cl->ResourceBarrier(1, &b);
        }
        execute();
    };
    auto submit = [&](const std::function<bool(void*)>& record) { al->Reset(), cl->Reset(al.Get(), nullptr), record(cl.Get()), execute(); };
    auto dev = nri ? CreateNriD3D12Device(d.Get()) : CreateD3D12Device(d.Get());
    return dev ? Run(*dev, im, upload, submit, out) : (printf("  device failed\n"), false);
}

// --- Vulkan
#define VKC(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("  %s = %d\n", #x, r_); return false; } } while (0)
static bool RunVulkan(Results& out)
{
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &ai;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    if (getenv("NFRUX_VK_VALIDATION")) ii.enabledLayerCount = 1, ii.ppEnabledLayerNames = &layer;
    VkInstance inst;
    VKC(vkCreateInstance(&ii, nullptr, &inst));
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    std::vector<const char*> ext;
    const uint32_t ml = ffx_ml::QueryDeviceFeatures(inst, pd, vkGetInstanceProcAddr, &ext);
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR, &cm};
    VkPhysicalDeviceVulkan13Features v13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &bf};
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
    auto memory = [&](VkMemoryRequirements r, VkMemoryPropertyFlags want) {
        VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, r.size, 0};
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((r.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) { ma.memoryTypeIndex = i; break; }
        VkDeviceMemory m;
        vkAllocateMemory(dv, &ma, nullptr, &m);
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
        im[i].width = i ? g.rw : g.w, im[i].height = i ? g.rh : g.h, im[i].format = fm[i], im[i].state = VK_IMAGE_LAYOUT_GENERAL;
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
    bc.size = uint64_t(g.w) * g.h * 4 + uint64_t(g.rw) * g.rh * 8, bc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkBuffer up;
    VKC(vkCreateBuffer(dv, &bc, nullptr, &up));
    VkMemoryRequirements ur;
    vkGetBufferMemoryRequirements(dv, up, &ur);
    VkDeviceMemory um = memory(ur, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkBindBufferMemory(dv, up, um, 0);
    uint8_t* map;
    VKC(vkMapMemory(dv, um, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&map)));
    auto upload = [&](const Frame& f) {
        const size_t off[3] = {0, f.color.size(), f.color.size() + f.depth.size() * 4};
        memcpy(map, f.color.data(), f.color.size()), memcpy(map + off[1], f.depth.data(), f.depth.size() * 4), memcpy(map + off[2], f.motion.data(), f.motion.size() * 2);
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
    host.queueFamily = family, host.minorVersion = 3, host.deviceExtensions = ext.data(), host.deviceExtensionCount = uint32_t(ext.size());
    host.mlFeatures = ml, host.atomic64 = v12.shaderBufferInt64Atomics && f2.features.shaderInt64;
    auto dev = CreateNriVulkanDevice(host);
    return dev ? Run(*dev, im, upload, submit, out) : (printf("  device failed\n"), false);
}

int main(int argc, char** argv)
{
    std::vector<std::string> which;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--size") sscanf(argv[++i], "%ux%u", &g.w, &g.h), g.rw = g.w, g.rh = g.h;
        else if (a == "--render") sscanf(argv[++i], "%ux%u", &g.rw, &g.rh);
        else if (a == "--frames") g.frames = atoi(argv[++i]);
        else if (a == "--gen") g.gen = atoi(argv[++i]);
        else if (a == "--dump") g.dump = argv[++i];
        else which.push_back(a);
    }
    if (which.empty()) which = {"d3d12", "nri12", "nrivk"};
    std::vector<Results> all;
    int bad = 0;
    for (auto& w : which)
    {
        printf("%s %ux%u render %ux%u gen %u\n", w.c_str(), g.w, g.h, g.rw, g.rh, g.gen);
        Results r;
        const bool ok = w == "nrivk" ? RunVulkan(r) : RunD3D12(w == "nri12", r);
        if (!ok) printf("  FAILED\n"), ++bad;
        // Against the truth: what the frame between two rendered ones really is.
        double sum = 0;
        uint32_t cnt = 0;
        for (uint32_t i = 1; i < g.frames && ok; ++i)
            for (uint32_t k = 0; k < g.gen; ++k)
            {
                const auto& o = r["frame" + std::to_string(i) + "_gen" + std::to_string(k)];
                const Frame t = Make(g.w, g.h, 16, 16, float(i - 1) + float(k + 1) / float(g.gen + 1));
                if (o.size() != t.color.size()) continue;
                double e = 0;
                for (size_t p = 0; p < o.size(); ++p) if ((p & 3) != 3) e += (double(o[p]) - t.color[p]) * (double(o[p]) - t.color[p]);
                sum += 10 * log10(255.0 * 255.0 / std::max(1e-9, e / (o.size() * 0.75))), ++cnt;
            }
        if (cnt) printf("  %u generated frames, mean PSNR against the truth %.2f dB\n", cnt, sum / cnt);
        if (!g.dump.empty())
            for (auto& [n, v] : r)
                if (FILE* f = fopen((g.dump + "\\" + w + "_" + n + ".bin").c_str(), "wb")) fwrite(v.data(), 1, v.size(), f), fclose(f);
        all.push_back(std::move(r));
    }
    for (size_t b = 1; b < all.size(); ++b)
    {
        printf("%s against %s\n", which[b].c_str(), which[0].c_str());
        size_t frames = 0, total = 0, off1 = 0, off3 = 0;
        double worst = 1e9;
        for (auto& [n, v] : all[0])
        {
            const auto& o = all[b][n];
            if (o.size() != v.size()) { printf("  %-14s sizes %zu %zu\n", n.c_str(), v.size(), o.size()), ++bad; continue; }
            size_t d = 0, d1 = 0, d3 = 0;
            double e = 0;
            for (size_t i = 0; i < v.size(); ++i)
            {
                const int x = abs(int(v[i]) - int(o[i]));
                d += x != 0, d1 += x > 1, d3 += x > 3, e += double(x) * x;
            }
            const double psnr = 10 * log10(255.0 * 255.0 / std::max(1e-9, e / std::max<size_t>(1, v.size())));
            if (n.rfind("frame", 0) == 0) ++frames, total += v.size(), off1 += d1, off3 += d3, worst = std::min(worst, psnr);
            else printf("  %-14s %9zu bytes, %zu differ (%.4f%%)\n", n.c_str(), v.size(), d, 100.0 * d / std::max<size_t>(1, v.size()));
        }
        printf("  %zu generated frames: worst PSNR %.2f dB, off by more than 1: %.5f%%, by more than 3: %.5f%%\n", frames, worst, 100.0 * off1 / std::max<size_t>(1, total), 100.0 * off3 / std::max<size_t>(1, total));
    }
    return bad != 0;
}
