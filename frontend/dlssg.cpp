// nvngx_dlssg.dll: the NGX entry points of frame generation (feature 11) for D3D12 and Vulkan. SPDX-License-Identifier: MIT
#include "frame_generator.h"

#include "../core/log.h"
#include "../core/ml/ffx_vk_ml_runtime.h"
#include "../loader/ngx.h"
#include "rhi_nri.h"

#include <windows.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>

using namespace nfrux;
#define NFRUX_EXPORT extern "C" __declspec(dllexport) NgxResult __cdecl

namespace
{
std::mutex g_mutex;
struct Vk { VkInstance i = VK_NULL_HANDLE; VkPhysicalDevice pd = VK_NULL_HANDLE; VkDevice d = VK_NULL_HANDLE; PFN_vkGetInstanceProcAddr gi = nullptr; PFN_vkGetDeviceProcAddr gd = nullptr; } g_vk;
uint32_t g_vkMl = 0;          // known enabled (NFRUX_SetVulkanFeatures)
bool     g_vkAtomic64 = false, g_vkKnown = false;
struct Feature { NgxHandle handle; std::shared_ptr<FrameGenerator> fg; bool vulkan; void* host; };
std::map<const NgxHandle*, std::unique_ptr<Feature>> g_features;
std::map<void*, std::shared_ptr<FrameGenerator>> g_warm;  // released features' generators, by host device, for the next create
unsigned g_next = 0x4e460001;

uint32_t MaxFrames()
{
    const char* v = Env("NFRUX_MAX_FRAMES");
    return v ? std::clamp(atoi(v), 1, int(kMaxGenerated)) : kMaxGenerated;
}
std::unique_ptr<Device> DeviceForVk()
{
    if (!g_vk.d) return nullptr;
    VulkanHost h;
    h.instance = g_vk.i, h.physicalDevice = g_vk.pd, h.device = g_vk.d, h.gipa = g_vk.gi, h.gdpa = g_vk.gd;
    if (HMODULE m = GetModuleHandleA("vulkan-1.dll"); m && (!h.gipa || !h.gdpa))
        h.gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(m, "vkGetInstanceProcAddr")),
        h.gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(m, "vkGetDeviceProcAddr"));
    if (!h.gipa || !h.gdpa) return nullptr;
    VkPhysicalDeviceProperties p = {};
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(h.gipa(h.instance, "vkGetPhysicalDeviceProperties"))(h.physicalDevice, &p);
    h.minorVersion = std::clamp(VK_API_VERSION_MINOR(p.apiVersion), 2u, 3u);
    if (g_vkKnown) h.mlFeatures = g_vkMl, h.atomic64 = g_vkAtomic64;
    else if (EnvFlag("NFRUX_VK_ASSUME_FEATURES", false))
    {
        std::vector<const char*> ext;
        h.mlFeatures = ffx_ml::QueryDeviceFeatures(h.instance, h.physicalDevice, h.gipa, &ext);
        Log("NFRUX_VK_ASSUME_FEATURES: using device features whether or not the game enabled them");
    }
    return CreateNriVulkanDevice(h);
}
NgxResult Add(void* host, bool vulkan, NgxHandle** out)
{
    auto f    = std::make_unique<Feature>();
    f->handle = {g_next++}, f->vulkan = vulkan, f->host = host;
    if (auto w = g_warm.find(host); w != g_warm.end())
        f->fg = w->second, f->fg->Restart(), g_warm.erase(w);
    else if (auto dev = vulkan ? DeviceForVk() : CreateNriD3D12Device(static_cast<ID3D12Device*>(host)))
        f->fg = std::make_shared<FrameGenerator>(std::move(dev), vulkan);
    else return kNgxPlatformError;
    *out      = &f->handle;
    Log("feature %x created (%s)", f->handle.Id, vulkan ? "Vulkan" : "D3D12");
    g_features[&f->handle] = std::move(f);
    return kNgxSuccess;
}
Feature* Find(const NgxHandle* h)
{
    auto it = g_features.find(h);
    return it == g_features.end() ? nullptr : it->second.get();
}

int Int(NgxParameter* p, const char* n, int d = 0)
{
    int i;
    if (p->Get(n, &i) == kNgxSuccess) return i;
    unsigned u;
    return p->Get(n, &u) == kNgxSuccess ? int(u) : d;
}
float Float(NgxParameter* p, const char* n, float d = 0) { float f; return p->Get(n, &f) == kNgxSuccess ? f : d; }
void* Ptr(NgxParameter* p, const char* n) { void* v = nullptr; return p->Get(n, &v) == kNgxSuccess ? v : nullptr; }
Rect GetRect(NgxParameter* p, std::string n)
{
    n = "DLSSG." + n + "Subrect";
    return {uint32_t(Int(p, (n + "BaseX").c_str())), uint32_t(Int(p, (n + "BaseY").c_str())), uint32_t(Int(p, (n + "Width").c_str())), uint32_t(Int(p, (n + "Height").c_str()))};
}
ExternalImage Image12(NgxParameter* p, const char* n)
{
    ID3D12Resource* r = nullptr;
    if (p->Get(n, &r) != kNgxSuccess || !r) r = static_cast<ID3D12Resource*>(Ptr(p, n));
    ExternalImage e;
    if (!r) return e;
    const D3D12_RESOURCE_DESC d = r->GetDesc();
    e.handle = r, e.format = d.Format, e.width = uint32_t(d.Width), e.height = d.Height, e.state = D3D12_RESOURCE_STATE_COMMON;
    return e;
}
ExternalImage ImageVk(NgxParameter* p, const char* n)
{
    auto* r = static_cast<NgxResourceVk*>(Ptr(p, n));
    ExternalImage e;
    if (!r || r->Type != 0 || !r->R.Image.Image) return e;
    e.handle = reinterpret_cast<void*>(r->R.Image.Image), e.format = r->R.Image.Format, e.width = r->R.Image.Width, e.height = r->R.Image.Height;
    e.state = VK_IMAGE_LAYOUT_GENERAL;
    return e;
}
EvalInputs Read(NgxParameter* p, bool vk)
{
    auto image = [&](const char* n) { return vk ? ImageVk(p, n) : Image12(p, n); };
    EvalInputs in;
    in.back = image("DLSSG.Backbuffer"), in.output = image("DLSSG.OutputInterpolated"), in.outputReal = image("DLSSG.OutputReal");
    in.hudless = image("DLSSG.HUDLess"), in.ui = image("DLSSG.UI"), in.uiAlpha = image("DLSSG.UIAlpha"), in.noWarp = image("DLSSG.NoWarp");
    in.depth = image("DLSSG.Depth"), in.motion = image("DLSSG.MVecs");
    in.backRect = GetRect(p, "Backbuffer"), in.hudlessRect = GetRect(p, "HUDLess"), in.uiRect = GetRect(p, "UI"), in.uiAlphaRect = GetRect(p, "UIAlpha");
    in.noWarpRect = GetRect(p, "NoWarp"), in.depthRect = GetRect(p, "Depth"), in.motionRect = GetRect(p, "MVecs");
    in.mvScale[0] = Float(p, "DLSSG.MvecScaleX", 1), in.mvScale[1] = Float(p, "DLSSG.MvecScaleY", 1);
    in.jitter[0] = Float(p, "DLSSG.JitterOffsetX"), in.jitter[1] = Float(p, "DLSSG.JitterOffsetY");
    in.cameraNear = Float(p, "DLSSG.CameraNear", 0.1f), in.cameraFar = Float(p, "DLSSG.CameraFar", 1000.0f), in.fovY = Float(p, "DLSSG.CameraFOV", 1.0f);
    in.depthInverted = Int(p, "DLSSG.DepthInverted") != 0, in.hdr = Int(p, "DLSSG.ColorBuffersHDR") != 0, in.reset = Int(p, "DLSSG.Reset") != 0;
    in.notRendering = Int(p, "DLSSG.NotRenderingGameFrames") != 0, in.ortho = Int(p, "DLSSG.OrthoProjection") != 0;
    in.count = uint32_t(std::max(1, Int(p, "DLSSG.MultiFrameCount", 1))), in.index = uint32_t(std::max(1, Int(p, "DLSSG.MultiFrameIndex", 1)));
    if (auto* m = static_cast<const float*>(Ptr(p, "DLSSG.CameraViewToClip"))) memcpy(in.viewToClip, m, 64);
    if (auto* m = static_cast<const float*>(Ptr(p, "DLSSG.ClipToPrevClip"))) memcpy(in.clipToPrevClip, m, 64), in.hasClipToPrevClip = true;
    const char axis[3] = {'X', 'Y', 'Z'};
    for (int k = 0; k < 3; ++k)
    {
        const std::string a(1, axis[k]);
        in.pos[k] = Float(p, ("DLSSG.CameraPos" + a).c_str()), in.up[k] = Float(p, ("DLSSG.CameraUp" + a).c_str());
        in.right[k] = Float(p, ("DLSSG.CameraRight" + a).c_str()), in.fwd[k] = Float(p, ("DLSSG.CameraFwd" + a).c_str());
    }
    return in;
}
NgxResult Eval(void* cmd, const NgxHandle* h, NgxParameter* p)
{
    std::shared_ptr<FrameGenerator> fg;  // alive through this call even if released meanwhile
    bool            vk;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        Feature* f = Find(h);
        if (!f || !p || !cmd) return f ? kNgxInvalidParameter : kNgxFeatureNotFound;
        fg = f->fg, vk = f->vulkan;
    }
    return fg->Evaluate(cmd, Read(p, vk)) ? kNgxSuccess : kNgxInvalidParameter;
}
NgxResult Release(NgxHandle* h)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    Feature* f = Find(h);
    if (!f) return kNgxFeatureNotFound;
    g_warm[f->host] = f->fg;
    g_features.erase(h);
    Log("feature released");
    return kNgxSuccess;
}
NgxResult Populate(NgxParameter* p)
{
    if (!p) return kNgxInvalidParameter;
    p->Set("FrameGeneration.Available", 1), p->Set("FrameGeneration.NeedsUpdatedDriver", 0);
    p->Set("FrameGeneration.MinDriverVersionMajor", 0u), p->Set("FrameGeneration.MinDriverVersionMinor", 0u);
    p->Set("FrameGeneration.FeatureInitResult", int(kNgxSuccess));
    p->Set("DLSSG.MultiFrameCountMax", MaxFrames());
    return kNgxSuccess;
}
NgxResult Requirements(NgxFeatureRequirement* r)
{
    if (!r) return kNgxInvalidParameter;
    memset(r, 0, sizeof(*r));
    strcpy_s(r->MinOSVersion, "10.0.0");
    return kNgxSuccess;
}
void SetVk(VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_vk = {i, pd, d, gi, gd};
}
// Of one API (and device, when given): the other API's features stay.
void Drop(bool vulkan, void* host)
{
    std::vector<std::shared_ptr<FrameGenerator>> dead;  // destroyed after the lock
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto it = g_features.begin(); it != g_features.end();)
        if (it->second->vulkan == vulkan && (!host || it->second->host == host)) { dead.push_back(it->second->fg), it = g_features.erase(it); } else ++it;
    for (auto it = g_warm.begin(); it != g_warm.end();)
        if (it->second->vulkan() == vulkan && (!host || it->first == host)) { dead.push_back(it->second), it = g_warm.erase(it); } else ++it;
}
}  // namespace

// --- D3D12
NFRUX_EXPORT NVSDK_NGX_D3D12_Init(unsigned long long, const wchar_t*, ID3D12Device*, const void*, int) { return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_D3D12_Init_Ext(unsigned long long, const wchar_t*, ID3D12Device*, int, const void*) { return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_D3D12_Shutdown() { Drop(false, nullptr); return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_D3D12_Shutdown1(ID3D12Device* d) { Drop(false, d); return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_D3D12_GetScratchBufferSize(int, const NgxParameter*, size_t* size) { if (size) *size = 0; return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList* list, int feature, NgxParameter*, NgxHandle** out)
{
    if (feature != kFeatureFrameGeneration) return kNgxNotSupported;
    if (!list || !out) return kNgxInvalidParameter;
    ID3D12Device* d = nullptr;
    if (FAILED(list->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&d)))) return kNgxPlatformError;
    std::lock_guard<std::mutex> lock(g_mutex);
    const NgxResult r = Add(d, false, out);
    d->Release();
    return r;
}
NFRUX_EXPORT NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList* list, const NgxHandle* h, NgxParameter* p, void*) { return Eval(list, h, p); }
NFRUX_EXPORT NVSDK_NGX_D3D12_ReleaseFeature(NgxHandle* h) { return Release(h); }
NFRUX_EXPORT NVSDK_NGX_D3D12_GetFeatureRequirements(void*, const void*, NgxFeatureRequirement* r) { return Requirements(r); }
NFRUX_EXPORT NVSDK_NGX_D3D12_PopulateParameters_Impl(NgxParameter* p) { return Populate(p); }
NFRUX_EXPORT NVSDK_NGX_D3D12_PopulateDeviceParameters_Impl(ID3D12Device*, NgxParameter* p) { return Populate(p); }
// --- Vulkan
NFRUX_EXPORT NVSDK_NGX_VULKAN_Init(unsigned long long, const wchar_t*, VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd, const void*, int) { SetVk(i, pd, d, gi, gd); return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_VULKAN_Init_Ext(unsigned long long, const wchar_t*, VkInstance i, VkPhysicalDevice pd, VkDevice d, int, const void*) { SetVk(i, pd, d, nullptr, nullptr); return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_VULKAN_Init_Ext2(unsigned long long, const wchar_t*, VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd, int, const void*) { SetVk(i, pd, d, gi, gd); return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_VULKAN_Shutdown() { Drop(true, nullptr); return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_VULKAN_Shutdown1(VkDevice d) { Drop(true, d); return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_VULKAN_GetScratchBufferSize(int, const NgxParameter*, size_t* size) { if (size) *size = 0; return kNgxSuccess; }
NFRUX_EXPORT NVSDK_NGX_VULKAN_CreateFeature1(VkDevice d, VkCommandBuffer, int feature, NgxParameter*, NgxHandle** out)
{
    if (feature != kFeatureFrameGeneration) return kNgxNotSupported;
    if (!out) return kNgxInvalidParameter;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (d) g_vk.d = d;
    return Add(g_vk.d, true, out);
}
NFRUX_EXPORT NVSDK_NGX_VULKAN_CreateFeature(VkCommandBuffer c, int feature, NgxParameter* p, NgxHandle** out) { return NVSDK_NGX_VULKAN_CreateFeature1(VK_NULL_HANDLE, c, feature, p, out); }
NFRUX_EXPORT NVSDK_NGX_VULKAN_EvaluateFeature(VkCommandBuffer c, const NgxHandle* h, NgxParameter* p, void*) { return Eval(c, h, p); }
NFRUX_EXPORT NVSDK_NGX_VULKAN_ReleaseFeature(NgxHandle* h) { return Release(h); }
NFRUX_EXPORT NVSDK_NGX_VULKAN_GetFeatureRequirements(VkInstance, VkPhysicalDevice, const void*, NgxFeatureRequirement* r) { return Requirements(r); }
NFRUX_EXPORT NVSDK_NGX_VULKAN_PopulateParameters_Impl(NgxParameter* p) { return Populate(p); }
NFRUX_EXPORT NVSDK_NGX_VULKAN_PopulateDeviceParameters_Impl(VkInstance, VkPhysicalDevice, VkDevice, NgxParameter* p) { return Populate(p); }
// --- both
NFRUX_EXPORT NVSDK_NGX_GetAPIVersion() { return 0x15; }
NFRUX_EXPORT NVSDK_NGX_GetSnippetVersion() { return 0x00040000; }
// The loader's record of what the game's Vulkan device has enabled.
extern "C" __declspec(dllexport) void __cdecl NFRUX_SetVulkanFeatures(VkDevice, uint32_t mlFeatures, int atomic64) { g_vkMl = mlFeatures, g_vkAtomic64 = atomic64 != 0, g_vkKnown = true; }
extern "C" __declspec(dllexport) uint32_t __cdecl NFRUX_MaxFrames() { return MaxFrames(); }

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
