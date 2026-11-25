// The loader role of NFRUX's one DLL, when its file is named as a Windows DLL the game loads (dbghelp.dll, version.dll): Streamline's calls to the NGX core for feature 11 (frame generation) go to NFRUX, the
// rest to the core untouched. Nothing outside this process changes. SPDX-License-Identifier: MIT
#include "ngx.h"
#include "proxy_exports.h"
#include <windows.h>
#include <winternl.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <map>
#include <memory>
#include <vector>
#include <mutex>
#include <set>
#include <string>

static FILE* g_log;
static std::mutex g_m;
static void L(const char* f, ...) { static const bool on = getenv("NFRUX_LOG") && strcmp(getenv("NFRUX_LOG"), "0"); if (!on) return; std::lock_guard<std::mutex> k(g_m); if (!g_log && !(g_log = fopen("nfrux_loader.log", "w"))) return; va_list a; va_start(a, f); vfprintf(g_log, f, a); va_end(a); fputc('\n', g_log); fflush(g_log); }

static std::map<std::string, void*> g_real;
static std::set<void*> g_handles;
static bool g_usedNfrux;

template <class F> static F Real(const char* n) { return reinterpret_cast<F>(g_real[n]); }

// NFRUX is this DLL (NFRUX_DLL: another one, for tests).
static HMODULE Nfrux()
{
    static HMODULE m = [] {
        HMODULE self = nullptr;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)&L, &self);
        if (const char* e = getenv("NFRUX_DLL")) self = LoadLibraryExA(e, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        g_usedNfrux = self != nullptr;
        return self;
    }();
    return m;
}template <class F> static F Ours(const char* n) { return Nfrux() ? reinterpret_cast<F>(GetProcAddress(Nfrux(), n)) : nullptr; }
static std::mutex g_hm;
static bool IsOurs(const NgxHandle* h) { std::lock_guard<std::mutex> k(g_hm); return g_handles.count((void*)h) != 0; }
template <class T> static NgxResult Keep(NgxResult r, T** h) { if (r == kNgxSuccess) { std::lock_guard<std::mutex> k(g_hm); g_handles.insert(*h); } L("feature 11: %x", r); return r; }

// --- D3D12
namespace nfrux { void LogD3D12Messages(ID3D12Device* device); }
static std::atomic<bool> g_feature12;
static NgxResult __cdecl Create12(ID3D12GraphicsCommandList* c, int f, NgxParameter* p, NgxHandle** h)
{
    if (f == kFeatureFrameGeneration && getenv("NFRUX_TEST_PASSTHROUGH"))  // the control: NVIDIA's own feature, the debug layer's messages logged
    {
        ID3D12Device* d = nullptr;
        c->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&d));
        nfrux::LogD3D12Messages(d);
        d->Release();
        return Real<decltype(&Create12)>("NVSDK_NGX_D3D12_CreateFeature")(c, f, p, h);
    }
    if (auto o = f == kFeatureFrameGeneration ? Ours<decltype(&Create12)>("NVSDK_NGX_D3D12_CreateFeature") : nullptr) { g_feature12 = true; return Keep(o(c, f, p, h), h); }  // (see Tags)
    return Real<decltype(&Create12)>("NVSDK_NGX_D3D12_CreateFeature")(c, f, p, h);
}
static NgxResult __cdecl Eval12(ID3D12GraphicsCommandList* c, const NgxHandle* h, NgxParameter* p, void* cb)
{
    static auto o = Ours<decltype(&Eval12)>("NVSDK_NGX_D3D12_EvaluateFeature");
    return IsOurs(h) ? o(c, h, p, cb) : Real<decltype(&Eval12)>("NVSDK_NGX_D3D12_EvaluateFeature")(c, h, p, cb);
}
static NgxResult __cdecl Eval12C(ID3D12GraphicsCommandList* c, const NgxHandle* h, NgxParameter* p, void* cb)
{
    static auto o = Ours<decltype(&Eval12C)>("NVSDK_NGX_D3D12_EvaluateFeature");
    return IsOurs(h) ? o(c, h, p, cb) : Real<decltype(&Eval12C)>("NVSDK_NGX_D3D12_EvaluateFeature_C")(c, h, p, cb);
}
static NgxResult __cdecl Release12(NgxHandle* h)
{
    if (IsOurs(h)) { { std::lock_guard<std::mutex> k(g_hm); g_handles.erase(h); } return Ours<decltype(&Release12)>("NVSDK_NGX_D3D12_ReleaseFeature")(h); }
    return Real<decltype(&Release12)>("NVSDK_NGX_D3D12_ReleaseFeature")(h);
}
static NgxResult __cdecl Req12(void* a, const NgxFeatureDiscoveryInfo* i, NgxFeatureRequirement* r)
{
    if (auto o = i && i->FeatureID == kFeatureFrameGeneration ? Ours<decltype(&Req12)>("NVSDK_NGX_D3D12_GetFeatureRequirements") : nullptr) return o(a, i, r);
    return Real<decltype(&Req12)>("NVSDK_NGX_D3D12_GetFeatureRequirements")(a, i, r);
}
// --- Vulkan
using PFN_InitVk2 = NgxResult(__cdecl*)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, int, const void*);
static uint32_t g_ml_(); static VkDevice g_dev_();
static void NoteVk(VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd) { if (auto f = Ours<void(__cdecl*)(VkDevice, uint32_t)>("NFRUX_SetVulkanFeatures"); f && g_dev_()) f(d, g_ml_()); if (auto o = Ours<PFN_InitVk2>("NVSDK_NGX_VULKAN_Init_Ext2")) o(0, nullptr, i, pd, d, gi, gd, 0x15, nullptr); }
static NgxResult __cdecl InitVk(unsigned long long a, const wchar_t* b, VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd, const void* fi, int v) { NoteVk(i, pd, d, gi, gd); return Real<decltype(&InitVk)>("NVSDK_NGX_VULKAN_Init")(a, b, i, pd, d, gi, gd, fi, v); }
static NgxResult __cdecl InitVk2(unsigned long long a, const wchar_t* b, VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd, int v, const void* fi) { NoteVk(i, pd, d, gi, gd); return Real<decltype(&InitVk2)>("NVSDK_NGX_VULKAN_Init_Ext2")(a, b, i, pd, d, gi, gd, v, fi); }
static NgxResult __cdecl InitVkP(const char* pid, int e, const char* ev, const wchar_t* b, VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd, int v, const void* fi) { NoteVk(i, pd, d, gi, gd); return Real<decltype(&InitVkP)>("NVSDK_NGX_VULKAN_Init_ProjectID_Ext")(pid, e, ev, b, i, pd, d, gi, gd, v, fi); }
static NgxResult __cdecl CreateVk(VkCommandBuffer c, int f, NgxParameter* p, NgxHandle** h)
{
    if (auto o = f == kFeatureFrameGeneration ? Ours<decltype(&CreateVk)>("NVSDK_NGX_VULKAN_CreateFeature") : nullptr) return Keep(o(c, f, p, h), h);
    return Real<decltype(&CreateVk)>("NVSDK_NGX_VULKAN_CreateFeature")(c, f, p, h);
}
static NgxResult __cdecl CreateVk1(VkDevice d, VkCommandBuffer c, int f, NgxParameter* p, NgxHandle** h)
{
    if (auto o = f == kFeatureFrameGeneration ? Ours<decltype(&CreateVk1)>("NVSDK_NGX_VULKAN_CreateFeature1") : nullptr) return Keep(o(d, c, f, p, h), h);
    return Real<decltype(&CreateVk1)>("NVSDK_NGX_VULKAN_CreateFeature1")(d, c, f, p, h);
}
static NgxResult __cdecl EvalVk(VkCommandBuffer c, const NgxHandle* h, NgxParameter* p, void* cb)
{
    static auto o = Ours<decltype(&EvalVk)>("NVSDK_NGX_VULKAN_EvaluateFeature");
    return IsOurs(h) ? o(c, h, p, cb) : Real<decltype(&EvalVk)>("NVSDK_NGX_VULKAN_EvaluateFeature")(c, h, p, cb);
}
static NgxResult __cdecl EvalVkC(VkCommandBuffer c, const NgxHandle* h, NgxParameter* p, void* cb)
{
    static auto o = Ours<decltype(&EvalVkC)>("NVSDK_NGX_VULKAN_EvaluateFeature");
    return IsOurs(h) ? o(c, h, p, cb) : Real<decltype(&EvalVkC)>("NVSDK_NGX_VULKAN_EvaluateFeature_C")(c, h, p, cb);
}
static NgxResult __cdecl ReqVk(VkInstance i, VkPhysicalDevice pd, const NgxFeatureDiscoveryInfo* d, NgxFeatureRequirement* r)
{
    if (auto o = d && d->FeatureID == kFeatureFrameGeneration ? Ours<decltype(&ReqVk)>("NVSDK_NGX_VULKAN_GetFeatureRequirements") : nullptr) return o(i, pd, d, r);
    return Real<decltype(&ReqVk)>("NVSDK_NGX_VULKAN_GetFeatureRequirements")(i, pd, d, r);
}
static NgxResult __cdecl InitVkE(unsigned long long a, const wchar_t* b, VkInstance i, VkPhysicalDevice pd, VkDevice d, int v, const void* fi) { NoteVk(i, pd, d, nullptr, nullptr); return Real<decltype(&InitVkE)>("NVSDK_NGX_VULKAN_Init_Ext")(a, b, i, pd, d, v, fi); }
static NgxResult __cdecl ReleaseVk(NgxHandle* h)
{
    if (IsOurs(h)) { { std::lock_guard<std::mutex> k(g_hm); g_handles.erase(h); } return Ours<decltype(&ReleaseVk)>("NVSDK_NGX_VULKAN_ReleaseFeature")(h); }
    return Real<decltype(&ReleaseVk)>("NVSDK_NGX_VULKAN_ReleaseFeature")(h);
}
// --- the end: NFRUX's features first, then the core's
static NgxResult __cdecl Shut12() { L("D3D12 Shutdown"); if (auto o = Ours<decltype(&Shut12)>("NVSDK_NGX_D3D12_Shutdown")) o(); return Real<decltype(&Shut12)>("NVSDK_NGX_D3D12_Shutdown")(); }
static NgxResult __cdecl Shut12D(ID3D12Device* d) { L("D3D12 Shutdown1"); if (auto o = Ours<decltype(&Shut12D)>("NVSDK_NGX_D3D12_Shutdown1")) o(d); return Real<decltype(&Shut12D)>("NVSDK_NGX_D3D12_Shutdown1")(d); }
static NgxResult __cdecl ShutVk() { L("Vulkan Shutdown"); if (auto o = Ours<decltype(&ShutVk)>("NVSDK_NGX_VULKAN_Shutdown")) o(); return Real<decltype(&ShutVk)>("NVSDK_NGX_VULKAN_Shutdown")(); }
static NgxResult __cdecl ShutVkD(VkDevice d) { L("Vulkan Shutdown1"); if (auto o = Ours<decltype(&ShutVkD)>("NVSDK_NGX_VULKAN_Shutdown1")) o(d); return Real<decltype(&ShutVkD)>("NVSDK_NGX_VULKAN_Shutdown1")(d); }
// --- vkCreateDevice: what the game enables, plus what NFRUX's fast paths want (DESIGN D3).
#include "../core/ml/vulkan_network.h"
static PFN_vkCreateDevice g_createDevice;
static uint32_t           g_ml;
static VkDevice           g_featureDevice;
static PFN_vkGetInstanceProcAddr Gipa() { return (PFN_vkGetInstanceProcAddr)GetProcAddress(GetModuleHandleA("vulkan-1.dll"), "vkGetInstanceProcAddr"); }

static VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo* ci, const VkAllocationCallbacks* a, VkDevice* out)
{
    auto getF = (PFN_vkGetPhysicalDeviceFeatures2)GetProcAddress(GetModuleHandleA("vulkan-1.dll"), "vkGetPhysicalDeviceFeatures2");
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmS = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features s13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &cmS};
    VkPhysicalDeviceVulkan11Features s11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &s13};
    VkPhysicalDeviceVulkan12Features s12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &s11};
    VkPhysicalDeviceFeatures2 s2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &s12};
    if (getF) getF(pd, &s2);
    // The chain as it is, with copies of the structures changed: found by type, relinked in order.
    struct Copy { alignas(16) unsigned char b[512]; };
    std::vector<std::unique_ptr<Copy>> copies;
    VkDeviceCreateInfo c = *ci;
    VkPhysicalDeviceFeatures core = {};
    VkPhysicalDeviceFeatures2* f2 = nullptr; VkPhysicalDeviceVulkan12Features* f12 = nullptr; VkPhysicalDeviceVulkan13Features* f13 = nullptr;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR* fcm = nullptr; VkPhysicalDeviceVulkanMemoryModelFeatures* fmm = nullptr; VkPhysicalDeviceShaderFloat16Int8Features* f8 = nullptr;
    VkPhysicalDevice8BitStorageFeatures* f8s = nullptr; VkPhysicalDeviceShaderIntegerDotProductFeatures* fdot = nullptr;
    VkPhysicalDeviceVulkan11Features* f11 = nullptr; VkPhysicalDevice16BitStorageFeatures* f16s = nullptr; VkPhysicalDeviceSubgroupSizeControlFeatures* fsg = nullptr;
    auto size = [](VkStructureType t) -> size_t {
        switch (t)
        {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: return sizeof(VkPhysicalDeviceFeatures2);
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: return sizeof(VkPhysicalDeviceVulkan12Features);
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: return sizeof(VkPhysicalDeviceVulkan13Features);
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR: return sizeof(VkPhysicalDeviceCooperativeMatrixFeaturesKHR);
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES: return sizeof(VkPhysicalDeviceVulkanMemoryModelFeatures);
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: return sizeof(VkPhysicalDeviceShaderFloat16Int8Features);
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: return sizeof(VkPhysicalDevice8BitStorageFeatures);
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES: return sizeof(VkPhysicalDeviceShaderIntegerDotProductFeatures);
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: return sizeof(VkPhysicalDeviceVulkan11Features);
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: return sizeof(VkPhysicalDevice16BitStorageFeatures);
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES: return sizeof(VkPhysicalDeviceSubgroupSizeControlFeatures);
        default: return 0;
        }
    };
    // Copying stops at the first structure it does not know the size of after the ones to change: everything up to the last
    // structure to change is needed, so unknown ones before it keep their place by patching their (the game's) pNext back after.
    struct Patch { VkBaseOutStructure* at; VkBaseOutStructure* old; };
    std::vector<Patch> patches;
    VkBaseOutStructure head = {}; head.pNext = (VkBaseOutStructure*)ci->pNext;
    VkBaseOutStructure* prev = &head;
    for (auto* s = (VkBaseOutStructure*)ci->pNext; s;)
    {
        VkBaseOutStructure* next = s->pNext;
        VkBaseOutStructure* use  = s;
        if (const size_t n = size(s->sType))
        {
            copies.push_back(std::make_unique<Copy>());
            memcpy(copies.back()->b, s, n);
            use = (VkBaseOutStructure*)copies.back()->b;
            switch (s->sType)
            {
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: f2 = (VkPhysicalDeviceFeatures2*)use; break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: f12 = (VkPhysicalDeviceVulkan12Features*)use; break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: f13 = (VkPhysicalDeviceVulkan13Features*)use; break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR: fcm = (VkPhysicalDeviceCooperativeMatrixFeaturesKHR*)use; break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES: fmm = (VkPhysicalDeviceVulkanMemoryModelFeatures*)use; break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: f8 = (VkPhysicalDeviceShaderFloat16Int8Features*)use; break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: f8s = (VkPhysicalDevice8BitStorageFeatures*)use; break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES: fdot = (VkPhysicalDeviceShaderIntegerDotProductFeatures*)use; break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: f11 = (VkPhysicalDeviceVulkan11Features*)use; break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: f16s = (VkPhysicalDevice16BitStorageFeatures*)use; break;
            default: fsg = (VkPhysicalDeviceSubgroupSizeControlFeatures*)use; break;
            }
        }
        if (prev != &head && prev->pNext != use) { if (!size(prev->sType)) patches.push_back({prev, prev->pNext}); prev->pNext = use; }
        else if (prev == &head) head.pNext = use;
        prev = use, s = next;
    }
    c.pNext = head.pNext;
    std::vector<const char*> ext(ci->ppEnabledExtensionNames, ci->ppEnabledExtensionNames + ci->enabledExtensionCount);
    auto has = [&](const char* n) { for (auto* e : ext) if (!strcmp(e, n)) return true; return false; };
    auto chain = [&](auto* s) { s->pNext = (void*)c.pNext; c.pNext = s; };
    // Core features.
    auto wantCore = [&](VkPhysicalDeviceFeatures& f) {
        f.shaderStorageImageExtendedFormats |= s2.features.shaderStorageImageExtendedFormats;
    };
    if (f2) wantCore(f2->features);
    else { if (ci->pEnabledFeatures) core = *ci->pEnabledFeatures; wantCore(core); c.pEnabledFeatures = &core; }
    VkPhysicalDeviceVulkanMemoryModelFeatures mm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
    VkPhysicalDeviceShaderFloat16Int8Features i8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice8BitStorageFeatures st8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeatures sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    VkPhysicalDevice16BitStorageFeatures st16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    bool float16 = false;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    if (f12)
    {
        f12->vulkanMemoryModel |= s12.vulkanMemoryModel, f12->vulkanMemoryModelDeviceScope |= s12.vulkanMemoryModelDeviceScope;
        f12->shaderInt8 |= s12.shaderInt8, f12->storageBuffer8BitAccess |= s12.storageBuffer8BitAccess;
    }
    else
    {
        if (!fmm) chain(fmm = &mm); if (!f8) chain(f8 = &i8); if (!f8s) chain(f8s = &st8);
        fmm->vulkanMemoryModel |= s12.vulkanMemoryModel, fmm->vulkanMemoryModelDeviceScope |= s12.vulkanMemoryModelDeviceScope;
        f8->shaderInt8 |= s12.shaderInt8, f8s->storageBuffer8BitAccess |= s12.storageBuffer8BitAccess;
    }
    // Promoted in 1.3: without its aggregate, only when the extension is there to enable.
    const bool own13 = !f13;
    auto device = [&](const char* n) {
        static auto en = (PFN_vkEnumerateDeviceExtensionProperties)GetProcAddress(GetModuleHandleA("vulkan-1.dll"), "vkEnumerateDeviceExtensionProperties");
        uint32_t count = 0;
        if (!en || en(pd, nullptr, &count, nullptr)) return false;
        std::vector<VkExtensionProperties> e(count);
        en(pd, nullptr, &count, e.data());
        for (auto& x : e) if (!strcmp(x.extensionName, n)) return true;
        return false;
    };
    if (own13)
        for (auto [n, b] : {std::pair<const char*, int>{VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, 0}, std::pair<const char*, int>{VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME, 1}})
        {
            if (device(n)) { if (!has(n)) ext.push_back(n); }
            else if (b) s13.subgroupSizeControl = s13.computeFullSubgroups = VK_FALSE;
            else s13.shaderIntegerDotProduct = VK_FALSE;
        }
    if (f13) f13->shaderIntegerDotProduct |= s13.shaderIntegerDotProduct, f13->subgroupSizeControl |= s13.subgroupSizeControl, f13->computeFullSubgroups |= s13.computeFullSubgroups;
    else
    {
        if (!fdot) { fdot = &dot; if (s13.shaderIntegerDotProduct) chain(fdot); }
        if (!fsg) { fsg = &sg; if (s13.subgroupSizeControl) chain(fsg); }
        fdot->shaderIntegerDotProduct |= s13.shaderIntegerDotProduct, fsg->subgroupSizeControl |= s13.subgroupSizeControl, fsg->computeFullSubgroups |= s13.computeFullSubgroups;
    }
    if (cmS.cooperativeMatrix)
    {
        if (!has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) ext.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        if (!fcm) chain(fcm = &cm);
        fcm->cooperativeMatrix = VK_TRUE;
        // fp16 matrices: the fastest on NVIDIA, the only ones on some.
        if (f12) f12->shaderFloat16 |= s12.shaderFloat16; else f8->shaderFloat16 |= s12.shaderFloat16;
        float16 = s12.shaderFloat16 && s11.storageBuffer16BitAccess;
        if (float16 && f11) f11->storageBuffer16BitAccess = VK_TRUE;
        else if (float16) { if (!f16s) chain(f16s = &st16); f16s->storageBuffer16BitAccess = VK_TRUE; }
    }
    c.enabledExtensionCount = uint32_t(ext.size()), c.ppEnabledExtensionNames = ext.data();
    VkResult r = g_createDevice(pd, &c, a, out);
    for (size_t i = patches.size(); i--;) patches[i].at->pNext = patches[i].old;
    const bool added = r == VK_SUCCESS;
    if (!added) r = g_createDevice(pd, ci, a, out);
    uint32_t ml = 0;
    if (added)
    {
        using namespace ffx_ml;
        if (s12.shaderInt8) ml |= FEATURE_SHADER_INT8;
        if (s12.storageBuffer8BitAccess) ml |= FEATURE_STORAGE_BUFFER_8BIT;
        if (s13.shaderIntegerDotProduct) ml |= FEATURE_INTEGER_DOT_PRODUCT;
        if (s12.vulkanMemoryModel) ml |= FEATURE_VULKAN_MEMORY_MODEL;
        if (s13.subgroupSizeControl && s13.computeFullSubgroups) ml |= FEATURE_SUBGROUP_SIZE_CONTROL;
        if (cmS.cooperativeMatrix) ml |= FEATURE_COOPERATIVE_MATRIX;
        if (float16) ml |= FEATURE_SHADER_FLOAT16 | FEATURE_STORAGE_BUFFER_16BIT;
    }
    g_ml = ml;
    if (r == VK_SUCCESS) g_featureDevice = *out;
    L("vkCreateDevice -> %d, features %s, ml 0x%x", r, added ? "added" : "refused: the game's own call", ml);
    return r;
}
static uint32_t g_ml_() { return g_ml; } static VkDevice g_dev_() { return g_featureDevice; }
static PFN_vkDestroyDevice g_destroyDevice;
static void VKAPI_CALL DestroyDevice(VkDevice d, const VkAllocationCallbacks* a)
{
    L("vkDestroyDevice %p", d);
    if (g_usedNfrux) if (auto f = Ours<void(__cdecl*)(VkDevice)>("NFRUX_DropVulkanDevice")) f(d);
    if (g_featureDevice == d) g_featureDevice = VK_NULL_HANDLE;
    g_destroyDevice(d, a);  // vulkan-1's, which goes to this device's driver
}
static void Trampoline() { if (!g_destroyDevice) g_destroyDevice = (PFN_vkDestroyDevice)GetProcAddress(GetModuleHandleA("vulkan-1.dll"), "vkDestroyDevice"); }
static PFN_vkGetDeviceProcAddr g_gdpa;
static PFN_vkVoidFunction VKAPI_CALL HookGdpa(VkDevice d, const char* n)
{
    PFN_vkVoidFunction f = g_gdpa(d, n);
    if (f && n && !strcmp(n, "vkDestroyDevice")) { Trampoline(); return (PFN_vkVoidFunction)DestroyDevice; }
    return f;
}
static PFN_vkGetInstanceProcAddr g_gipa;
static PFN_vkVoidFunction VKAPI_CALL HookGipa(VkInstance i, const char* n)
{
    PFN_vkVoidFunction f = g_gipa(i, n);
    if (f && n && !strcmp(n, "vkCreateDevice")) { if (!g_createDevice) g_createDevice = (PFN_vkCreateDevice)f; return (PFN_vkVoidFunction)CreateDevice; }
    if (f && n && !strcmp(n, "vkDestroyDevice")) { Trampoline(); return (PFN_vkVoidFunction)DestroyDevice; }
    if (f && n && !strcmp(n, "vkGetDeviceProcAddr")) { if (!g_gdpa) g_gdpa = (PFN_vkGetDeviceProcAddr)f; return (PFN_vkVoidFunction)HookGdpa; }
    return f;
}

static void Caps(NgxParameter* p) { if (!p) return; static auto mf = Ours<uint32_t(__cdecl*)()>("NFRUX_MaxFrames"); if (!mf) return; p->Set("DLSSG.MultiFrameCountMax", mf()); p->Set("FrameGeneration.Available", 1); p->Set("FrameGeneration.NeedsUpdatedDriver", 0); p->Set("FrameGeneration.FeatureInitResult", 1); if (auto e = Ours<NgxResult(__cdecl*)(NgxParameter*)>("NVSDK_NGX_D3D12_PopulateParameters_Impl")) e(p); }
template <int N> static const char* kCapName;
template <> const char* kCapName<0> = "NVSDK_NGX_D3D12_GetCapabilityParameters";
template <> const char* kCapName<1> = "NVSDK_NGX_VULKAN_GetCapabilityParameters";
template <> const char* kCapName<2> = "NVSDK_NGX_D3D12_GetParameters";
template <> const char* kCapName<3> = "NVSDK_NGX_VULKAN_GetParameters";
template <int N> static NgxResult __cdecl Cap(NgxParameter** p) { const NgxResult r = Real<NgxResult(__cdecl*)(NgxParameter**)>(kCapName<N>)(p); L("%s -> %x", kCapName<N>, r); if (r == kNgxSuccess) Caps(*p); return r; }
static const std::map<std::string, void*> kWrap = {
    {"NVSDK_NGX_D3D12_GetCapabilityParameters", (void*)Cap<0>}, {"NVSDK_NGX_VULKAN_GetCapabilityParameters", (void*)Cap<1>}, {"NVSDK_NGX_D3D12_GetParameters", (void*)Cap<2>}, {"NVSDK_NGX_VULKAN_GetParameters", (void*)Cap<3>},
    {"NVSDK_NGX_D3D12_CreateFeature", (void*)Create12}, {"NVSDK_NGX_D3D12_EvaluateFeature", (void*)Eval12}, {"NVSDK_NGX_D3D12_ReleaseFeature", (void*)Release12}, {"NVSDK_NGX_D3D12_GetFeatureRequirements", (void*)Req12},
    {"NVSDK_NGX_VULKAN_Init", (void*)InitVk}, {"NVSDK_NGX_VULKAN_Init_Ext2", (void*)InitVk2}, {"NVSDK_NGX_VULKAN_Init_ProjectID_Ext", (void*)InitVkP}, {"NVSDK_NGX_VULKAN_CreateFeature", (void*)CreateVk}, {"NVSDK_NGX_VULKAN_CreateFeature1", (void*)CreateVk1}, {"NVSDK_NGX_VULKAN_EvaluateFeature", (void*)EvalVk}, {"NVSDK_NGX_VULKAN_ReleaseFeature", (void*)ReleaseVk},
    {"NVSDK_NGX_D3D12_EvaluateFeature_C", (void*)Eval12C}, {"NVSDK_NGX_VULKAN_EvaluateFeature_C", (void*)EvalVkC}, {"NVSDK_NGX_VULKAN_GetFeatureRequirements", (void*)ReqVk}, {"NVSDK_NGX_VULKAN_Init_Ext", (void*)InitVkE},
    {"NVSDK_NGX_D3D12_Shutdown", (void*)Shut12}, {"NVSDK_NGX_D3D12_Shutdown1", (void*)Shut12D}, {"NVSDK_NGX_VULKAN_Shutdown", (void*)ShutVk}, {"NVSDK_NGX_VULKAN_Shutdown1", (void*)ShutVkD}};

// --- Streamline's tags (D3D12). DLSS-G's inputs reach the feature in the state they were tagged with ("its state when it is used"): Streamline does
// not transition them. The state is handed to the feature. Where a COMPUTE list cannot use it (RENDER_TARGET, DEPTH_WRITE ...), the tag is passed on as
// valid only now: Streamline copies it on the game's list, taking the declared state for the one it has there, and the copy's state is known. Not done for
// any other state, nor before NFRUX's feature exists. (Streamline 2.14 sl_core_types.h: BaseStructure is next, 16-byte type, version.)
struct SlResource { void* next; uint32_t type[4]; size_t version; char kind; void *native, *memory, *view; uint32_t state; };
struct SlTag { void* next; uint32_t type[4]; size_t version; SlResource* resource; uint32_t buffer; int lifecycle; uint32_t extent[4]; };
static_assert(sizeof(SlTag) == 64);
static std::vector<SlTag> Tags(const SlTag* t, uint32_t n, void* list)
{
    std::vector<SlTag> o;
    if (reinterpret_cast<uintptr_t>(t) < 65536 || !n || n > 64 || !g_feature12 || g_featureDevice) return o;  // (Streamline 1's slSetTag has other arguments)
    for (uint32_t i = 0; i < n; ++i) if (t[i].type[0] != 0x4c6a5aad) return o;
    static auto note = Ours<void(__cdecl*)(void*, uint32_t)>("NFRUX_NoteState");
    bool changed = false;
    o.assign(t, t + n);
    for (SlTag& x : o)
        for (uint32_t b : {0u, 1u, 2u, 23u, 50u, 54u, 69u})  // depth, vectors, HUD-less, UI, distortion, no-warp, UI alpha
            if (x.buffer == b && x.lifecycle == 1 && x.resource && x.resource->native && x.resource->state != UINT32_MAX)
            {
                const uint32_t s = x.resource->state, usable = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_COPY_DEST | D3D12_RESOURCE_STATE_COPY_SOURCE;
                const bool readable = (s & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) != 0, copied = b == 2 || b == 23 || b == 69;  // colours are copied from as well as read
                if ((s & ~usable) == 0 || (readable && !copied)) { if (note) note(x.resource->native, s); continue; }
                static uint32_t said = 0;
                if (!(said >> (b % 32) & 1)) said |= 1u << (b % 32), L("tag %u in state 0x%x: %s", b, s, list ? "copied when tagged" : "no command list to copy it on: used as it is");
                if (note) note(x.resource->native, list ? uint32_t(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) : s);  // copied: no older state of it stays
                if (list) x.lifecycle = 0, changed = true;
            }
    if (!changed) o.clear();
    return o;
}using SlSetTag = int (*)(const void*, const SlTag*, uint32_t, void*);
using SlSetTagForFrame = int (*)(const void*, const void*, const SlTag*, uint32_t, void*);
static SlSetTag g_setTag;
static SlSetTagForFrame g_setTagForFrame;
static int SetTag(const void* v, const SlTag* t, uint32_t n, void* list) { const auto o = Tags(t, n, list); return g_setTag(v, o.empty() ? t : o.data(), n, list); }
static int SetTagForFrame(const void* f, const void* v, const SlTag* t, uint32_t n, void* list) { const auto o = Tags(t, n, list); return g_setTagForFrame(f, v, o.empty() ? t : o.data(), n, list); }

static FARPROC WINAPI HookGPA(HMODULE m, LPCSTR n)
{
    FARPROC r = GetProcAddress(m, n);
    if (HIWORD(n) && !strncmp(n, "NVSDK_NGX_", 10))
    {
        char path[MAX_PATH] = {}; GetModuleFileNameA(m, path, MAX_PATH);
        const bool core = strstr(path, "nvngx.dll") != nullptr;
        L("GetProcAddress %s %s -> %p", path, n, r);
        auto it = kWrap.find(n);
        if (core && r && it != kWrap.end()) { g_real[n] = (void*)r; return (FARPROC)it->second; }
    }
    if (HIWORD(n) && r && !strncmp(n, "slSetTag", 8) && m == GetModuleHandleA("sl.interposer.dll"))
    {
        if (!strcmp(n, "slSetTag")) { g_setTag = (SlSetTag)r; return (FARPROC)SetTag; }
        if (!strcmp(n, "slSetTagForFrame")) { g_setTagForFrame = (SlSetTagForFrame)r; return (FARPROC)SetTagForFrame; }
    }
    if (HIWORD(n) && r && m == GetModuleHandleA("vulkan-1.dll"))
    {
        if (!strcmp(n, "vkCreateDevice")) { g_createDevice = (PFN_vkCreateDevice)r; return (FARPROC)CreateDevice; }
        if (!strcmp(n, "vkDestroyDevice")) { g_destroyDevice = (PFN_vkDestroyDevice)r; return (FARPROC)DestroyDevice; }
        if (!strcmp(n, "vkGetDeviceProcAddr")) { g_gdpa = (PFN_vkGetDeviceProcAddr)r; return (FARPROC)HookGdpa; }
        if (!strcmp(n, "vkGetInstanceProcAddr")) { g_gipa = (PFN_vkGetInstanceProcAddr)r; return (FARPROC)HookGipa; }
    }
    return r;
}
static HMODULE WINAPI HookLLX(LPCWSTR n, HANDLE f, DWORD fl) { HMODULE m = LoadLibraryExW(n, f, fl); if (n && wcsstr(n, L"nvngx")) L("LoadLibraryExW %ls -> %p", n, m); return m; }
static void Patch(HMODULE m)
{
    auto* base = (uint8_t*)m; auto* dos = (IMAGE_DOS_HEADER*)base; if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    auto* nt = (IMAGE_NT_HEADERS*)(base + dos->e_lfanew); auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT]; if (!dir.VirtualAddress) return;
    for (auto* imp = (IMAGE_IMPORT_DESCRIPTOR*)(base + dir.VirtualAddress); imp->Name; ++imp)
    {
        if (!imp->OriginalFirstThunk) continue;
        auto* o = (IMAGE_THUNK_DATA*)(base + imp->OriginalFirstThunk); auto* t = (IMAGE_THUNK_DATA*)(base + imp->FirstThunk);
        for (; o->u1.AddressOfData; ++o, ++t)
        {
            if (IMAGE_SNAP_BY_ORDINAL(o->u1.Ordinal)) continue;
            const char* n = ((IMAGE_IMPORT_BY_NAME*)(base + o->u1.AddressOfData))->Name; void* h = nullptr;
            if (!strcmp(n, "GetProcAddress")) h = (void*)HookGPA; else if (!strcmp(n, "LoadLibraryExW")) h = (void*)HookLLX;
            else if (!strncmp(n, "slSetTag", 8))
            {
                if (auto* sl = GetModuleHandleA("sl.interposer.dll"))
                    if (void* real = (void*)GetProcAddress(sl, n))
                    {
                        if (!strcmp(n, "slSetTag")) g_setTag = (SlSetTag)real, h = (void*)SetTag;
                        else if (!strcmp(n, "slSetTagForFrame")) g_setTagForFrame = (SlSetTagForFrame)real, h = (void*)SetTagForFrame;
                    }
            }
            if (h) { DWORD old; if (VirtualProtect(&t->u1.Function, 8, PAGE_READWRITE, &old)) { t->u1.Function = (ULONGLONG)h; VirtualProtect(&t->u1.Function, 8, old, &old); } }
        }
    }
}
struct NOTIFY { ULONG Flags; const UNICODE_STRING *Full, *Base; PVOID DllBase; ULONG Size; };
static wchar_t g_win[MAX_PATH]; static size_t g_winLen;
static bool IsWindows(const UNICODE_STRING* n) { return g_winLen && n->Buffer && n->Length / 2 > g_winLen && !_wcsnicmp(n->Buffer, g_win, g_winLen); }
static VOID CALLBACK OnDll(ULONG reason, const NOTIFY* d, PVOID) { if (reason == 1 && !IsWindows(d->Full)) { Patch((HMODULE)d->DllBase); } }
// From DllMain.
void LoaderAttach(HINSTANCE self)
{
    {
        HMODULE pin = nullptr;  // patched imports point here: never unloaded
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN, (LPCSTR)&LoaderAttach, &pin);
        auto reg = (NTSTATUS(NTAPI*)(ULONG, void*, void*, void**))GetProcAddress(GetModuleHandleA("ntdll.dll"), "LdrRegisterDllNotification");
        void* cookie; if (reg) reg(0, (void*)OnDll, nullptr, &cookie);
        // Every module already here that is not Windows' own.
        wchar_t win[MAX_PATH] = {}; GetWindowsDirectoryW(win, MAX_PATH); g_winLen = wcslen(win); wcscpy_s(g_win, win);
        auto* peb = NtCurrentTeb()->ProcessEnvironmentBlock;
        for (auto* e = peb->Ldr->InMemoryOrderModuleList.Flink; e != &peb->Ldr->InMemoryOrderModuleList; e = e->Flink)
        {
            auto* m = CONTAINING_RECORD(e, LDR_DATA_TABLE_ENTRY, InMemoryOrderLinks);
            if (m->DllBase && m->DllBase != (void*)self && !IsWindows(&m->FullDllName))  Patch((HMODULE)m->DllBase);
        }
    }
}
