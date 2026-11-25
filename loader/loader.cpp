// NFRUX's process-local loader, a dbghelp.dll proxy: Streamline's calls to the NGX core for feature 11 (frame generation) go to NFRUX, the
// rest to the core untouched. Nothing outside this process changes. SPDX-License-Identifier: MIT
#include "ngx.h"
#include "dbghelp_exports.h"
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
static void L(const char* f, ...) { std::lock_guard<std::mutex> k(g_m); if (!g_log) g_log = fopen("nfrux_loader.log", "w"); va_list a; va_start(a, f); vfprintf(g_log, f, a); va_end(a); fputc('\n', g_log); fflush(g_log); }

static std::map<std::string, void*> g_real;
static std::set<void*> g_handles;

template <class F> static F Real(const char* n) { return reinterpret_cast<F>(g_real[n]); }

// NFRUX's DLL, from NFRUX\nvngx_dlssg.dll beside this one (NFRUX_DLL: another path).
static HMODULE Nfrux()
{
    static HMODULE m = [] {
        char path[MAX_PATH] = {};
        if (const char* e = getenv("NFRUX_DLL")) strcpy_s(path, e);
        else
        {
            HMODULE self = nullptr;
            GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)&L, &self);
            GetModuleFileNameA(self, path, MAX_PATH);
            if (char* s = strrchr(path, '\\')) strcpy_s(s + 1, MAX_PATH - (s + 1 - path), "NFRUX\\nvngx_dlssg.dll");
        }
        HMODULE x = LoadLibraryExA(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        L("%s -> %p (%lu)", path, x, x ? 0 : GetLastError());
        return x;
    }();
    return m;
}
template <class F> static F Ours(const char* n) { return Nfrux() ? reinterpret_cast<F>(GetProcAddress(Nfrux(), n)) : nullptr; }
static std::mutex g_hm;
static bool IsOurs(const NgxHandle* h) { std::lock_guard<std::mutex> k(g_hm); return g_handles.count((void*)h) != 0; }
template <class T> static NgxResult Keep(NgxResult r, T** h) { if (r == kNgxSuccess) { std::lock_guard<std::mutex> k(g_hm); g_handles.insert(*h); } L("feature 11: %x", r); return r; }

// --- D3D12
static NgxResult __cdecl Create12(ID3D12GraphicsCommandList* c, int f, NgxParameter* p, NgxHandle** h)
{
    if (auto o = f == kFeatureFrameGeneration ? Ours<decltype(&Create12)>("NVSDK_NGX_D3D12_CreateFeature") : nullptr) return Keep(o(c, f, p, h), h);
    return Real<decltype(&Create12)>("NVSDK_NGX_D3D12_CreateFeature")(c, f, p, h);
}
static NgxResult __cdecl Eval12(ID3D12GraphicsCommandList* c, const NgxHandle* h, NgxParameter* p, void* cb)
{
    static auto o = Ours<decltype(&Eval12)>("NVSDK_NGX_D3D12_EvaluateFeature");
    return IsOurs(h) ? o(c, h, p, cb) : Real<decltype(&Eval12)>("NVSDK_NGX_D3D12_EvaluateFeature")(c, h, p, cb);
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
static uint32_t g_ml_(); static bool g_atomic_(); static VkDevice g_dev_();
static void NoteVk(VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd) { if (auto f = Ours<void(__cdecl*)(VkDevice, uint32_t, int)>("NFRUX_SetVulkanFeatures"); f && g_dev_()) f(d, g_ml_(), g_atomic_()); if (auto o = Ours<PFN_InitVk2>("NVSDK_NGX_VULKAN_Init_Ext2")) o(0, nullptr, i, pd, d, gi, gd, 0x15, nullptr); }
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
static NgxResult __cdecl ReleaseVk(NgxHandle* h)
{
    if (IsOurs(h)) { { std::lock_guard<std::mutex> k(g_hm); g_handles.erase(h); } return Ours<decltype(&ReleaseVk)>("NVSDK_NGX_VULKAN_ReleaseFeature")(h); }
    return Real<decltype(&ReleaseVk)>("NVSDK_NGX_VULKAN_ReleaseFeature")(h);
}
// --- vkCreateDevice: what the game enables, plus what NFRUX's fast paths want (DESIGN D3).
#include "../core/ml/ffx_vk_ml_runtime.h"
static PFN_vkCreateDevice g_createDevice;
static uint32_t           g_ml;
static bool               g_atomic64;
static VkDevice           g_featureDevice;
static PFN_vkGetInstanceProcAddr Gipa() { return (PFN_vkGetInstanceProcAddr)GetProcAddress(GetModuleHandleA("vulkan-1.dll"), "vkGetInstanceProcAddr"); }

static VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo* ci, const VkAllocationCallbacks* a, VkDevice* out)
{
    auto getF = (PFN_vkGetPhysicalDeviceFeatures2)GetProcAddress(GetModuleHandleA("vulkan-1.dll"), "vkGetPhysicalDeviceFeatures2");
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmS = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features s13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &cmS};
    VkPhysicalDeviceVulkan12Features s12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &s13};
    VkPhysicalDeviceFeatures2 s2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &s12};
    if (getF) getF(pd, &s2);
    // The chain as it is, with copies of the structures changed: found by type, relinked in order.
    struct Copy { alignas(16) unsigned char b[512]; };
    std::vector<std::unique_ptr<Copy>> copies;
    VkDeviceCreateInfo c = *ci;
    VkPhysicalDeviceFeatures core = {};
    VkPhysicalDeviceFeatures2* f2 = nullptr; VkPhysicalDeviceVulkan12Features* f12 = nullptr; VkPhysicalDeviceVulkan13Features* f13 = nullptr;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR* fcm = nullptr; VkPhysicalDeviceVulkanMemoryModelFeatures* fmm = nullptr; VkPhysicalDeviceShaderFloat16Int8Features* f8 = nullptr;
    VkPhysicalDevice8BitStorageFeatures* f8s = nullptr; VkPhysicalDeviceShaderIntegerDotProductFeatures* fdot = nullptr; VkPhysicalDeviceShaderAtomicInt64Features* fat = nullptr;
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
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES: return sizeof(VkPhysicalDeviceShaderAtomicInt64Features);
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
            default: fat = (VkPhysicalDeviceShaderAtomicInt64Features*)use; break;
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
        f.shaderInt64 |= s2.features.shaderInt64;
    };
    if (f2) wantCore(f2->features);
    else { if (ci->pEnabledFeatures) core = *ci->pEnabledFeatures; wantCore(core); c.pEnabledFeatures = &core; }
    VkPhysicalDeviceVulkanMemoryModelFeatures mm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
    VkPhysicalDeviceShaderFloat16Int8Features i8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice8BitStorageFeatures st8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDeviceShaderAtomicInt64Features at = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    if (f12)
    {
        f12->vulkanMemoryModel |= s12.vulkanMemoryModel, f12->vulkanMemoryModelDeviceScope |= s12.vulkanMemoryModelDeviceScope;
        f12->shaderInt8 |= s12.shaderInt8, f12->storageBuffer8BitAccess |= s12.storageBuffer8BitAccess, f12->shaderBufferInt64Atomics |= s12.shaderBufferInt64Atomics;
    }
    else
    {
        if (!fmm) chain(fmm = &mm); if (!f8) chain(f8 = &i8); if (!f8s) chain(f8s = &st8); if (!fat) chain(fat = &at);
        fmm->vulkanMemoryModel |= s12.vulkanMemoryModel, fmm->vulkanMemoryModelDeviceScope |= s12.vulkanMemoryModelDeviceScope;
        f8->shaderInt8 |= s12.shaderInt8, f8s->storageBuffer8BitAccess |= s12.storageBuffer8BitAccess, fat->shaderBufferInt64Atomics |= s12.shaderBufferInt64Atomics;
    }
    if (f13) f13->shaderIntegerDotProduct |= s13.shaderIntegerDotProduct, f13->subgroupSizeControl |= s13.subgroupSizeControl, f13->computeFullSubgroups |= s13.computeFullSubgroups, f13->synchronization2 |= s13.synchronization2;
    else { if (!fdot) chain(fdot = &dot); fdot->shaderIntegerDotProduct |= s13.shaderIntegerDotProduct; }
    if (cmS.cooperativeMatrix)
    {
        if (!has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) ext.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        if (!fcm) chain(fcm = &cm);
        fcm->cooperativeMatrix = VK_TRUE;
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
        if (f13 && s13.subgroupSizeControl && s13.computeFullSubgroups) ml |= FEATURE_SUBGROUP_SIZE_CONTROL;
        if (cmS.cooperativeMatrix) ml |= FEATURE_COOPERATIVE_MATRIX;
    }
    g_ml = ml, g_atomic64 = added && s12.shaderBufferInt64Atomics && s2.features.shaderInt64;
    if (r == VK_SUCCESS) g_featureDevice = *out;
    L("vkCreateDevice -> %d, features %s, ml 0x%x, 64-bit atomics %d", r, added ? "added" : "refused: the game's own call", ml, g_atomic64);
    return r;
}
static uint32_t g_ml_() { return g_ml; } static bool g_atomic_() { return g_atomic64; } static VkDevice g_dev_() { return g_featureDevice; }
static PFN_vkGetInstanceProcAddr g_gipa;
static PFN_vkVoidFunction VKAPI_CALL HookGipa(VkInstance i, const char* n)
{
    PFN_vkVoidFunction f = g_gipa(i, n);
    if (f && n && !strcmp(n, "vkCreateDevice")) { if (!g_createDevice) g_createDevice = (PFN_vkCreateDevice)f; return (PFN_vkVoidFunction)CreateDevice; }
    return f;
}

static void Caps(NgxParameter* p) { if (!p) return; static auto mf = Ours<uint32_t(__cdecl*)()>("NFRUX_MaxFrames"); if (!mf) return; p->Set("DLSSG.MultiFrameCountMax", mf()); p->Set("FrameGeneration.Available", 1); p->Set("FrameGeneration.NeedsUpdatedDriver", 0); p->Set("FrameGeneration.FeatureInitResult", 1); }
template <int N> static const char* kCapName;
template <> const char* kCapName<0> = "NVSDK_NGX_D3D12_GetCapabilityParameters";
template <> const char* kCapName<1> = "NVSDK_NGX_VULKAN_GetCapabilityParameters";
template <> const char* kCapName<2> = "NVSDK_NGX_D3D12_GetParameters";
template <> const char* kCapName<3> = "NVSDK_NGX_VULKAN_GetParameters";
template <int N> static NgxResult __cdecl Cap(NgxParameter** p) { const NgxResult r = Real<NgxResult(__cdecl*)(NgxParameter**)>(kCapName<N>)(p); L("%s -> %x", kCapName<N>, r); if (r == kNgxSuccess) Caps(*p); return r; }
static const std::map<std::string, void*> kWrap = {
    {"NVSDK_NGX_D3D12_GetCapabilityParameters", (void*)Cap<0>}, {"NVSDK_NGX_VULKAN_GetCapabilityParameters", (void*)Cap<1>}, {"NVSDK_NGX_D3D12_GetParameters", (void*)Cap<2>}, {"NVSDK_NGX_VULKAN_GetParameters", (void*)Cap<3>},
    {"NVSDK_NGX_D3D12_CreateFeature", (void*)Create12}, {"NVSDK_NGX_D3D12_EvaluateFeature", (void*)Eval12}, {"NVSDK_NGX_D3D12_ReleaseFeature", (void*)Release12}, {"NVSDK_NGX_D3D12_GetFeatureRequirements", (void*)Req12},
    {"NVSDK_NGX_VULKAN_Init", (void*)InitVk}, {"NVSDK_NGX_VULKAN_Init_Ext2", (void*)InitVk2}, {"NVSDK_NGX_VULKAN_Init_ProjectID_Ext", (void*)InitVkP}, {"NVSDK_NGX_VULKAN_CreateFeature", (void*)CreateVk}, {"NVSDK_NGX_VULKAN_CreateFeature1", (void*)CreateVk1}, {"NVSDK_NGX_VULKAN_EvaluateFeature", (void*)EvalVk}, {"NVSDK_NGX_VULKAN_ReleaseFeature", (void*)ReleaseVk}};

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
    if (HIWORD(n) && r && m == GetModuleHandleA("vulkan-1.dll"))
    {
        if (!strcmp(n, "vkCreateDevice")) { g_createDevice = (PFN_vkCreateDevice)r; return (FARPROC)CreateDevice; }
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
            if (h) { DWORD old; if (VirtualProtect(&t->u1.Function, 8, PAGE_READWRITE, &old)) { t->u1.Function = (ULONGLONG)h; VirtualProtect(&t->u1.Function, 8, old, &old); } }
        }
    }
}
struct NOTIFY { ULONG Flags; const UNICODE_STRING *Full, *Base; PVOID DllBase; ULONG Size; };
static wchar_t g_win[MAX_PATH]; static size_t g_winLen;
static bool IsWindows(const UNICODE_STRING* n) { return g_winLen && n->Buffer && n->Length / 2 > g_winLen && !_wcsnicmp(n->Buffer, g_win, g_winLen); }
static VOID CALLBACK OnDll(ULONG reason, const NOTIFY* d, PVOID) { if (reason == 1 && !IsWindows(d->Full)) { Patch((HMODULE)d->DllBase); } }
BOOL WINAPI DllMain(HINSTANCE, DWORD r, LPVOID)
{
    if (r == DLL_PROCESS_ATTACH)
    {
        auto reg = (NTSTATUS(NTAPI*)(ULONG, void*, void*, void**))GetProcAddress(GetModuleHandleA("ntdll.dll"), "LdrRegisterDllNotification");
        void* cookie; if (reg) reg(0, (void*)OnDll, nullptr, &cookie);
        // Every module already here that is not Windows' own.
        wchar_t win[MAX_PATH] = {}; GetWindowsDirectoryW(win, MAX_PATH); g_winLen = wcslen(win); wcscpy_s(g_win, win);
        auto* peb = NtCurrentTeb()->ProcessEnvironmentBlock;
        for (auto* e = peb->Ldr->InMemoryOrderModuleList.Flink; e != &peb->Ldr->InMemoryOrderModuleList; e = e->Flink)
        {
            auto* m = CONTAINING_RECORD(e, LDR_DATA_TABLE_ENTRY, InMemoryOrderLinks);
            if (m->DllBase && !IsWindows(&m->FullDllName)) Patch((HMODULE)m->DllBase);
        }
    }
    return TRUE;
}