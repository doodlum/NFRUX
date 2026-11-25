// M0a probe: a dbghelp.dll proxy that, in this process only, logs what Streamline asks of the NGX core and answers feature 11
// (frame generation) with a stub that copies the back buffer. Not shipped. SPDX-License-Identifier: MIT
#include "ngx.h"
#include "dbghelp_exports.h"
#include <windows.h>
#include <winternl.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <string>

static FILE* g_log;
static std::mutex g_m;
static void L(const char* f, ...) { std::lock_guard<std::mutex> k(g_m); if (!g_log) g_log = fopen("nfrux_probe.log", "w"); va_list a; va_start(a, f); vfprintf(g_log, f, a); va_end(a); fputc('\n', g_log); fflush(g_log); }

static std::map<std::string, void*> g_real;
static std::set<void*> g_handles;
static PFN_vkGetDeviceProcAddr g_gdpa; static VkDevice g_dev;
template <class F> static F Real(const char* n) { return reinterpret_cast<F>(g_real[n]); }

static const char* kRes[] = {"DLSSG.Backbuffer", "DLSSG.OutputInterpolated", "DLSSG.OutputReal", "DLSSG.Depth", "DLSSG.MVecs", "DLSSG.HUDLess", "DLSSG.UI", "DLSSG.UIAlpha", "DLSSG.NoWarp", "DLSSG.BidirectionalDistortionField"};
static const char* kInts[] = {"DLSSG.MultiFrameCount", "DLSSG.MultiFrameIndex", "DLSSG.FrameID", "DLSSG.IsRecording", "DLSSG.Reset", "DLSSG.DepthInverted", "DLSSG.ColorBuffersHDR", "DLSSG.NotRenderingGameFrames", "DLSSG.DepthSubrectWidth", "DLSSG.DepthSubrectHeight", "DLSSG.BackbufferSubrectWidth", "Width", "Height", "DLSSG.MultiFrameCountMax", "DLSSG.OptLevel", "DLSSG.UseNativeOFA"};
static std::atomic<int> g_n{0};
static void Dump(const char* what, NgxParameter* p, bool vk)
{
    L("== %s", what);
    for (auto n : kInts) { int i; if (p->Get(n, &i) == kNgxSuccess) L("  %s = %d", n, i); }
    for (auto n : {"DLSSG.MvecScaleX", "DLSSG.MvecScaleY", "DLSSG.CameraNear", "DLSSG.CameraFar", "DLSSG.CameraFOV", "DLSSG.JitterOffsetX"}) { float f; if (p->Get(n, &f) == kNgxSuccess) L("  %s = %g", n, f); }
    for (auto n : {"DLSSG.CmdQueue", "DLSSG.CmdAlloc", "DLSSG.CmdList", "DLSSG.ClipToPrevClip", "DLSSG.CameraViewToClip", "DLSSG.Fence"}) { void* v = nullptr; if (p->Get(n, &v) == kNgxSuccess && v) L("  %s = %p", n, v); }
    for (auto n : kRes)
    {
        void* v = nullptr;
        if (vk) { if (p->Get(n, &v) == kNgxSuccess && v) { auto* r = (NgxResourceVk*)v; L("  %s = %ux%u fmt %d aspect %x rw %d", n, r->R.Image.Width, r->R.Image.Height, r->R.Image.Format, r->R.Image.Range.aspectMask, r->ReadWrite); } }
        else { ID3D12Resource* r = nullptr; if ((p->Get(n, &r) == kNgxSuccess && r) || (p->Get(n, &v) == kNgxSuccess && (r = (ID3D12Resource*)v))) { auto d = r->GetDesc(); L("  %s = %llux%u fmt %d flags %x", n, d.Width, d.Height, d.Format, d.Flags); } }
    }
}
static ID3D12Resource* Res12(NgxParameter* p, const char* n) { ID3D12Resource* r = nullptr; void* v = nullptr; if (p->Get(n, &r) == kNgxSuccess && r) return r; return p->Get(n, &v) == kNgxSuccess ? (ID3D12Resource*)v : nullptr; }

// --- D3D12
static NgxResult __cdecl Create12(ID3D12GraphicsCommandList* c, int f, NgxParameter* p, NgxHandle** h)
{
    L("D3D12_CreateFeature %d list %p type %d", f, c, c ? c->GetType() : -1);
    if (f != kFeatureFrameGeneration) return Real<decltype(&Create12)>("NVSDK_NGX_D3D12_CreateFeature")(c, f, p, h);
    Dump("create", p, false); *h = new NgxHandle{0x4e46}; g_handles.insert(*h); return kNgxSuccess;
}
static NgxResult __cdecl Eval12(ID3D12GraphicsCommandList* c, const NgxHandle* h, NgxParameter* p, void* cb)
{
    if (!g_handles.count((void*)h)) return Real<decltype(&Eval12)>("NVSDK_NGX_D3D12_EvaluateFeature")(c, h, p, cb);
    const int n = g_n++;
    if (n < 12 || n % 600 == 0) { L("eval %d list type %d", n, c->GetType()); Dump("evaluate", p, false); }
    ID3D12Resource* b = Res12(p, "DLSSG.Backbuffer");
    for (auto name : {"DLSSG.OutputInterpolated", "DLSSG.OutputReal"})
        if (ID3D12Resource* o = Res12(p, name); o && b && o != b)
        {
            D3D12_RESOURCE_BARRIER x[2] = {};
            x[0].Transition = {b, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE};
            x[1].Transition = {o, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST};
            c->ResourceBarrier(2, x); c->CopyResource(o, b);
            for (auto& y : x) std::swap(y.Transition.StateBefore, y.Transition.StateAfter);
            c->ResourceBarrier(2, x);
        }
    return kNgxSuccess;
}
static NgxResult __cdecl Release12(NgxHandle* h) { if (g_handles.erase(h)) { L("release"); delete h; return kNgxSuccess; } return Real<decltype(&Release12)>("NVSDK_NGX_D3D12_ReleaseFeature")(h); }
static NgxResult __cdecl Req12(void* a, const NgxFeatureDiscoveryInfo* i, NgxFeatureRequirement* r) { L("D3D12_GetFeatureRequirements %d", i ? i->FeatureID : -1); return Real<decltype(&Req12)>("NVSDK_NGX_D3D12_GetFeatureRequirements")(a, i, r); }
// --- Vulkan
static NgxResult __cdecl InitVk(unsigned long long a, const wchar_t* b, VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd, const void* fi, int v) { L("VULKAN_Init"); g_dev = d; g_gdpa = gd; return Real<decltype(&InitVk)>("NVSDK_NGX_VULKAN_Init")(a, b, i, pd, d, gi, gd, fi, v); }
static NgxResult __cdecl InitVk2(unsigned long long a, const wchar_t* b, VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd, int v, const void* fi) { L("VULKAN_Init_Ext2 %p", gd); g_dev = d; g_gdpa = gd; return Real<decltype(&InitVk2)>("NVSDK_NGX_VULKAN_Init_Ext2")(a, b, i, pd, d, gi, gd, v, fi); }
static NgxResult __cdecl InitVkP(const char* pid, int e, const char* ev, const wchar_t* b, VkInstance i, VkPhysicalDevice pd, VkDevice d, PFN_vkGetInstanceProcAddr gi, PFN_vkGetDeviceProcAddr gd, int v, const void* fi) { L("VULKAN_Init_ProjectID_Ext %p", gd); g_dev = d; g_gdpa = gd; return Real<decltype(&InitVkP)>("NVSDK_NGX_VULKAN_Init_ProjectID_Ext")(pid, e, ev, b, i, pd, d, gi, gd, v, fi); }
static NgxResult __cdecl CreateVk(VkCommandBuffer c, int f, NgxParameter* p, NgxHandle** h)
{
    L("VULKAN_CreateFeature %d", f);
    if (f != kFeatureFrameGeneration) return Real<decltype(&CreateVk)>("NVSDK_NGX_VULKAN_CreateFeature")(c, f, p, h);
    Dump("create", p, true); *h = new NgxHandle{0x4e46}; g_handles.insert(*h); return kNgxSuccess;
}
static NgxResult __cdecl CreateVk1(VkDevice d, VkCommandBuffer c, int f, NgxParameter* p, NgxHandle** h)
{
    L("VULKAN_CreateFeature1 %d", f); g_dev = d;
    if (f != kFeatureFrameGeneration) return Real<decltype(&CreateVk1)>("NVSDK_NGX_VULKAN_CreateFeature1")(d, c, f, p, h);
    Dump("create", p, true); *h = new NgxHandle{0x4e46}; g_handles.insert(*h); return kNgxSuccess;
}
static NgxResult __cdecl EvalVk(VkCommandBuffer c, const NgxHandle* h, NgxParameter* p, void* cb)
{
    if (!g_handles.count((void*)h)) return Real<decltype(&EvalVk)>("NVSDK_NGX_VULKAN_EvaluateFeature")(c, h, p, cb);
    const int n = g_n++;
    if (n < 12 || n % 600 == 0) { L("eval %d", n); Dump("evaluate", p, true); }
    void *bv = nullptr, *ov = nullptr;
    p->Get("DLSSG.Backbuffer", &bv);
    if (!g_gdpa) { if (auto v = GetModuleHandleA("vulkan-1.dll")) g_gdpa = (PFN_vkGetDeviceProcAddr)GetProcAddress(v, "vkGetDeviceProcAddr"); }
    static auto copy = g_gdpa && g_dev ? (PFN_vkCmdCopyImage)g_gdpa(g_dev, "vkCmdCopyImage") : nullptr;
    static auto barrier = g_gdpa && g_dev ? (PFN_vkCmdPipelineBarrier)g_gdpa(g_dev, "vkCmdPipelineBarrier") : nullptr;
    for (auto name : {"DLSSG.OutputInterpolated", "DLSSG.OutputReal"})
        if (bv && copy && barrier && p->Get(name, &ov) == kNgxSuccess && ov)
        {
            auto &b = ((NgxResourceVk*)bv)->R.Image, &o = ((NgxResourceVk*)ov)->R.Image;
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
            barrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            VkImageCopy r = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {b.Width, b.Height, 1}};
            copy(c, b.Image, VK_IMAGE_LAYOUT_GENERAL, o.Image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
            barrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
    return kNgxSuccess;
}
static NgxResult __cdecl ReleaseVk(NgxHandle* h) { if (g_handles.erase(h)) { L("release"); delete h; return kNgxSuccess; } return Real<decltype(&ReleaseVk)>("NVSDK_NGX_VULKAN_ReleaseFeature")(h); }

static void Caps(NgxParameter* p) { if (!p) return; p->Set("DLSSG.MultiFrameCountMax", 5u); p->Set("FrameGeneration.Available", 1); p->Set("FrameGeneration.NeedsUpdatedDriver", 0); p->Set("FrameGeneration.FeatureInitResult", 1); }
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
    if (HIWORD(n) && !strcmp(n, "vkCreateDevice")) L("GetProcAddress vkCreateDevice asked");
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
static VOID CALLBACK OnDll(ULONG reason, const NOTIFY* d, PVOID) { if (reason == 1 && _wcsnicmp(d->Full->Buffer, L"C:\\Windows", 10)) { Patch((HMODULE)d->DllBase); } }
BOOL WINAPI DllMain(HINSTANCE, DWORD r, LPVOID)
{
    if (r == DLL_PROCESS_ATTACH)
    {
        auto reg = (NTSTATUS(NTAPI*)(ULONG, void*, void*, void**))GetProcAddress(GetModuleHandleA("ntdll.dll"), "LdrRegisterDllNotification");
        void* cookie; if (reg) reg(0, (void*)OnDll, nullptr, &cookie);
        Patch(GetModuleHandleA(nullptr));
        if (auto m = GetModuleHandleA("sl.interposer.dll")) Patch(m);
    }
    return TRUE;
}