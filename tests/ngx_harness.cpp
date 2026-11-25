// A9, A10: loads nvngx_dlssg.dll as an NGX core does, on D3D12 and Vulkan, and checks the protocol and every UI rule by its
// pixels, on each API and between them.
//   ngx_harness DLL [--minimal] [--soak N] [d3d12|vulkan]     NFRUX_VK_VALIDATION=1, NFRUX_D3D12_DEBUG=1
// SPDX-License-Identifier: MIT
#include "../core/ml/vulkan_network.h"
#include "../loader/ngx.h"

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using Microsoft::WRL::ComPtr;
#ifndef NFRUX_AGILITY
#define NFRUX_AGILITY 620
#endif

// --- NGX parameters, as a core's
struct Params : NgxParameter
{
    std::map<std::string, std::variant<unsigned long long, float, double, unsigned, int, void*>> v;
    template <class T> void Put(const char* n, T x) { v[n] = x; }
    template <class T> NgxResult Number(const char* n, T* o) const
    {
        auto it = v.find(n);
        if (it == v.end()) return kNgxUnsupportedParameter;
        return std::visit([&](auto x) -> NgxResult {
            if constexpr (std::is_pointer_v<decltype(x)>) return kNgxInvalidParameter;
            else return *o = T(x), kNgxSuccess;
        }, it->second);
    }
    NgxResult Pointer(const char* n, void** o) const
    {
        auto it = v.find(n);
        if (it == v.end() || !std::holds_alternative<void*>(it->second)) return kNgxUnsupportedParameter;
        return *o = std::get<void*>(it->second), kNgxSuccess;
    }
    void Set(const char* n, unsigned long long x) override { Put(n, x); }
    void Set(const char* n, float x) override { Put(n, x); }
    void Set(const char* n, double x) override { Put(n, x); }
    void Set(const char* n, unsigned x) override { Put(n, x); }
    void Set(const char* n, int x) override { Put(n, x); }
    void Set(const char* n, ID3D11Resource* x) override { Put(n, static_cast<void*>(x)); }
    void Set(const char* n, ID3D12Resource* x) override { Put(n, static_cast<void*>(x)); }
    void Set(const char* n, void* x) override { Put(n, x); }
    NgxResult Get(const char* n, unsigned long long* o) const override { return Number(n, o); }
    NgxResult Get(const char* n, float* o) const override { return Number(n, o); }
    NgxResult Get(const char* n, double* o) const override { return Number(n, o); }
    NgxResult Get(const char* n, unsigned* o) const override { return Number(n, o); }
    NgxResult Get(const char* n, int* o) const override { return Number(n, o); }
    NgxResult Get(const char* n, ID3D11Resource** o) const override { return Pointer(n, reinterpret_cast<void**>(o)); }
    NgxResult Get(const char* n, ID3D12Resource** o) const override { return Pointer(n, reinterpret_cast<void**>(o)); }
    NgxResult Get(const char* n, void** o) const override { return Pointer(n, o); }
    void Reset() override { v.clear(); }
};

// --- a graphics API with host textures
enum Fmt { F_RGBA8, F_BGRA8, F_RGB10, F_RGBA16F, F_R32F, F_RG16F, F_R8, F_RGBA8S, F_BGRA8S, F_COUNT };  // S: sRGB
static const uint32_t kBytes[F_COUNT] = {4, 4, 4, 8, 4, 4, 1, 4, 4};
using Bytes = std::vector<uint8_t>;
struct Tex
{
    virtual ~Tex() = default;
    Fmt f = F_RGBA8;
    uint32_t w = 0, h = 0;
};
using TexP = std::shared_ptr<Tex>;
struct Api
{
    HMODULE dll = nullptr;
    uint32_t messages = 0;
    virtual ~Api() = default;
    virtual const char* Name() const = 0;
    virtual TexP Create(Fmt f, uint32_t w, uint32_t h, bool output) = 0;
    virtual void Upload(Tex& t, const Bytes& b) = 0;
    virtual Bytes Read(Tex& t) = 0;
    virtual void Bind(Params& p, const char* n, Tex* t) = 0;
    virtual void BindWord(Params& p, const char* n) = 0;  // a 4-byte host buffer the feature writes
    virtual uint32_t ReadWord() = 0;
    virtual NgxHandle* CreateFeature(Params& p) = 0;
    virtual NgxResult Evaluate(NgxHandle* h, Params& p) = 0;  // recorded, submitted, waited for
    bool evaluateC = false;  // through EvaluateFeature_C
    virtual NgxResult Release(NgxHandle* h) = 0;
    virtual void Shutdown() = 0;
    virtual bool Recreate() = 0;
    ComPtr<IDXGIAdapter3> adapter;  // video memory of the process on it, whatever the API
    void FindAdapter(LUID luid)
    {
        ComPtr<IDXGIFactory4> f;
        CreateDXGIFactory1(IID_PPV_ARGS(&f));
        ComPtr<IDXGIAdapter> a;
        if (SUCCEEDED(f->EnumAdapterByLuid(luid, IID_PPV_ARGS(&a)))) a.As(&adapter);
    }
    uint64_t VideoMemory()
    {
        DXGI_QUERY_VIDEO_MEMORY_INFO i = {};
        if (!adapter || FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &i)) || !i.CurrentUsage) printf("  video memory not readable\n"), exit(2);
        return i.CurrentUsage;
    }
    template <class F> F Fn(const char* n)
    {
        F f = reinterpret_cast<F>(GetProcAddress(dll, n));
        if (!f) printf("  export %s missing\n", n), exit(2);
        return f;
    }
};

// Inputs rest in NON_PIXEL_SHADER_RESOURCE, outputs in UNORDERED_ACCESS, as the programming guide asks of a host.
struct Tex12 : Tex { ComPtr<ID3D12Resource> r; D3D12_RESOURCE_STATES home = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; };
struct Api12 : Api
{
    ComPtr<ID3D12Device> d;
    ComPtr<ID3D12CommandQueue> q;
    ComPtr<ID3D12CommandAllocator> al;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12InfoQueue> info;
    uint64_t fv = 0;
    bool Recreate() override
    {
        Shutdown();
        word.Reset(), cl.Reset(), al.Reset(), q.Reset(), fence.Reset(), info.Reset();
        const ULONG left = d.Reset();
        if (left) printf("    (%lu references left on the old device)\n", left);
        return Init() && left == 0;
    }
    static constexpr DXGI_FORMAT kF[F_COUNT] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                                DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB};
    const char* Name() const override { return "d3d12"; }
    bool Init()
    {
        if (getenv("NFRUX_D3D12_DEBUG"))
        {
            // From the Agility SDK package in .\D3D12 beside the exe (Windows' own core would want Graphics Tools).
            ComPtr<ID3D12SDKConfiguration1> sdk;
            ComPtr<ID3D12DeviceFactory> factory;
            ComPtr<ID3D12Debug1> dbg;
            HRESULT hr = D3D12GetInterface(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(&sdk));
            if (SUCCEEDED(hr)) hr = sdk->CreateDeviceFactory(NFRUX_AGILITY, ".\\D3D12\\", IID_PPV_ARGS(&factory));
            if (SUCCEEDED(hr)) hr = factory->GetConfigurationInterface(CLSID_D3D12Debug, IID_PPV_ARGS(&dbg));
            if (FAILED(hr)) return printf("  no D3D12 debug layer: %08lx\n", hr), false;
            dbg->EnableDebugLayer(), dbg->SetEnableGPUBasedValidation(atoi(getenv("NFRUX_D3D12_DEBUG")) > 1);
            if (FAILED(hr = factory->CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d)))) return printf("  debug device: %08lx\n", hr), false;
        }
        if (getenv("NFRUX_WARP"))  // the software adapter: no vendor, no CUDA
        {
            ComPtr<IDXGIFactory4> wf;
            ComPtr<IDXGIAdapter> wa;
            CreateDXGIFactory1(IID_PPV_ARGS(&wf)), wf->EnumWarpAdapter(IID_PPV_ARGS(&wa));
            if (FAILED(D3D12CreateDevice(wa.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d)))) return printf("  no WARP\n"), false;
        }
        if (!d && FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d)))) return false;
        if (SUCCEEDED(d.As(&info)))
        {
            info->SetMessageCountLimit(100000);
            printf("  D3D12 debug layer on%s\n", atoi(getenv("NFRUX_D3D12_DEBUG")) > 1 ? ", GPU validation on" : "");
        }
        FindAdapter(d->GetAdapterLuid());
        D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_COMPUTE};
        d->CreateCommandQueue(&qd, IID_PPV_ARGS(&q));
        d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&al));
        d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, al.Get(), nullptr, IID_PPV_ARGS(&cl)), cl->Close();
        d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        return Fn<NgxResult (*)(unsigned long long, const wchar_t*, ID3D12Device*, int, const void*)>("NVSDK_NGX_D3D12_Init_Ext")(0, L".", d.Get(), 0x15, nullptr) == kNgxSuccess;
    }
    void Begin() { al->Reset(), cl->Reset(al.Get(), nullptr); }
    void End()
    {
        cl->Close();
        ID3D12CommandList* l = cl.Get();
        q->ExecuteCommandLists(1, &l), q->Signal(fence.Get(), ++fv), fence->SetEventOnCompletion(fv, nullptr);
        for (uint64_t i = 0; info && i < info->GetNumStoredMessages(); ++i)
        {
            SIZE_T n = 0;
            info->GetMessage(i, nullptr, &n);
            std::vector<uint8_t> b(n);
            auto* m = reinterpret_cast<D3D12_MESSAGE*>(b.data());
            if (FAILED(info->GetMessage(i, m, &n)) || m->Severity > D3D12_MESSAGE_SEVERITY_WARNING) continue;
            if (messages++ < 12) printf("  D3D12: %s\n", m->pDescription);
        }
        if (info) info->ClearStoredMessages();
    }
    ComPtr<ID3D12Resource> Buffer(D3D12_HEAP_TYPE heap, uint64_t size)
    {
        D3D12_HEAP_PROPERTIES hp = {heap};
        D3D12_RESOURCE_DESC r = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, size, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        ComPtr<ID3D12Resource> b;
        d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, heap == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&b));
        return b;
    }
    TexP Create(Fmt f, uint32_t w, uint32_t h, bool output) override
    {
        auto t = std::make_shared<Tex12>();
        t->f = f, t->w = w, t->h = h;
        D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC r = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, w, h, 1, 1, output && f == F_RGBA8S ? DXGI_FORMAT_R8G8B8A8_TYPELESS : output && f == F_BGRA8S ? DXGI_FORMAT_B8G8R8A8_TYPELESS : kF[f], {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                 output ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
        if (output) t->home = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, t->home, nullptr, IID_PPV_ARGS(&t->r));
        return t;
    }
    void Copy(Tex& t, ID3D12Resource* buf, uint32_t pitch, bool up)
    {
        auto* r = static_cast<Tex12&>(t).r.Get();
        D3D12_TEXTURE_COPY_LOCATION lt = {r, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, lb = {buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        lb.PlacedFootprint.Footprint = {kF[t.f], t.w, t.h, 1, pitch};
        Begin();
        D3D12_RESOURCE_BARRIER b = {};
        b.Transition = {r, 0, static_cast<Tex12&>(t).home, up ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE};
        cl->ResourceBarrier(1, &b);
        cl->CopyTextureRegion(up ? &lt : &lb, 0, 0, 0, up ? &lb : &lt, nullptr);
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        cl->ResourceBarrier(1, &b);
        End();
    }
    void Upload(Tex& t, const Bytes& data) override
    {
        const uint32_t row = t.w * kBytes[t.f], pitch = (row + 255) & ~255u;
        auto up = Buffer(D3D12_HEAP_TYPE_UPLOAD, uint64_t(pitch) * t.h);
        uint8_t* p;
        up->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (uint32_t y = 0; y < t.h; ++y) memcpy(p + size_t(y) * pitch, &data[size_t(y) * row], row);
        up->Unmap(0, nullptr);
        Copy(t, up.Get(), pitch, true);
    }
    Bytes Read(Tex& t) override
    {
        const uint32_t row = t.w * kBytes[t.f], pitch = (row + 255) & ~255u;
        auto rb = Buffer(D3D12_HEAP_TYPE_READBACK, uint64_t(pitch) * t.h);
        Copy(t, rb.Get(), pitch, false);
        Bytes o(size_t(row) * t.h);
        uint8_t* p;
        rb->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (uint32_t y = 0; y < t.h; ++y) memcpy(&o[size_t(y) * row], p + size_t(y) * pitch, row);
        rb->Unmap(0, nullptr);
        return o;
    }
    void Bind(Params& p, const char* n, Tex* t) override { p.Set(n, t ? static_cast<Tex12*>(t)->r.Get() : static_cast<ID3D12Resource*>(nullptr)); }
    ComPtr<ID3D12Resource> word;
    void BindWord(Params& p, const char* n) override
    {
        if (!word)
        {
            D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC r = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
            d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&word));
            D3D12_RESOURCE_BARRIER b = {};
            b.Transition = {word.Get(), 0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
            Begin(), cl->ResourceBarrier(1, &b), End();
        }
        p.Set(n, word.Get());
    }
    uint32_t ReadWord() override
    {
        auto rb = Buffer(D3D12_HEAP_TYPE_READBACK, 256);
        D3D12_RESOURCE_BARRIER b = {};
        b.Transition = {word.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
        Begin(), cl->ResourceBarrier(1, &b), cl->CopyResource(rb.Get(), word.Get()), std::swap(b.Transition.StateBefore, b.Transition.StateAfter), cl->ResourceBarrier(1, &b), End();
        uint32_t* p;
        rb->Map(0, nullptr, reinterpret_cast<void**>(&p));
        const uint32_t v = *p;
        rb->Unmap(0, nullptr);
        return v;
    }
    NgxHandle* CreateFeature(Params& p) override
    {
        NgxHandle* h = nullptr;
        Begin();
        Fn<NgxResult (*)(ID3D12GraphicsCommandList*, int, NgxParameter*, NgxHandle**)>("NVSDK_NGX_D3D12_CreateFeature")(cl.Get(), kFeatureFrameGeneration, &p, &h);
        End();
        return h;
    }
    NgxResult Evaluate(NgxHandle* h, Params& p) override
    {
        Begin();
        const NgxResult r = Fn<NgxResult (*)(ID3D12GraphicsCommandList*, const NgxHandle*, NgxParameter*, void*)>(evaluateC ? "NVSDK_NGX_D3D12_EvaluateFeature_C" : "NVSDK_NGX_D3D12_EvaluateFeature")(cl.Get(), h, &p, nullptr);
        End();
        return r;
    }
    NgxResult Release(NgxHandle* h) override { return Fn<NgxResult (*)(NgxHandle*)>("NVSDK_NGX_D3D12_ReleaseFeature")(h); }
    void Shutdown() override { Fn<NgxResult (*)(ID3D12Device*)>("NVSDK_NGX_D3D12_Shutdown1")(d.Get()); }
};

struct TexV : Tex
{
    VkDevice d;
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
    NgxResourceVk res = {};
    VkImageLayout home = VK_IMAGE_LAYOUT_GENERAL;  // an input in the minimal run: SHADER_READ_ONLY_OPTIMAL, not read-write
    ~TexV() { vkDestroyImageView(d, view, nullptr), vkDestroyImage(d, img, nullptr), vkFreeMemory(d, mem, nullptr); }
};
struct ApiV : Api
{
    bool minimal = false;
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dv = VK_NULL_HANDLE;
    VkQueue queue;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkPhysicalDeviceMemoryProperties mp;
    static constexpr VkFormat kF[F_COUNT] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_R16G16B16A16_SFLOAT,
                                             VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB};
    const char* Name() const override { return minimal ? "vulkan (minimal 1.2)" : "vulkan"; }
    static VKAPI_ATTR VkBool32 VKAPI_CALL OnMessage(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* m, void* self)
    {
        if (static_cast<ApiV*>(self)->messages++ < 12) printf("  Vulkan: %s\n", m->pMessage);
        return VK_FALSE;
    }
    bool Init()
    {
        VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        ai.apiVersion = minimal ? VK_API_VERSION_1_2 : VK_API_VERSION_1_3;
        VkInstanceCreateInfo ii = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ii.pApplicationInfo = &ai;
        const char *layer = "VK_LAYER_KHRONOS_validation", *dbg = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        const bool validation = getenv("NFRUX_VK_VALIDATION") != nullptr;
        const VkValidationFeatureEnableEXT sync = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
        VkValidationFeaturesEXT vf = {VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, nullptr, 1, &sync};
        if (validation) ii.enabledLayerCount = 1, ii.ppEnabledLayerNames = &layer, ii.enabledExtensionCount = 1, ii.ppEnabledExtensionNames = &dbg, ii.pNext = &vf;
        if (vkCreateInstance(&ii, nullptr, &inst)) return false;
        if (validation)
        {
            VkDebugUtilsMessengerCreateInfoEXT m = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            m.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            m.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT, m.pfnUserCallback = OnMessage, m.pUserData = this;
            VkDebugUtilsMessengerEXT h;
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT"))(inst, &m, nullptr, &h);
            printf("  Vulkan validation on, synchronisation included\n");
        }
        uint32_t n = 1;
        vkEnumeratePhysicalDevices(inst, &n, &pd);
        VkPhysicalDeviceIDProperties idp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 pp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idp};
        vkGetPhysicalDeviceProperties2(pd, &pp);
        LUID luid;
        memcpy(&luid, idp.deviceLUID, sizeof(luid)), FindAdapter(luid);
        std::vector<const char*> ext;
        const uint32_t ml = minimal ? 0 : ffx_ml::QueryDeviceFeatures(inst, pd, vkGetInstanceProcAddr, &ext);
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR, &cm};
        VkPhysicalDeviceVulkan13Features v13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &bf};
        VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
        VkPhysicalDeviceVulkan11Features v11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
        VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
        vkGetPhysicalDeviceFeatures2(pd, &f2);
        f2.features.robustBufferAccess = VK_FALSE;
        VkPhysicalDeviceFeatures2 least = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        least.features.shaderStorageImageExtendedFormats = VK_TRUE;
        const float prio = 1;
        uint32_t family = 0, qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
        for (uint32_t i = 0; i < qn; ++i) if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { family = i; break; }
        VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, family, 1, &prio};
        VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, minimal ? &least : &f2};
        di.queueCreateInfoCount = 1, di.pQueueCreateInfos = &qi, di.enabledExtensionCount = uint32_t(ext.size()), di.ppEnabledExtensionNames = ext.data();
        if (vkCreateDevice(pd, &di, nullptr, &dv)) return false;
        vkGetDeviceQueue(dv, family, 0, &queue);
        VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family};
        vkCreateCommandPool(dv, &pi, nullptr, &pool);
        VkCommandBufferAllocateInfo ca = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        vkAllocateCommandBuffers(dv, &ca, &cb);
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        if (!minimal) Fn<void (*)(VkDevice, uint32_t)>("NFRUX_SetVulkanFeatures")(dv, ml);
        return Fn<NgxResult (*)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, int, const void*)>("NVSDK_NGX_VULKAN_Init_Ext2")(
                   0, L".", inst, pd, dv, nullptr, nullptr, 0x15, nullptr) == kNgxSuccess;
    }
    bool Recreate() override
    {
        vkDestroyCommandPool(dv, pool, nullptr);
        vkDestroyBuffer(dv, word, nullptr), vkFreeMemory(dv, wordMemory, nullptr), word = VK_NULL_HANDLE;
        Fn<void (*)(VkDevice)>("NFRUX_DropVulkanDevice")(dv);  // what the loader does before vkDestroyDevice
        vkDestroyDevice(dv, nullptr);
        return Init();
    }
    VkDeviceMemory Memory(VkMemoryRequirements r, VkMemoryPropertyFlags want)
    {
        VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, r.size, 0};
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((r.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) { ma.memoryTypeIndex = i; break; }
        VkDeviceMemory m = VK_NULL_HANDLE;
        vkAllocateMemory(dv, &ma, nullptr, &m);
        return m;
    }
    void Begin() { VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(cb, &bi); }
    void End()
    {
        vkEndCommandBuffer(cb);
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1, si.pCommandBuffers = &cb;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), vkQueueWaitIdle(queue);
    }
    void Barrier(VkImage i, VkImageLayout from, VkImageLayout to)
    {
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, from, to,
                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, i, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
    TexP Create(Fmt f, uint32_t w, uint32_t h, bool output) override
    {
        auto t = std::make_shared<TexV>();
        t->f = f, t->w = w, t->h = h, t->d = dv;
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ic.imageType = VK_IMAGE_TYPE_2D, ic.format = kF[f], ic.extent = {w, h, 1}, ic.mipLevels = ic.arrayLayers = 1, ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (output && f != F_BGRA8 && f < F_RGBA8S ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
        vkCreateImage(dv, &ic, nullptr, &t->img);
        VkMemoryRequirements r;
        vkGetImageMemoryRequirements(dv, t->img, &r);
        t->mem = Memory(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkBindImageMemory(dv, t->img, t->mem, 0);
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = t->img, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = kF[f], vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(dv, &vi, nullptr, &t->view);
        t->res.R.Image = {t->view, t->img, vi.subresourceRange, kF[f], w, h}, t->res.Type = 0, t->res.ReadWrite = output || !minimal;
        if (!t->res.ReadWrite) t->home = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        Begin(), Barrier(t->img, VK_IMAGE_LAYOUT_UNDEFINED, t->home), End();
        return t;
    }
    void Copy(Tex& t, Bytes* out, const Bytes* in)
    {
        auto& x = static_cast<TexV&>(t);
        VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bc.size = uint64_t(t.w) * t.h * kBytes[t.f], bc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBuffer b;
        vkCreateBuffer(dv, &bc, nullptr, &b);
        VkMemoryRequirements r;
        vkGetBufferMemoryRequirements(dv, b, &r);
        VkDeviceMemory m = Memory(r, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkBindBufferMemory(dv, b, m, 0);
        void* p;
        vkMapMemory(dv, m, 0, VK_WHOLE_SIZE, 0, &p);
        if (in) memcpy(p, in->data(), bc.size);
        Begin();
        Barrier(x.img, x.home, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {t.w, t.h, 1}};
        if (in) vkCmdCopyBufferToImage(cb, b, x.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        else vkCmdCopyImageToBuffer(cb, x.img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c);
        Barrier(x.img, VK_IMAGE_LAYOUT_GENERAL, x.home);
        VkMemoryBarrier hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        End();
        if (out) out->resize(bc.size), memcpy(out->data(), p, bc.size);
        vkDestroyBuffer(dv, b, nullptr), vkFreeMemory(dv, m, nullptr);
    }
    VkBuffer word = VK_NULL_HANDLE;
    VkDeviceMemory wordMemory = VK_NULL_HANDLE;
    uint32_t* wordMap = nullptr;
    NgxResourceVk wordRes = {};
    void BindWord(Params& p, const char* n) override
    {
        if (!word)
        {
            VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bc.size = 256, bc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            vkCreateBuffer(dv, &bc, nullptr, &word);
            VkMemoryRequirements r;
            vkGetBufferMemoryRequirements(dv, word, &r);
            wordMemory = Memory(r, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkBindBufferMemory(dv, word, wordMemory, 0), vkMapMemory(dv, wordMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&wordMap));
            wordRes.R.Buffer.Buffer = word, wordRes.Type = 1, wordRes.ReadWrite = true;
        }
        p.Set(n, static_cast<void*>(&wordRes));
    }
    uint32_t ReadWord() override { return *wordMap; }
    void Upload(Tex& t, const Bytes& b) override { Copy(t, nullptr, &b); }
    Bytes Read(Tex& t) override { Bytes o; Copy(t, &o, nullptr); return o; }
    void Bind(Params& p, const char* n, Tex* t) override { p.Set(n, t ? static_cast<void*>(&static_cast<TexV*>(t)->res) : nullptr); }
    NgxHandle* CreateFeature(Params& p) override
    {
        NgxHandle* h = nullptr;
        Begin();
        Fn<NgxResult (*)(VkDevice, VkCommandBuffer, int, NgxParameter*, NgxHandle**)>("NVSDK_NGX_VULKAN_CreateFeature1")(dv, cb, kFeatureFrameGeneration, &p, &h);
        End();
        return h;
    }
    NgxResult Evaluate(NgxHandle* h, Params& p) override
    {
        Begin();
        const NgxResult r = Fn<NgxResult (*)(VkCommandBuffer, const NgxHandle*, NgxParameter*, void*)>(evaluateC ? "NVSDK_NGX_VULKAN_EvaluateFeature_C" : "NVSDK_NGX_VULKAN_EvaluateFeature")(cb, h, &p, nullptr);
        End();
        return r;
    }
    NgxResult Release(NgxHandle* h) override { return Fn<NgxResult (*)(NgxHandle*)>("NVSDK_NGX_VULKAN_ReleaseFeature")(h); }
    void Shutdown() override { Fn<NgxResult (*)(VkDevice)>("NVSDK_NGX_VULKAN_Shutdown1")(dv); }
};

// --- images
static uint16_t Half(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    const int e = int((x >> 23) & 255) - 112;
    if (e <= 0) return uint16_t(s);
    return uint16_t(s | (std::min(e, 30) << 10) | ((x >> 13) & 1023));
}
static float Unhalf(uint16_t h)
{
    const int e = (h >> 10) & 31;
    const float m = float(h & 1023) / 1024.0f, v = e ? ldexpf(1.0f + m, e - 15) : ldexpf(m, -14);
    return h & 0x8000 ? -v : v;
}
struct Image  // RGBA float
{
    uint32_t w = 0, h = 0;
    std::vector<float> p;
    Image() = default;
    Image(uint32_t w_, uint32_t h_, float v = 0) : w(w_), h(h_), p(size_t(w_) * h_ * 4, v) {}
    float* at(int x, int y) { return &p[(size_t(y) * w + x) * 4]; }
    const float* at(int x, int y) const { return &p[(size_t(y) * w + x) * 4]; }
};
static Bytes Encode(const Image& im, Fmt f)
{
    Bytes o(im.p.size() / 4 * kBytes[f]);
    auto u = [](float v, float m) { return uint32_t(std::clamp(v, 0.0f, 1.0f) * m + 0.5f); };
    for (size_t i = 0; i < im.p.size() / 4; ++i)
    {
        const float* c = &im.p[i * 4];
        uint8_t* d = &o[i * kBytes[f]];
        if (f == F_RGBA8 || f == F_RGBA8S) for (int k = 0; k < 4; ++k) d[k] = uint8_t(u(c[k], 255));
        if (f == F_BGRA8 || f == F_BGRA8S) d[0] = uint8_t(u(c[2], 255)), d[1] = uint8_t(u(c[1], 255)), d[2] = uint8_t(u(c[0], 255)), d[3] = uint8_t(u(c[3], 255));
        if (f == F_RGB10) { const uint32_t v = u(c[0], 1023) | u(c[1], 1023) << 10 | u(c[2], 1023) << 20 | u(c[3], 3) << 30; memcpy(d, &v, 4); }
        if (f == F_RGBA16F) { uint16_t h[4]; for (int k = 0; k < 4; ++k) h[k] = Half(c[k]); memcpy(d, h, 8); }
        if (f == F_RG16F) { uint16_t h[2] = {Half(c[0]), Half(c[1])}; memcpy(d, h, 4); }
        if (f == F_R32F) memcpy(d, c, 4);
        if (f == F_R8) d[0] = uint8_t(u(c[0], 255));
    }
    return o;
}
static Image Decode(const Bytes& b, Fmt f, uint32_t w, uint32_t h)
{
    Image o(w, h);
    for (size_t i = 0; i < size_t(w) * h; ++i)
    {
        const uint8_t* d = &b[i * kBytes[f]];
        float* c = &o.p[i * 4];
        if (f == F_RGBA8 || f == F_RGBA8S) for (int k = 0; k < 4; ++k) c[k] = d[k] / 255.0f;
        if (f == F_BGRA8 || f == F_BGRA8S) c[0] = d[2] / 255.0f, c[1] = d[1] / 255.0f, c[2] = d[0] / 255.0f, c[3] = d[3] / 255.0f;
        if (f == F_RGB10) { uint32_t v; memcpy(&v, d, 4); c[0] = (v & 1023) / 1023.0f, c[1] = (v >> 10 & 1023) / 1023.0f, c[2] = (v >> 20 & 1023) / 1023.0f; }
        if (f == F_RGBA16F) { uint16_t hh[4]; memcpy(hh, d, 8); for (int k = 0; k < 4; ++k) c[k] = Unhalf(hh[k]); }
    }
    return o;
}
static float Hash(int x, int y) { uint32_t h = uint32_t(x) * 374761393u + uint32_t(y) * 668265263u; h = (h ^ (h >> 13)) * 1274126177u; return float((h ^ (h >> 16)) & 255) / 255.0f; }
static float Smooth(float x, float y)
{
    const int ix = int(floorf(x)), iy = int(floorf(y));
    const float fx = x - ix, fy = y - iy;
    const float a = Hash(ix, iy) * (1 - fx) + Hash(ix + 1, iy) * fx, b = Hash(ix, iy + 1) * (1 - fx) + Hash(ix + 1, iy + 1) * fx;
    return a * (1 - fy) + b * fy;
}
// The world at frame t: a texture panning (8, 2) pixels a frame, offset (ox, oy).
static Image World(uint32_t w, uint32_t h, float t, float scale = 1, float ox = 0, float oy = 0)
{
    Image o(w, h, 1);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (int k = 0; k < 3; ++k)
                o.at(x, y)[k] = scale * (0.1f + 0.7f * Smooth((x + ox - 8 * t) / 19.0f + k * 91, (y + oy - 2 * t) / 19.0f));
    return o;
}
static double Psnr(const Image& a, const Image& b, int border = 0, double peak = 1)
{
    double e = 0, n = 0;
    for (int y = border; y < int(a.h) - border; ++y)
        for (int x = border; x < int(a.w) - border; ++x)
            for (int k = 0; k < 3; ++k) e += double(a.at(x, y)[k] - b.at(x, y)[k]) * (a.at(x, y)[k] - b.at(x, y)[k]), ++n;
    return 10 * log10(peak * peak / std::max(1e-12, e / n));
}
static double MaxDiff(const Image& a, const Image& b, const std::function<bool(int, int)>& in)
{
    double m = 0;
    for (uint32_t y = 0; y < a.h; ++y)
        for (uint32_t x = 0; x < a.w; ++x)
            if (in(x, y)) for (int k = 0; k < 3; ++k) m = std::max(m, double(fabsf(a.at(x, y)[k] - b.at(x, y)[k])));
    return m;
}
static Image Cut(const Image& a, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h)
{
    Image o(w, h);
    for (uint32_t y = 0; y < h; ++y) memcpy(o.at(0, y), a.at(x0, y0 + y), w * 16);
    return o;
}

static int g_bad = 0, g_checks = 0;
static void Check(bool ok, const char* what, double value)
{
    ++g_checks;
    printf("    %s  %s (%.4f)\n", ok ? "ok  " : "FAIL", what, value);
    if (!ok) ++g_bad;
}

// --- one feature with its inputs
constexpr uint32_t W = 640, H = 360;
struct Scene
{
    Fmt back = F_RGBA8, hud = F_RGBA8, ui = F_RGBA8, alpha = F_R8;
    bool hdr = false;
    float mvOffset[2] = {};
    uint32_t w = W, h = H, dw = W, dh = H, gen = 2, frames = 4;  // dw x dh: depth and vectors
    std::function<Image(int)> backI, hudI, uiI, alphaI, noWarpI, fieldI;
    std::function<void(Params&, int)> extra;
};
struct Out { std::vector<Image> gen; std::vector<Bytes> raw; Image back; bool real = true, ok = true; };
static Out Run(Api& a, const Scene& s, NgxHandle* use = nullptr)
{
    Out o;
    Params p;
    auto tex = [&](Fmt f, uint32_t w, uint32_t h, bool out = false) { return a.Create(f, w, h, out); };
    TexP back = tex(s.back, s.w, s.h), out = tex(s.back, s.w, s.h, true), real = tex(s.back, s.w, s.h, true), depth = tex(F_R32F, s.dw, s.dh), mv = tex(F_RG16F, s.dw, s.dh);
    TexP hud, ui, al, nw, fd;
    Image dI(s.dw, s.dh, 0.9f), mI(s.dw, s.dh);
    for (uint32_t i = 0; i < s.dw * s.dh; ++i) mI.p[i * 4] = -8 + s.mvOffset[0], mI.p[i * 4 + 1] = -2 + s.mvOffset[1];
    a.Upload(*depth, Encode(dI, F_R32F)), a.Upload(*mv, Encode(mI, F_RG16F));
    a.Bind(p, "DLSSG.Backbuffer", back.get()), a.Bind(p, "DLSSG.OutputInterpolated", out.get()), a.Bind(p, "DLSSG.OutputReal", real.get());
    a.Bind(p, "DLSSG.Depth", depth.get()), a.Bind(p, "DLSSG.MVecs", mv.get());
    float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    p.Set("DLSSG.ClipToPrevClip", static_cast<void*>(id)), p.Set("DLSSG.CameraViewToClip", static_cast<void*>(id));
    p.Set("DLSSG.MvecScaleX", 1.0f), p.Set("DLSSG.MvecScaleY", 1.0f), p.Set("DLSSG.CameraNear", 0.1f), p.Set("DLSSG.CameraFar", 100.0f), p.Set("DLSSG.CameraFOV", 1.0f);
    p.Set("DLSSG.ColorBuffersHDR", int(s.hdr)), p.Set("DLSSG.MultiFrameCount", s.gen);
    NgxHandle* h = use ? use : a.CreateFeature(p);
    if (!h) { o.ok = false; return o; }
    auto put = [&](TexP& t, const char* n, Fmt f, const Image& im) { if (!t) t = tex(f, im.w, im.h), a.Bind(p, n, t.get()); a.Upload(*t, Encode(im, f)); };
    for (uint32_t i = 0; i < s.frames; ++i)
    {
        o.back = s.backI(i);
        const Bytes backBytes = Encode(o.back, s.back);
        a.Upload(*back, backBytes);
        if (s.hudI) put(hud, "DLSSG.HUDLess", s.hud, s.hudI(i));
        if (s.uiI) put(ui, "DLSSG.UI", s.ui, s.uiI(i));
        if (s.alphaI) put(al, "DLSSG.UIAlpha", s.alpha, s.alphaI(i));
        if (s.noWarpI) put(nw, "DLSSG.NoWarp", F_R8, s.noWarpI(i));
        if (s.fieldI) put(fd, "DLSSG.BidirectionalDistortionField", F_RGBA16F, s.fieldI(i));
        p.Set("DLSSG.Reset", int(i == 0));
        o.gen.clear(), o.raw.clear();
        for (uint32_t k = 1; k <= s.gen; ++k)
        {
            p.Set("DLSSG.MultiFrameIndex", k);
            if (s.extra) s.extra(p, i);
            o.ok = a.Evaluate(h, p) == kNgxSuccess && o.ok;
            o.raw.push_back(a.Read(*out));
            o.gen.push_back(Decode(o.raw.back(), s.back, s.w, s.h));
            o.real = o.real && a.Read(*real) == backBytes;
        }
        o.back = Decode(backBytes, s.back, s.w, s.h);
    }
    if (!use) a.Release(h);
    return o;
}

// The UI: an opaque rectangle and a half-transparent one.
static bool InUi(int x, int y) { return x >= 100 && x < 220 && y >= 60 && y < 120; }
static bool InSoft(int x, int y) { return x >= 300 && x < 420 && y >= 200 && y < 280; }
static Image WithUi(Image im, float level = 1)
{
    for (uint32_t y = 0; y < im.h; ++y)
        for (uint32_t x = 0; x < im.w; ++x)
            for (int k = 0; k < 3; ++k)
            {
                float& c = im.at(x, y)[k];
                if (InUi(x, y)) c = level * (k == 0 ? 1.0f : 0.95f);
                else if (InSoft(x, y)) c = 0.5f * c + 0.5f * level;
            }
    return im;
}
static Image Opacity(bool red)
{
    Image a(W, H);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) a.at(x, y)[red ? 0 : 3] = InUi(x, y) ? 1.0f : InSoft(x, y) ? 0.5f : 0.0f;
    return a;
}
static auto All = [](int, int) { return true; };

using Results = std::map<std::string, Bytes>;
static void Scenes(Api& a, Results& res)
{
    auto keep = [&](const char* n, const Out& o) { for (size_t k = 0; k < o.raw.size(); ++k) res[std::string(n) + " gen" + std::to_string(k)] = o.raw[k]; };
    auto truth = [](int frame, uint32_t k, uint32_t n) { return World(W, H, frame - 1 + float(k + 1) / float(n + 1)); };
    const double code = 1.01 / 255;
    auto world = [](int i) { return World(W, H, float(i)); };
    auto withUi = [](int i) { return WithUi(World(W, H, float(i))); };
    Scene plain;
    plain.backI = world;
    printf("  no UI\n");
    const Out base = Run(a, plain);
    keep("plain", base);
    for (uint32_t k = 0; k < 2; ++k) { const double q = Psnr(base.gen[k], truth(3, k, 2), 24); Check(q > 30, "the frame between, dB against the truth", q); }
    Check(base.real && base.ok, "OutputReal is the back buffer, every evaluate succeeds", 1);
    for (uint32_t n : {1u, 3u, 5u})
    {
        Scene s = plain;
        s.gen = n;
        const Out o = Run(a, s);
        double worst = 1e9;
        for (uint32_t k = 0; k < n; ++k) worst = std::min(worst, Psnr(o.gen[k], truth(3, k, n), 24));
        char t[64];
        snprintf(t, sizeof(t), "%ux: worst dB against the truth", n + 1);
        Check(o.ok && worst > 30, t, worst);
        keep(t, o);
    }

    printf("  HUD-less, difference test\n");
    Scene d = plain;
    d.backI = withUi, d.hudI = world;
    Out o = Run(a, d);
    keep("difference", o);
    Check(MaxDiff(o.gen[0], o.back, InUi) == 0, "opaque UI is the frame exactly", MaxDiff(o.gen[0], o.back, InUi));
    auto outside = [](int x, int y) { return !InUi(x & ~1, y & ~1) && !InUi(x | 1, y | 1) && !InSoft(x & ~1, y & ~1) && !InSoft(x | 1, y | 1); };
    Check(MaxDiff(o.gen[0], base.gen[0], outside) == 0, "outside the UI: as without it", MaxDiff(o.gen[0], base.gen[0], outside));
    for (int layer = 0; layer < 2; ++layer)
    {
        printf("  HUD-less, %s\n", layer ? "UI layer" : "UI alpha");
        Scene s = d;
        if (layer) s.uiI = [](int) { return Opacity(false); };
        else s.alphaI = [](int) { return Opacity(true); };
        Out u = Run(a, s);
        keep(layer ? "layer" : "alpha", u);
        Check(MaxDiff(u.gen[0], u.back, InUi) <= code, "opaque UI is the frame", MaxDiff(u.gen[0], u.back, InUi));
        Check(MaxDiff(u.gen[0], base.gen[0], outside) <= code, "no UI: as without it", MaxDiff(u.gen[0], base.gen[0], outside));
        Image expect = base.gen[0];
        const Image cur = Decode(Encode(World(W, H, 3), F_RGBA8), F_RGBA8, W, H);
        for (size_t i = 0; i < expect.p.size(); ++i) { const float al = 128.0f / 255.0f; expect.p[i] = std::clamp(u.back.p[i] - (1 - al) * cur.p[i] + (1 - al) * expect.p[i], 0.0f, 1.0f); }
        Check(MaxDiff(u.gen[0], expect, InSoft) <= 1.6 / 255, "half-transparent UI over the interpolated frame", MaxDiff(u.gen[0], expect, InSoft));
        printf("  no HUD-less, %s\n", layer ? "UI layer" : "UI alpha");
        s.hudI = nullptr;
        u = Run(a, s);
        keep(layer ? "layer only" : "alpha only", u);
        auto both = [](int x, int y) { return InUi(x, y) || InSoft(x, y); };
        Check(MaxDiff(u.gen[0], u.back, both) == 0, "where the opacity is over 0.4: the frame", MaxDiff(u.gen[0], u.back, both));
    }
    printf("  no-warp mask\n");
    {
        Scene s = plain;
        s.noWarpI = [](int) { Image m(W, H); for (uint32_t y = 0; y < H; ++y) for (uint32_t x = 0; x < W; ++x) m.at(x, y)[0] = InUi(x, y); return m; };
        Out u = Run(a, s);
        keep("nowarp", u);
        Check(MaxDiff(u.gen[1], u.back, InUi) == 0, "masked: the frame", MaxDiff(u.gen[1], u.back, InUi));
        auto rest = [](int x, int y) { return !InUi(x, y); };
        Check(MaxDiff(u.gen[1], base.gen[1], rest) == 0, "the rest: as without it", MaxDiff(u.gen[1], base.gen[1], rest));
    }
    printf("  colour subrects (back buffer 800x500 at (80, 60), HUD-less 700x400 at (10, 6))\n");
    {
        Scene s = plain;
        s.w = 800, s.h = 500;
        s.backI = [](int i) { Image b(800, 500, 0.3f); Image u = WithUi(World(W, H, float(i))); for (uint32_t y = 0; y < H; ++y) memcpy(b.at(80, 60 + y), u.at(0, y), W * 16); return b; };
        s.hudI = [](int i) { Image b(700, 400, 0.9f); Image u = World(W, H, float(i)); for (uint32_t y = 0; y < H; ++y) memcpy(b.at(10, 6 + y), u.at(0, y), W * 16); return b; };
        s.extra = [](Params& p, int) {
            p.Set("DLSSG.BackbufferSubrectBaseX", 80), p.Set("DLSSG.BackbufferSubrectBaseY", 60), p.Set("DLSSG.BackbufferSubrectWidth", W), p.Set("DLSSG.BackbufferSubrectHeight", H);
            p.Set("DLSSG.HUDLessSubrectBaseX", 10), p.Set("DLSSG.HUDLessSubrectBaseY", 6), p.Set("DLSSG.HUDLessSubrectWidth", W), p.Set("DLSSG.HUDLessSubrectHeight", H);
        };
        Out u = Run(a, s);
        keep("subrect", u);
        auto around = [](int x, int y) { return x < 80 || y < 60 || x >= 80 + int(W) || y >= 60 + int(H); };
        Check(MaxDiff(u.gen[0], u.back, around) == 0, "around the region: the frame", MaxDiff(u.gen[0], u.back, around));
        const Image in = Cut(u.gen[0], 80, 60, W, H);
        const double q = Psnr(in, o.gen[0]);
        Check(q > 45, "the region: as the whole image, dB", q);
        Check(MaxDiff(in, Cut(u.back, 80, 60, W, H), InUi) == 0, "the UI in it: the frame", 0);
    }
    printf("  depth and motion subrects\n");
    {
        Scene s = plain;
        s.extra = [](Params& p, int) { for (const char* n : {"Depth", "MVecs"}) { const std::string b = std::string("DLSSG.") + n + "Subrect"; p.Set((b + "BaseX").c_str(), 0), p.Set((b + "BaseY").c_str(), 0), p.Set((b + "Width").c_str(), W / 2), p.Set((b + "Height").c_str(), H / 2); } };
        Out u = Run(a, s);  // half-resolution inputs in the corner of a bigger image: vectors in their own pixels would be half; the scale says display
        keep("depth subrect", u);
        Check(u.ok, "evaluates", 1);
    }
    printf("  output subrects, two viewports on one back buffer\n");
    {
        auto half = [](int i) { Image b(800, 500, 0.3f); Image u = World(W, H, float(i)); for (uint32_t y = 0; y < H; ++y) memcpy(b.at(80, 60 + y), u.at(0, y), W * 16); return b; };
        auto rect = [](Params& p, const char* n, uint32_t x, uint32_t y) { const std::string b = std::string("DLSSG.") + n + "Subrect"; p.Set((b + "BaseX").c_str(), x), p.Set((b + "BaseY").c_str(), y), p.Set((b + "Width").c_str(), W), p.Set((b + "Height").c_str(), H); };
        Scene s = plain;
        s.w = 800, s.h = 500, s.backI = half;
        s.extra = [&](Params& p, int) { rect(p, "InputBackbuffer", 80, 60); };
        const Out whole = Run(a, s);
        s.extra = [&](Params& p, int) { rect(p, "InputBackbuffer", 80, 60), rect(p, "OutputInterpolated", 20, 10), p.Set("DLSSG.EvalFlags", 1); };
        const Out moved = Run(a, s);
        Check(MaxDiff(Cut(moved.gen[0], 20, 10, W, H), Cut(whole.gen[0], 80, 60, W, H), All) == 0, "the region, at its place in the output", 0);
        // Only the region is written: the second viewport's leaves the first's.
        Params p;
        TexP back = a.Create(F_RGBA8, 2 * W, H, false), out = a.Create(F_RGBA8, 2 * W, H, true), depth = a.Create(F_R32F, W, H, false), mv = a.Create(F_RG16F, W, H, false);
        Image dI(W, H, 0.9f), mI(W, H);
        for (uint32_t i = 0; i < W * H; ++i) mI.p[i * 4] = -8, mI.p[i * 4 + 1] = -2;
        a.Upload(*depth, Encode(dI, F_R32F)), a.Upload(*mv, Encode(mI, F_RG16F));
        a.Bind(p, "DLSSG.Backbuffer", back.get()), a.Bind(p, "DLSSG.OutputInterpolated", out.get()), a.Bind(p, "DLSSG.Depth", depth.get()), a.Bind(p, "DLSSG.MVecs", mv.get());
        float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        p.Set("DLSSG.ClipToPrevClip", static_cast<void*>(id)), p.Set("DLSSG.CameraViewToClip", static_cast<void*>(id)), p.Set("DLSSG.EvalFlags", 1);
        p.Set("DLSSG.MvecScaleX", 1.0f), p.Set("DLSSG.MvecScaleY", 1.0f), p.Set("DLSSG.CameraNear", 0.1f), p.Set("DLSSG.CameraFar", 100.0f), p.Set("DLSSG.CameraFOV", 1.0f), p.Set("DLSSG.MultiFrameCount", 1), p.Set("DLSSG.MultiFrameIndex", 1);
        NgxHandle* v[2] = {a.CreateFeature(p), a.CreateFeature(p)};
        bool ok = v[0] && v[1];
        for (int i = 0; ok && i < 4; ++i)
        {
            Image b(2 * W, H);
            for (int q = 0; q < 2; ++q) { const Image u = World(W, H, float(i), 1, q * 300.0f, 0); for (uint32_t y = 0; y < H; ++y) memcpy(b.at(q * W, y), u.at(0, y), W * 16); }
            a.Upload(*back, Encode(b, F_RGBA8));
            p.Set("DLSSG.Reset", int(i == 0));
            for (int q = 0; q < 2; ++q) rect(p, "InputBackbuffer", q * W, 0), rect(p, "OutputInterpolated", q * W, 0), ok = a.Evaluate(v[q], p) == kNgxSuccess && ok;
        }
        const Image got = Decode(a.Read(*out), F_RGBA8, 2 * W, H);
        const double q0 = Psnr(Cut(got, 0, 0, W, H), World(W, H, 2.5f), 24), q1 = Psnr(Cut(got, W, 0, W, H), World(W, H, 2.5f, 1, 300, 0), 24);
        Check(ok && q0 > 30 && q1 > 30, "two viewports, both interpolated, worst dB", std::min(q0, q1));
        for (NgxHandle* x : v) if (x) a.Release(x);
    }
    printf("  OutputDisableInterpolation\n");
    {
        Scene s = plain;
        s.extra = [&](Params& p, int) { a.BindWord(p, "DLSSG.OutputDisableInterpolation"); };
        Run(a, s);
        Check(a.ReadWord() == 0, "interpolated: 0", 0);
        s.extra = [&](Params& p, int i) { a.BindWord(p, "DLSSG.OutputDisableInterpolation"), p.Set("DLSSG.NotRenderingGameFrames", int(i == 3)); };
        Run(a, s);
        Check(a.ReadWord() == 1, "not rendering: 1", 1);
        s.extra = [&](Params& p, int) { a.BindWord(p, "DLSSG.OutputDisableInterpolation"), p.Set("DLSSG.MenuDetectionEnabled", 1); };
        s.uiI = [](int) { return Image(W, H, 1); }, s.frames = 12;
        Run(a, s);
        Check(a.ReadWord() == 1, "a menu: 1", 1);
        s = plain;
        s.extra = [&](Params& p, int) { a.BindWord(p, "DLSSG.OutputDisableInterpolation"), p.Set("DLSSG.Depth", static_cast<void*>(nullptr)); };
        Run(a, s);
        Check(a.ReadWord() == 1, "nothing to interpolate from: 1", 1);
    }
    printf("  _C evaluate, video memory estimate\n");
    {
        a.evaluateC = true;
        Check(Run(a, plain).raw == base.raw, "EvaluateFeature_C: the same frames", 1);
        a.evaluateC = false;
        Params p;
        a.Fn<NgxResult (*)(NgxParameter*)>("NVSDK_NGX_D3D12_PopulateParameters_Impl")(&p);
        void* cb = nullptr;
        size_t bytes = 0;
        p.Get("DLSSG.EstimateVRAMCallback", &cb);
        if (cb) reinterpret_cast<NgxResult (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t*)>(cb)(W, H, W, H, 28, 28, 41, 34, 28, &bytes);
        Check(bytes > (100u << 20) && bytes < (2048u << 20), "the estimate, MB", double(bytes) / 1048576);
    }
    printf("  frame ids, inverted axes, camera motion, dynamic resolution\n");
    {
        Scene s = plain;
        s.extra = [](Params& p, int i) { p.Set("DLSSG.BackbufferFrameID", 100ull + i); };
        Check(Run(a, s).raw == base.raw, "ids that follow: as without", 0);
        s.extra = [](Params& p, int i) { p.Set("DLSSG.BackbufferFrameID", 100ull + i + (i == 3)); };
        Out u = Run(a, s);
        Check(u.ok && MaxDiff(u.gen[0], u.back, All) == 0, "a frame skipped: the frame", 0);
        s = plain;
        s.extra = [](Params& p, int) { p.Set("DLSSG.MvecScaleX", -1.0f), p.Set("DLSSG.MvecScaleY", -1.0f), p.Set("DLSSG.InvertXAxis", 1), p.Set("DLSSG.InvertYAxis", 1); };
        Check(Run(a, s).raw == base.raw, "axes inverted twice: as without", 0);
        // The vectors hold none of the camera's motion: clipToPrevClip moves by (-8, -2) pixels.
        s = plain;
        s.mvOffset[0] = 8, s.mvOffset[1] = 2;
        static float clip[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -16.0f / W, 4.0f / H, 0, 1};
        s.extra = [](Params& p, int) { p.Set("DLSSG.CameraMotionIncluded", 0), p.Set("DLSSG.ClipToPrevClip", static_cast<void*>(clip)); };
        u = Run(a, s);
        keep("camera motion", u);
        double q = Psnr(u.gen[0], truth(3, 0, 2), 24);
        Check(q > 30, "camera motion added, dB against the truth", q);
        s = plain;
        s.extra = [](Params& p, int i) {  // another region every frame; vectors are in display pixels: the scale undoes the region's
            const uint32_t w = W - 40 * i, h = H - 20 * i;
            for (const char* n : {"Depth", "MVecs"}) { const std::string b = std::string("DLSSG.") + n + "Subrect"; p.Set((b + "BaseX").c_str(), 0), p.Set((b + "BaseY").c_str(), 0), p.Set((b + "Width").c_str(), w), p.Set((b + "Height").c_str(), h); }
            p.Set("DLSSG.DynamicResolution", 1), p.Set("DLSSG.MvecScaleX", float(w) / W), p.Set("DLSSG.MvecScaleY", float(h) / H);
        };
        u = Run(a, s);
        keep("dynamic resolution", u);
        q = Psnr(u.gen[0], truth(3, 0, 2), 24);
        Check(u.ok && q > 30, "dynamic resolution, history kept, dB against the truth", q);
        s.extra = [&](Params& p, int) { p.Set("DLSSG.DynamicResolution", 1), p.Set("DLSSG.InternalWidth", 400u), p.Set("DLSSG.InternalHeight", 240u); };
        u = Run(a, s);
        keep("dynamic internal", u);
        q = Psnr(u.gen[0], truth(3, 0, 2), 24);
        Check(u.ok && q > 30, "a given internal size, dB", q);
        // Jitter in the region's pixels, the vectors holding it.
        Scene j = plain;
        j.extra = [](Params& p, int i) {
            for (const char* n : {"Depth", "MVecs"}) { const std::string b = std::string("DLSSG.") + n + "Subrect"; p.Set((b + "BaseX").c_str(), 0), p.Set((b + "BaseY").c_str(), 0), p.Set((b + "Width").c_str(), W / 2), p.Set((b + "Height").c_str(), H / 2); }
            p.Set("DLSSG.DynamicResolution", 1), p.Set("DLSSG.InternalWidth", 400u), p.Set("DLSSG.InternalHeight", 240u), p.Set("DLSSG.MvecScaleX", 0.5f), p.Set("DLSSG.MvecScaleY", 0.5f);
        };
        const Out clean = Run(a, j);
        const auto base2 = j.extra;
        j.extra = [base2](Params& p, int i) { base2(p, i), p.Set("DLSSG.JitterOffsetX", 2.0f * float(i)), p.Set("DLSSG.JitterOffsetY", -1.5f * float(i)), p.Set("DLSSG.MvecJittered", 1); };
        j.mvOffset[0] = 4.0f, j.mvOffset[1] = -3.0f;
        u = Run(a, j);
        keep("dynamic jitter", u);
        q = Psnr(u.gen[0], clean.gen[0], 24);
        Check(u.ok && q > 90, "jitter with it: as without, dB", q);
    }
    printf("  jittered motion vectors\n");
    {
        // The camera's sample positions move by (0.4, -0.3) a frame and the vectors contain it: the same frames as without.
        Scene s = plain;
        s.extra = [](Params& p, int i) {
            p.Set("DLSSG.JitterOffsetX", 0.4f * float(i)), p.Set("DLSSG.JitterOffsetY", -0.3f * float(i)), p.Set("DLSSG.MvecJittered", 1);
        };
        s.mvOffset[0] = 0.4f, s.mvOffset[1] = -0.3f;
        Out u = Run(a, s);
        keep("jitter", u);
        double q = Psnr(u.gen[0], base.gen[0], 24);
        Check(q > 50, "taken out: as without, dB", q);
    }
    printf("  HUD-less in another format (RGB10A2, BGRA8)\n");
    for (Fmt f : {F_RGB10, F_BGRA8})
    {
        Scene s = d;
        s.hud = f;
        Out u = Run(a, s);
        keep(f == F_RGB10 ? "hud rgb10" : "hud bgra8", u);
        Check(MaxDiff(u.gen[0], u.back, InUi) == 0, "opaque UI is the frame exactly", 0);
        const double q = Psnr(u.gen[0], o.gen[0]);
        Check(q > 45, "as in the same format, dB", q);
    }
    printf("  output formats\n");
    for (Fmt f : {F_RGBA8S, F_BGRA8S})
    {
        Scene s = d;
        s.back = s.hud = f;
        Out u = Run(a, s);
        keep(f == F_RGBA8S ? "rgba8 srgb" : "bgra8 srgb", u);
        const double m = std::max(MaxDiff(u.gen[0], o.gen[0], All), MaxDiff(u.gen[1], o.gen[1], All));
        Check(m <= code, "sRGB images: the same stored values as UNORM", m);
        Scene st = s;  // the staged path too
        st.hud = f == F_RGBA8S ? F_BGRA8 : F_RGBA8;
        u = Run(a, st);
        Check(Psnr(u.gen[0], o.gen[0]) > 45, "sRGB back buffer, UNORM HUD-less of another order, dB", Psnr(u.gen[0], o.gen[0]));
    }
    for (Fmt f : {F_BGRA8, F_RGB10})
    {
        Scene s = d;
        s.back = s.hud = f;
        Out u = Run(a, s);
        keep(f == F_BGRA8 ? "bgra8" : "rgb10", u);
        Check(MaxDiff(u.gen[0], u.back, InUi) == 0, "opaque UI is the frame exactly", 0);
        const double q = Psnr(u.gen[0], o.gen[0], 0);
        Check(q > 40, "as in RGBA8, dB", q);
    }
    printf("  a float (scRGB) back buffer\n");
    {
        Scene s = plain;
        s.back = F_RGBA16F, s.hdr = true;
        const Out u = Run(a, s);
        Check(u.ok && MaxDiff(u.gen[0], u.back, All) == 0, "not supported: the frame, success", 0);
    }
    printf("  HDR10 UI (RGB10A2, PQ values)\n");
    {
        Scene s = d;
        s.back = s.hud = F_RGB10, s.hdr = true;
        const Out u = Run(a, s);
        Check(MaxDiff(u.gen[0], u.back, InUi) == 0, "opaque UI is the frame exactly", 0);
        Scene w = s;
        w.hudI = nullptr, w.backI = world;
        const Out v = Run(a, w);
        Check(MaxDiff(u.gen[0], v.gen[0], outside) == 0, "outside the UI: as without it", MaxDiff(u.gen[0], v.gen[0], outside));
    }
    printf("  HDR quality: a moving object over a bright, fine background (PSNR of the PQ signal)\n");
    {
        auto pq = [](float nits) { const float y = powf(std::clamp(nits / 10000.0f, 0.0f, 1.0f), 0.1593017578125f); return powf((0.8359375f + 18.8515625f * y) / (1 + 18.6875f * y), 78.84375f); };
        // Nits 4 to 1000: everything over 80 is past scRGB 1.0. A square moves against the background, so there are holes to fill.
        auto nits = [](float t) {
            Image o(W, H, 1);
            for (uint32_t y = 0; y < H; ++y)
                for (uint32_t x = 0; x < W; ++x)
                {
                    const bool in = x >= 250 - 10 * t && x < 350 - 10 * t && y >= 120 && y < 220;
                    for (int k = 0; k < 3; ++k) o.at(x, y)[k] = in ? 4 + 100 * Smooth((x + 10 * t) / 7.0f + k * 31, y / 7.0f) : 200 + 800 * Smooth((x - 8 * t) / 13.0f + k * 91, (y - 2 * t) / 13.0f);
                }
            return o;
        };
        auto map = [](Image o, auto f) { for (size_t i = 0; i < o.p.size(); ++i) if (i % 4 != 3) o.p[i] = f(o.p[i]); return o; };
        const Image truth1 = map(nits(2 + 1 / 3.0f), pq);
        Scene s = plain;
        s.hdr = true, s.back = F_RGB10;
        s.backI = [&](int i) { return map(nits(float(i)), pq); };
        const Out u = Run(a, s);
        keep("rgb10 pq quality", u);
        const double q = Psnr(u.gen[0], truth1, 24);
        Check(q > 40, "RGB10A2 PQ, dB", q);
    }
    printf("  distortion field\n");
    {
        Scene s = d;
        s.fieldI = [](int) { return Image(160, 90); };
        Out u = Run(a, s);
        keep("distortion zero", u);
        double q = Psnr(u.gen[0], o.gen[0]);
        Check(q > 45, "a zero field: as without, dB", q);
        Check(MaxDiff(u.gen[0], u.back, InUi) == 0, "UI is the frame", 0);
        // distorted = undistorted shifted by 16 pixels: RG (distorted to undistorted) +16, BA -16, in motion-vector pixels.
        s = plain;
        s.backI = [](int i) { return World(W, H, float(i), 1, 16, 0); };
        s.fieldI = [](int) { Image f(160, 90); for (size_t i = 0; i < f.p.size() / 4; ++i) f.p[i * 4] = 16, f.p[i * 4 + 2] = -16; return f; };
        u = Run(a, s);
        keep("distortion shift", u);
        const Image want = World(W, H, 2 + 1 / 3.0f, 1, 16, 0);
        q = Psnr(u.gen[0], want, 40);
        Check(q > 30, "a shifting field: the distorted frame between, dB against the truth", q);
        s.gen = 4;
        u = Run(a, s);
        keep("distortion 5x", u);
        q = 1e9;
        for (uint32_t k = 0; k < 4; ++k) q = std::min(q, Psnr(u.gen[k], World(W, H, 2 + (k + 1) / 5.0f, 1, 16, 0), 40));
        Check(q > 30, "5x with it, worst dB", q);
    }
    printf("  menu detection\n");
    {
        Scene s = plain;
        s.frames = 12;
        s.uiI = [](int) { return Image(W, H, 1); };
        s.extra = [](Params& p, int) { p.Set("DLSSG.MenuDetectionEnabled", 1); };
        Out u = Run(a, s);
        keep("menu", u);
        Check(MaxDiff(u.gen[0], u.back, All) == 0, "a menu over the screen: the frame", 0);
        s.uiI = [](int i) { return Image(W, H, i < 6 ? 1.0f : 0.0f); };
        s.frames = 20;
        u = Run(a, s);
        Check(Psnr(u.gen[0], truth(19, 0, 2), 24) > 30, "menu gone: interpolated again, dB", Psnr(u.gen[0], truth(19, 0, 2), 24));
    }
    printf("  reset, not rendering\n");
    {
        Scene s = d;
        s.extra = [](Params& p, int i) { if (i == 3) p.Set("DLSSG.Reset", 1); };
        Out u = Run(a, s);
        Check(MaxDiff(u.gen[0], u.back, All) == 0 && MaxDiff(u.gen[1], u.back, All) == 0, "reset with HUD-less: the frame with its UI", 0);
        s = plain;
        s.extra = [](Params& p, int i) { p.Set("DLSSG.NotRenderingGameFrames", int(i == 3)); };
        u = Run(a, s);
        Check(MaxDiff(u.gen[0], u.back, All) == 0, "not rendering game frames: the frame", 0);
    }
    printf("  protocol\n");
    {
        Scene s = plain;
        s.extra = [](Params& p, int) { p.Set("DLSSG.Depth", static_cast<void*>(nullptr)); };
        Out u = Run(a, s);
        Check(u.ok && MaxDiff(u.gen[0], u.back, All) == 0, "no depth: the frame, success", 0);
        s.extra = [](Params& p, int) { p.Set("DLSSG.Backbuffer", static_cast<void*>(nullptr)); };
        u = Run(a, s);
        Check(!u.ok, "no back buffer: an error", 0);
        s.extra = [](Params& p, int) { p.Set("DLSSG.MultiFrameIndex", 2); };
        u = Run(a, s);
        Check(u.ok && MaxDiff(u.gen[0], u.back, All) == 0, "index 2 with no batch: the frame", 0);
        s.extra = [](Params& p, int) { int k = 0; p.Get("DLSSG.MultiFrameIndex", &k); p.Set("DLSSG.MultiFrameCount", 7); if (k > 1) p.Set("DLSSG.MultiFrameIndex", 7); };
        u = Run(a, s);
        Check(u.ok && MaxDiff(u.gen[1], u.back, All) == 0, "index past 5 of 7: the frame", 0);
        s.backI = [](int) { return World(W, H, 3); };
        s.extra = [](Params& p, int) { p.Set("DLSSG.MvecScaleX", 0.0f), p.Set("DLSSG.MvecScaleY", 0.0f); };
        u = Run(a, s);
        Check(Psnr(u.gen[0], u.back) > 50, "identical frames, no motion: the frame, dB", Psnr(u.gen[0], u.back));
        // One feature through a change of count, size and format, its history restarted each time.
        Params p;
        NgxHandle* h = a.CreateFeature(p);
        bool ok = h != nullptr;
        for (int i = 0; ok && i < 6; ++i)
        {
            Scene c = plain;
            c.gen = i == 1 ? 5 : 1 + i % 3;
            c.back = i & 1 ? F_RGB10 : F_RGBA8;
            if (i >= 3) c.w = 1280, c.h = 720, c.backI = [](int f) { Image b(1280, 720, 1); Image wd = World(W, H, float(f)); for (uint32_t y = 0; y < 720; ++y) for (uint32_t x = 0; x < 1280; ++x) memcpy(b.at(x, y), wd.at(x / 2, y / 2), 16); return b; };
            c.frames = 3;
            ok = Run(a, c, h).ok;
        }
        Check(ok, "one feature across counts, formats and sizes", 1);
        NgxHandle* h2 = a.CreateFeature(p);
        Scene c = plain;
        const Out x = Run(a, c, h), y = Run(a, c, h2);
        Check(x.raw == y.raw && x.raw == base.raw, "two features: the same frames as one", 1);
        Check(a.Release(h) == kNgxSuccess && a.Release(h2) == kNgxSuccess, "released", 1);
        Check(a.Release(h) != kNgxSuccess, "released twice: an error", 1);
        Check(a.Evaluate(h, p) != kNgxSuccess, "evaluate after release: an error", 1);
        NgxHandle* h3 = a.CreateFeature(p);
        Check(Run(a, c, h3).raw == base.raw, "created again: the same frames", 1);
        a.Release(h3);
        // Released on one thread while another evaluates.
        NgxHandle *h4 = a.CreateFeature(p), *h5 = a.CreateFeature(p);
        Out o4;
        std::thread t([&] { Scene l = plain; l.frames = 8; o4 = Run(a, l, h4); });
        Sleep(30);
        const bool released = a.Release(h5) == kNgxSuccess;
        t.join();
        Check(released && o4.ok, "a release while another feature evaluates", 1);
        a.Release(h4);
        // The device goes and a new one comes (perhaps at its address).
        Check(a.Recreate(), "device destroyed and made again", 1);
        NgxHandle* h6 = a.CreateFeature(p);
        Check(h6 && Run(a, c, h6).raw == base.raw, "on the new device: the same frames", 1);
        if (h6) a.Release(h6);
    }
}
static void Soak(Api& a, uint32_t n)
{
    Params p;
    NgxHandle* h = a.CreateFeature(p);
    Scene s;
    s.backI = [](int i) { return World(W, H, float(i % 4)); };
    s.frames = 4, s.gen = 2;
    uint64_t v0 = 0;
    DWORD h0 = 0, h1 = 0;
    bool ok = true;
    for (uint32_t i = 0; i < n / 4 && ok; ++i)
    {
        ok = Run(a, s, h).ok;
        if (i == 20) v0 = a.VideoMemory(), GetProcessHandleCount(GetCurrentProcess(), &h0);
    }
    GetProcessHandleCount(GetCurrentProcess(), &h1);
    const double mb = (double(a.VideoMemory()) - double(v0)) / 1048576;
    Check(ok, "soak: every evaluate succeeds", n);
    Check(fabs(mb) < 16, "video memory flat after the start, MB", mb);
    Check(int(h1) - int(h0) < 20, "handles flat", double(int(h1) - int(h0)));
    a.Release(h);
}

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) return puts("ngx_harness DLL [--minimal] [--soak N] [d3d12|vulkan]"), 2;
    if (!getenv("NFRUX_ASYNC_TEST")) SetEnvironmentVariableA("NFRUX_SYNC_CREATE", "1");  // every frame counted on; else the worker thread's path
    HMODULE dll = LoadLibraryExA(argv[1], nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!dll) return printf("cannot load %s\n", argv[1]), 2;
    bool minimal = false;
    uint32_t soak = 0;
    std::vector<std::string> which;
    for (int i = 2; i < argc; ++i)
    {
        const std::string s = argv[i];
        if (s == "--minimal") minimal = true;
        else if (s == "--soak") soak = uint32_t(atoi(argv[++i]));
        else which.push_back(s);
    }
    if (which.empty()) which = {"d3d12", "vulkan"};
    std::vector<std::unique_ptr<Api>> apis;  // all alive together
    std::vector<Results> results;
    for (auto& w : which)
    {
        printf("%s\n", w.c_str());
        bool ok;
        if (w == "vulkan") { auto a = std::make_unique<ApiV>(); a->dll = dll, a->minimal = minimal, ok = a->Init(), apis.push_back(std::move(a)); }
        else { auto a = std::make_unique<Api12>(); a->dll = dll, ok = a->Init(), apis.push_back(std::move(a)); }
        if (!ok) return puts("  device failed"), 2;
        results.emplace_back();
        if (soak) Soak(*apis.back(), soak);
        else Scenes(*apis.back(), results.back());
    }
    if (results.size() == 2 && !soak)
    {
        printf("both APIs at once, a thread each\n");
        Scene s;
        s.backI = [](int i) { return World(W, H, float(i)); }, s.frames = 12;
        Out o[2], alone[2];
        for (int i = 0; i < 2; ++i) alone[i] = Run(*apis[i], s);
        std::thread t([&] { o[0] = Run(*apis[0], s); });
        o[1] = Run(*apis[1], s);
        t.join();
        Check(o[0].ok && o[1].ok && o[0].raw == alone[0].raw && o[1].raw == alone[1].raw, "each as when alone", 1);
        printf("between the APIs\n");
        size_t same = 0, pixels = 0, over1 = 0, over3 = 0;
        for (auto& [n, v] : results[0])
        {
            const Bytes& o = results[1][n];
            if (o.size() != v.size()) { printf("    FAIL  %s: sizes\n", n.c_str()), ++g_bad; continue; }
            same += o == v;
            if (o == v) { pixels += v.size(); continue; }
            size_t d1 = 0, d3 = 0;
            double e = 0;
            const bool fp16 = n.rfind("rgba16f", 0) == 0 || n.rfind("scrgb", 0) == 0;
            if (n.rfind("rgb10", 0) == 0)  // packed: by 10-bit code, on the 8-bit scale
                for (size_t i = 0; i + 3 < v.size(); i += 4)
                {
                    uint32_t x, y;
                    memcpy(&x, &v[i], 4), memcpy(&y, &o[i], 4);
                    for (int k = 0; k < 30; k += 10) { const int c = abs(int(x >> k & 1023) - int(y >> k & 1023)); d1 += c > 1, d3 += c > 3, e += double(c) * c / 16; }
                }
            else
            for (size_t i = 0; i < v.size(); ++i) { const int x = abs(int(v[i]) - int(o[i])); d1 += x > 1, d3 += x > 3, e += double(x) * x; }
            printf("    %s: %zu of %zu bytes off by more than 1, %.1f dB%s\n", n.c_str(), d1, v.size(), 10 * log10(65025.0 / std::max(1e-9, e / v.size())), fp16 ? " (fp16 bytes)" : "");
            if (!fp16 && 10 * log10(65025.0 / std::max(1e-9, e / v.size())) < 60) ++g_bad;
            pixels += v.size(), over1 += d1, over3 += d3;
        }
        Check(over1 * 1000 <= pixels, "share of bytes off by more than 1", double(over1) / double(pixels));
        printf("    %zu of %zu outputs byte-identical\n", same, results[0].size());
    }
    for (auto& a : apis)
    {
        a->Shutdown();
        if (a->messages) printf("%s: %u validation messages\n", a->Name(), a->messages), ++g_bad;
    }
    printf("%d checks, %d failed\n", g_checks, g_bad);
    return g_bad != 0;
}
