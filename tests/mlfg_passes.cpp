// Runs NFRUX's FSR Frame Generation 4 passes (kMlfg*, core/shaders/mlfg) in isolation on D3D12 or Vulkan, with raw
// files in and out, for tests/mlfg_passes.py (which compares them with the PyTorch transcription and AMD's captures).
//   mlfg_passes d3d12|vulkan JOBFILE
// JOBFILE, blocks of:
//   pass NAME                      a kMlfg shader (NAME or NAMEInv)
//   cb FILE                        the constants (binding 0)
//   tex B FMT W H FILE             read-only texture at binding B, texels from FILE (tightly packed)
//   utex B FMT W H FILE|- OUT      read-write texture: initial texels (- = zeros), read back into OUT
//   buf B BYTES FILE               read-only buffer
//   ubuf B BYTES FILE|- OUT        read-write buffer
//   run GX GY                      dispatch GX x GY groups, read back, end of the block
// FMT: rgba8 rg16f r32f r32ui rgba16f rgb10a2. Headless: no window, no swap chain. SPDX-License-Identifier: MIT
#include "nfrux_shaders.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <vulkan/vulkan.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace nfrux;
using Microsoft::WRL::ComPtr;

#define MLFG_SHADERS(X) X(kMlfgDilate) X(kMlfgSplatKey) X(kMlfgSplatFlow) X(kMlfgForwardMv) X(kMlfgNetInput) X(kMlfgDownsample) \
    X(kMlfgRefine) X(kMlfgFlowOut) X(kMlfgStage2Input) X(kMlfgSoftmax5) X(kMlfgCompositeRgba8) X(kMlfgCompositeRgb10a2) X(kMlfgCompositeRgba16f)
static const Shader* FindShader(const std::string& n)
{
#define X(s) if (n == #s) return &shaders::s; if (n == #s "Inv") return &shaders::s##Inv;
    MLFG_SHADERS(X)
#undef X
    return nullptr;
}

struct Fmt { const char* name; DXGI_FORMAT dx; VkFormat vk; uint32_t bytes; };
static const Fmt kFmts[] = {{"rgba8", DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, 4}, {"rg16f", DXGI_FORMAT_R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT, 4},
                            {"r32f", DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, 4}, {"r32ui", DXGI_FORMAT_R32_UINT, VK_FORMAT_R32_UINT, 4},
                            {"rgba16f", DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8},
                            {"rgb10a2", DXGI_FORMAT_R10G10B10A2_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4}};

struct Res
{
    uint32_t             binding = 0;
    bool                 texture = false, writable = false;
    const Fmt*           fmt     = nullptr;
    uint32_t             w = 0, h = 0;
    uint64_t             bytes = 0;
    std::vector<uint8_t> data;
    std::string          out;
};
struct Job
{
    const Shader*        shader = nullptr;
    std::string          name;
    std::vector<uint8_t> cb;
    std::vector<Res>     res;
    uint32_t             gx = 0, gy = 0;
};

static bool ReadFile(const std::string& p, std::vector<uint8_t>& d, uint64_t want)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) return printf("cannot read %s\n", p.c_str()), false;
    d.assign(std::istreambuf_iterator<char>(f), {});
    if (want && d.size() != want) return printf("%s: %zu bytes, want %llu\n", p.c_str(), d.size(), (unsigned long long)want), false;
    return true;
}
static bool WriteFile(const std::string& p, const void* d, size_t n)
{
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(d), std::streamsize(n));
    return bool(f);
}

static bool Parse(const char* path, std::vector<Job>& jobs)
{
    std::ifstream f(path);
    if (!f) return printf("cannot read %s\n", path), false;
    std::string line;
    Job j;
    while (std::getline(f, line))
    {
        std::istringstream s(line);
        std::string k;
        if (!(s >> k) || k[0] == '#') continue;
        if (k == "pass")
        {
            s >> j.name;
            if (!(j.shader = FindShader(j.name))) return printf("unknown pass %s\n", j.name.c_str()), false;
        }
        else if (k == "cb") { std::string p; s >> p; if (!ReadFile(p, j.cb, 0)) return false; }
        else if (k == "tex" || k == "utex" || k == "buf" || k == "ubuf")
        {
            Res r;
            r.texture = k == "tex" || k == "utex";
            r.writable = k[0] == 'u';
            s >> r.binding;
            if (r.texture)
            {
                std::string fn;
                s >> fn >> r.w >> r.h;
                for (const Fmt& m : kFmts) if (fn == m.name) r.fmt = &m;
                if (!r.fmt) return printf("unknown format %s\n", fn.c_str()), false;
                r.bytes = uint64_t(r.w) * r.h * r.fmt->bytes;
            }
            else
                s >> r.bytes;
            std::string src;
            s >> src;
            if (src == "-") r.data.assign(size_t(r.bytes), 0);
            else if (!ReadFile(src, r.data, r.bytes)) return false;
            if (r.writable) s >> r.out;
            j.res.push_back(std::move(r));
        }
        else if (k == "run")
        {
            s >> j.gx >> j.gy;
            jobs.push_back(std::move(j));
            j = Job();
        }
        else return printf("bad line: %s\n", line.c_str()), false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ D3D12
#define DXC(x) do { HRESULT h_ = (x); if (FAILED(h_)) { printf("%s = 0x%08lx\n", #x, h_); return false; } } while (0)
static bool RunD3D12(std::vector<Job>& jobs)
{
    ComPtr<ID3D12Device> dev;
    DXC(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    ComPtr<ID3D12CommandQueue> q;
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_COMPUTE};
    DXC(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    ComPtr<ID3D12CommandAllocator> al;
    DXC(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&al)));
    ComPtr<ID3D12GraphicsCommandList> cl;
    DXC(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, al.Get(), nullptr, IID_PPV_ARGS(&cl)));
    ComPtr<ID3D12Fence> fence;
    DXC(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    uint64_t fv = 0;
    auto buffer = [&](uint64_t size, D3D12_HEAP_TYPE heap, bool uav, D3D12_RESOURCE_STATES st) {
        ComPtr<ID3D12Resource> r;
        D3D12_HEAP_PROPERTIES hp = {heap};
        D3D12_RESOURCE_DESC d = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, size < 256 ? 256 : size, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                 uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, IID_PPV_ARGS(&r));
        return r;
    };
    auto barrier = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
        D3D12_RESOURCE_BARRIER x = {};
        x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
        cl->ResourceBarrier(1, &x);
    };
    for (Job& j : jobs)
    {
        al->Reset(), cl->Reset(al.Get(), nullptr);
        const Shader& sh = *j.shader;
        // Root signature: one table, a range per binding (as rhi_nri's layout: space 0).
        std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
        for (uint32_t i = 0; i < sh.bindingCount; ++i)
        {
            const ShaderBinding& b = sh.bindings[i];
            D3D12_DESCRIPTOR_RANGE r = {};
            r.RangeType = b.kind == BindKind::Cbv ? D3D12_DESCRIPTOR_RANGE_TYPE_CBV
                        : (b.kind == BindKind::SrvTexture || b.kind == BindKind::SrvBuffer) ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            r.NumDescriptors = 1, r.BaseShaderRegister = b.binding, r.OffsetInDescriptorsFromTableStart = i;
            ranges.push_back(r);
        }
        D3D12_ROOT_PARAMETER rp = {};
        rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, rp.DescriptorTable = {uint32_t(ranges.size()), ranges.data()};
        D3D12_ROOT_SIGNATURE_DESC rd = {1, &rp};
        ComPtr<ID3DBlob> blob, err;
        if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) return printf("root signature: %s\n", err ? (char*)err->GetBufferPointer() : "?"), false;
        ComPtr<ID3D12RootSignature> rs;
        DXC(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {rs.Get(), {sh.dxil, sh.dxilBytes}};
        ComPtr<ID3D12PipelineState> pso;
        DXC(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
        ComPtr<ID3D12DescriptorHeap> heap;
        D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, sh.bindingCount, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        DXC(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
        const uint32_t inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        std::vector<ComPtr<ID3D12Resource>> keep, gpu(j.res.size()), back(j.res.size());
        std::vector<uint32_t> pitch(j.res.size());
        ComPtr<ID3D12Resource> cbr = buffer((j.cb.size() + 255) & ~255ull, D3D12_HEAP_TYPE_UPLOAD, false, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* m;
        cbr->Map(0, nullptr, &m), memcpy(m, j.cb.data(), j.cb.size()), cbr->Unmap(0, nullptr);
        for (size_t i = 0; i < j.res.size(); ++i)
        {
            Res& r = j.res[i];
            const D3D12_RESOURCE_STATES use = r.writable ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            if (r.texture)
            {
                pitch[i] = (r.w * r.fmt->bytes + 255) & ~255u;
                D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
                D3D12_RESOURCE_DESC d = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, r.w, r.h, 1, 1, r.fmt->dx, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                         r.writable ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
                DXC(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gpu[i])));
                ComPtr<ID3D12Resource> up = buffer(uint64_t(pitch[i]) * r.h, D3D12_HEAP_TYPE_UPLOAD, false, D3D12_RESOURCE_STATE_GENERIC_READ);
                uint8_t* p;
                up->Map(0, nullptr, reinterpret_cast<void**>(&p));
                for (uint32_t y = 0; y < r.h; ++y) memcpy(p + size_t(y) * pitch[i], r.data.data() + size_t(y) * r.w * r.fmt->bytes, r.w * r.fmt->bytes);
                up->Unmap(0, nullptr);
                D3D12_TEXTURE_COPY_LOCATION dst = {gpu[i].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, src = {up.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
                src.PlacedFootprint.Footprint = {r.fmt->dx, r.w, r.h, 1, pitch[i]};
                cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                keep.push_back(up);
            }
            else
            {
                gpu[i] = buffer(r.bytes, D3D12_HEAP_TYPE_DEFAULT, true, D3D12_RESOURCE_STATE_COPY_DEST);
                ComPtr<ID3D12Resource> up = buffer(r.bytes, D3D12_HEAP_TYPE_UPLOAD, false, D3D12_RESOURCE_STATE_GENERIC_READ);
                up->Map(0, nullptr, &m), memcpy(m, r.data.data(), r.data.size()), up->Unmap(0, nullptr);
                cl->CopyBufferRegion(gpu[i].Get(), 0, up.Get(), 0, r.bytes);
                keep.push_back(up);
            }
            barrier(gpu[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST, use);
        }
        for (uint32_t k = 0; k < sh.bindingCount; ++k)
        {
            D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += size_t(k) * inc;
            const ShaderBinding& b = sh.bindings[k];
            if (b.kind == BindKind::Cbv)
            {
                D3D12_CONSTANT_BUFFER_VIEW_DESC c = {cbr->GetGPUVirtualAddress(), uint32_t((j.cb.size() + 255) & ~255ull)};
                dev->CreateConstantBufferView(&c, h);
                continue;
            }
            size_t i = 0;
            while (i < j.res.size() && j.res[i].binding != b.binding) ++i;
            if (i == j.res.size()) return printf("%s: binding %u not given\n", j.name.c_str(), b.binding), false;
            const Res& r = j.res[i];
            if (b.kind == BindKind::SrvTexture)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC v = {r.fmt->dx, D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
                v.Texture2D.MipLevels = 1;
                dev->CreateShaderResourceView(gpu[i].Get(), &v, h);
            }
            else if (b.kind == BindKind::UavTexture)
            {
                D3D12_UNORDERED_ACCESS_VIEW_DESC v = {r.fmt->dx, D3D12_UAV_DIMENSION_TEXTURE2D};
                dev->CreateUnorderedAccessView(gpu[i].Get(), nullptr, &v, h);
            }
            else if (b.kind == BindKind::SrvBuffer)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC v = {DXGI_FORMAT_R32_TYPELESS, D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
                v.Buffer.NumElements = uint32_t(r.bytes / 4), v.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
                dev->CreateShaderResourceView(gpu[i].Get(), &v, h);
            }
            else
            {
                D3D12_UNORDERED_ACCESS_VIEW_DESC v = {DXGI_FORMAT_R32_TYPELESS, D3D12_UAV_DIMENSION_BUFFER};
                v.Buffer.NumElements = uint32_t(r.bytes / 4), v.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                dev->CreateUnorderedAccessView(gpu[i].Get(), nullptr, &v, h);
            }
        }
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        cl->SetDescriptorHeaps(1, heaps);
        cl->SetComputeRootSignature(rs.Get());
        cl->SetPipelineState(pso.Get());
        cl->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        cl->Dispatch(j.gx, j.gy, 1);
        for (size_t i = 0; i < j.res.size(); ++i)
        {
            Res& r = j.res[i];
            if (!r.writable) continue;
            barrier(gpu[i].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            if (r.texture)
            {
                back[i] = buffer(uint64_t(pitch[i]) * r.h, D3D12_HEAP_TYPE_READBACK, false, D3D12_RESOURCE_STATE_COPY_DEST);
                D3D12_TEXTURE_COPY_LOCATION src = {gpu[i].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, dst = {back[i].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
                dst.PlacedFootprint.Footprint = {r.fmt->dx, r.w, r.h, 1, pitch[i]};
                cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            else
            {
                back[i] = buffer(r.bytes, D3D12_HEAP_TYPE_READBACK, false, D3D12_RESOURCE_STATE_COPY_DEST);
                cl->CopyBufferRegion(back[i].Get(), 0, gpu[i].Get(), 0, r.bytes);
            }
        }
        DXC(cl->Close());
        ID3D12CommandList* l = cl.Get();
        q->ExecuteCommandLists(1, &l), q->Signal(fence.Get(), ++fv);
        HANDLE ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        fence->SetEventOnCompletion(fv, ev), WaitForSingleObject(ev, 20000), CloseHandle(ev);
        if (fence->GetCompletedValue() < fv) return printf("%s: GPU timeout / device removed (0x%08lx)\n", j.name.c_str(), dev->GetDeviceRemovedReason()), false;
        for (size_t i = 0; i < j.res.size(); ++i)
        {
            Res& r = j.res[i];
            if (!r.writable) continue;
            uint8_t* p;
            back[i]->Map(0, nullptr, reinterpret_cast<void**>(&p));
            std::vector<uint8_t> o(size_t(r.bytes));
            if (r.texture)
                for (uint32_t y = 0; y < r.h; ++y) memcpy(o.data() + size_t(y) * r.w * r.fmt->bytes, p + size_t(y) * pitch[i], r.w * r.fmt->bytes);
            else
                memcpy(o.data(), p, o.size());
            back[i]->Unmap(0, nullptr);
            if (!WriteFile(r.out, o.data(), o.size())) return printf("cannot write %s\n", r.out.c_str()), false;
        }
        printf("d3d12 %s ok\n", j.name.c_str());
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ Vulkan
#define VKC(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("%s = %d\n", #x, r_); return false; } } while (0)
static bool RunVulkan(std::vector<Job>& jobs)
{
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_2;
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
    if (!n) return printf("no Vulkan device\n"), false;
    VkPhysicalDevice pd = pds[0];
    for (VkPhysicalDevice p : pds)
    {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = p; break; }
    }
    VkPhysicalDeviceFeatures feats;
    vkGetPhysicalDeviceFeatures(pd, &feats);
    feats.robustBufferAccess = VK_FALSE;
    uint32_t qn = 0, family = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    for (uint32_t i = 0; i < qn; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
    const float prio = 1;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, family, 1, &prio};
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1, di.pQueueCreateInfos = &qi, di.pEnabledFeatures = &feats;
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
        VkDeviceMemory m = VK_NULL_HANDLE;
        vkAllocateMemory(dv, &ma, nullptr, &m);
        mems.push_back(m);
        return m;
    };
    struct HostBuf { VkBuffer b; VkDeviceMemory m; uint8_t* p; };
    auto hostBuffer = [&](uint64_t size, VkBufferUsageFlags usage) {
        VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bc.size = size ? size : 4, bc.usage = usage;
        HostBuf h = {};
        vkCreateBuffer(dv, &bc, nullptr, &h.b);
        VkMemoryRequirements r;
        vkGetBufferMemoryRequirements(dv, h.b, &r);
        h.m = memory(r, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkBindBufferMemory(dv, h.b, h.m, 0);
        vkMapMemory(dv, h.m, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&h.p));
        return h;
    };
    auto submit = [&]() -> bool {
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1, si.pCommandBuffers = &cb;
        VKC(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        VKC(vkQueueWaitIdle(queue));
        return true;
    };
    for (Job& j : jobs)
    {
        const Shader& sh = *j.shader;
        std::vector<VkDescriptorSetLayoutBinding> lb;
        for (uint32_t i = 0; i < sh.bindingCount; ++i)
        {
            const ShaderBinding& b = sh.bindings[i];
            const VkDescriptorType t = b.kind == BindKind::Cbv ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : b.kind == BindKind::SrvTexture ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                     : b.kind == BindKind::UavTexture ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            lb.push_back({b.binding, t, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        }
        VkDescriptorSetLayoutCreateInfo lc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, uint32_t(lb.size()), lb.data()};
        VkDescriptorSetLayout dsl;
        VKC(vkCreateDescriptorSetLayout(dv, &lc, nullptr, &dsl));
        VkPipelineLayoutCreateInfo plc = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl};
        VkPipelineLayout pl;
        VKC(vkCreatePipelineLayout(dv, &plc, nullptr, &pl));
        VkShaderModuleCreateInfo smc = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, sh.spirvBytes, sh.spirv};
        VkShaderModule sm;
        VKC(vkCreateShaderModule(dv, &smc, nullptr, &sm));
        VkComputePipelineCreateInfo cpc = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpc.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"};
        cpc.layout = pl;
        VkPipeline pipe;
        VKC(vkCreateComputePipelines(dv, VK_NULL_HANDLE, 1, &cpc, nullptr, &pipe));
        VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 16}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}};
        VkDescriptorPoolCreateInfo dpc = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 4, ps};
        VkDescriptorPool dp;
        VKC(vkCreateDescriptorPool(dv, &dpc, nullptr, &dp));
        VkDescriptorSetAllocateInfo dsa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dp, 1, &dsl};
        VkDescriptorSet ds;
        VKC(vkAllocateDescriptorSets(dv, &dsa, &ds));

        HostBuf cbh = hostBuffer(j.cb.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        memcpy(cbh.p, j.cb.data(), j.cb.size());
        std::vector<VkImage> imgs(j.res.size());
        std::vector<VkImageView> views(j.res.size());
        std::vector<VkBuffer> bufs(j.res.size());
        std::vector<HostBuf> stage(j.res.size());
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VKC(vkBeginCommandBuffer(cb, &bi));
        for (size_t i = 0; i < j.res.size(); ++i)
        {
            Res& r = j.res[i];
            stage[i] = hostBuffer(r.bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            memcpy(stage[i].p, r.data.data(), size_t(r.bytes));
            if (r.texture)
            {
                VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                ic.imageType = VK_IMAGE_TYPE_2D, ic.format = r.fmt->vk, ic.extent = {r.w, r.h, 1}, ic.mipLevels = ic.arrayLayers = 1, ic.samples = VK_SAMPLE_COUNT_1_BIT;
                ic.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (r.writable ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
                VKC(vkCreateImage(dv, &ic, nullptr, &imgs[i]));
                VkMemoryRequirements mr;
                vkGetImageMemoryRequirements(dv, imgs[i], &mr);
                VKC(vkBindImageMemory(dv, imgs[i], memory(mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
                VkImageViewCreateInfo vc = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, imgs[i], VK_IMAGE_VIEW_TYPE_2D, r.fmt->vk, {}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
                VKC(vkCreateImageView(dv, &vc, nullptr, &views[i]));
                VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, imgs[i], {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
                VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {r.w, r.h, 1}};
                vkCmdCopyBufferToImage(cb, stage[i].b, imgs[i], VK_IMAGE_LAYOUT_GENERAL, 1, &c);
            }
            else
            {
                VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                bc.size = r.bytes, bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                VKC(vkCreateBuffer(dv, &bc, nullptr, &bufs[i]));
                VkMemoryRequirements mr;
                vkGetBufferMemoryRequirements(dv, bufs[i], &mr);
                VKC(vkBindBufferMemory(dv, bufs[i], memory(mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
                VkBufferCopy c = {0, 0, r.bytes};
                vkCmdCopyBuffer(cb, stage[i].b, bufs[i], 1, &c);
            }
        }
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        std::vector<VkWriteDescriptorSet> wr;
        std::vector<VkDescriptorImageInfo> ii2(sh.bindingCount);
        std::vector<VkDescriptorBufferInfo> bi2(sh.bindingCount);
        for (uint32_t k = 0; k < sh.bindingCount; ++k)
        {
            VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, lb[k].binding, 0, 1, lb[k].descriptorType};
            if (sh.bindings[k].kind == BindKind::Cbv)
                bi2[k] = {cbh.b, 0, VK_WHOLE_SIZE}, w.pBufferInfo = &bi2[k];
            else
            {
                size_t i = 0;
                while (i < j.res.size() && j.res[i].binding != sh.bindings[k].binding) ++i;
                if (i == j.res.size()) return printf("%s: binding %u not given\n", j.name.c_str(), sh.bindings[k].binding), false;
                if (j.res[i].texture) ii2[k] = {VK_NULL_HANDLE, views[i], VK_IMAGE_LAYOUT_GENERAL}, w.pImageInfo = &ii2[k];
                else bi2[k] = {bufs[i], 0, VK_WHOLE_SIZE}, w.pBufferInfo = &bi2[k];
            }
            wr.push_back(w);
        }
        vkUpdateDescriptorSets(dv, uint32_t(wr.size()), wr.data(), 0, nullptr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdDispatch(cb, j.gx, j.gy, 1);
        mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        for (size_t i = 0; i < j.res.size(); ++i)
        {
            Res& r = j.res[i];
            if (!r.writable) continue;
            if (r.texture)
            {
                VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {r.w, r.h, 1}};
                vkCmdCopyImageToBuffer(cb, imgs[i], VK_IMAGE_LAYOUT_GENERAL, stage[i].b, 1, &c);
            }
            else
            {
                VkBufferCopy c = {0, 0, r.bytes};
                vkCmdCopyBuffer(cb, bufs[i], stage[i].b, 1, &c);
            }
        }
        mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        VKC(vkEndCommandBuffer(cb));
        if (!submit()) return false;
        for (size_t i = 0; i < j.res.size(); ++i)
            if (j.res[i].writable && !WriteFile(j.res[i].out, stage[i].p, size_t(j.res[i].bytes))) return printf("cannot write %s\n", j.res[i].out.c_str()), false;
        printf("vulkan %s ok\n", j.name.c_str());
        for (size_t i = 0; i < j.res.size(); ++i)
        {
            if (views[i]) vkDestroyImageView(dv, views[i], nullptr);
            if (imgs[i]) vkDestroyImage(dv, imgs[i], nullptr);
            if (bufs[i]) vkDestroyBuffer(dv, bufs[i], nullptr);
            vkDestroyBuffer(dv, stage[i].b, nullptr);
        }
        vkDestroyBuffer(dv, cbh.b, nullptr);
        for (VkDeviceMemory m : mems) vkFreeMemory(dv, m, nullptr);
        mems.clear();
        vkDestroyDescriptorPool(dv, dp, nullptr), vkDestroyPipeline(dv, pipe, nullptr), vkDestroyShaderModule(dv, sm, nullptr);
        vkDestroyPipelineLayout(dv, pl, nullptr), vkDestroyDescriptorSetLayout(dv, dsl, nullptr);
        vkResetCommandBuffer(cb, 0);
    }
    vkDestroyCommandPool(dv, pool, nullptr);
    vkDestroyDevice(dv, nullptr);
    vkDestroyInstance(inst, nullptr);
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 3) return printf("mlfg_passes d3d12|vulkan JOBFILE\n"), 2;
    std::vector<Job> jobs;
    if (!Parse(argv[2], jobs)) return 1;
    const bool ok = !strcmp(argv[1], "vulkan") ? RunVulkan(jobs) : RunD3D12(jobs);
    return ok ? 0 : 1;
}
