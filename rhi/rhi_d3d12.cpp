// SPDX-License-Identifier: MIT
#include "rhi_d3d12.h"

#include "d3d12_network.h"
#include "log.h"
#include "model.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstring>
#include <map>

using Microsoft::WRL::ComPtr;

namespace nfrux
{
namespace
{
constexpr uint32_t kSlots = 16, kDescriptors = 4096, kUpload = 512 * 1024;
constexpr D3D12_RESOURCE_STATES kSrv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, kUav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

DXGI_FORMAT Dxgi(Format f)
{
    switch (f)
    {
    case Format::R8Unorm: return DXGI_FORMAT_R8_UNORM;
    case Format::RG16Float: return DXGI_FORMAT_R16G16_FLOAT;
    case Format::R32Float: return DXGI_FORMAT_R32_FLOAT;
    case Format::R32Uint: return DXGI_FORMAT_R32_UINT;
    case Format::RGBA8Unorm: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case Format::RGBA8Snorm: return DXGI_FORMAT_R8G8B8A8_SNORM;
    case Format::BGRA8Unorm: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case Format::RGBA16Float: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case Format::RGB10A2Unorm: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case Format::RG11B10Float: return DXGI_FORMAT_R11G11B10_FLOAT;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}
DXGI_FORMAT Typeless(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_SNORM: return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    default: return f;
    }
}
// The view of a host resource: depth planes as their colour formats, typeless colours as UNORM.
DXGI_FORMAT ViewFormat(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R8_TYPELESS: return DXGI_FORMAT_R8_UNORM;
    default: return f;
    }
}

struct Tex12 : Texture
{
    ComPtr<ID3D12Resource>        res;
    DXGI_FORMAT                   view  = DXGI_FORMAT_UNKNOWN;
    mutable D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES         home  = D3D12_RESOURCE_STATE_COMMON;
    mutable bool                  dirty = false;  // written as UAV since the last barrier
};
struct Buf12 : Buffer
{
    ComPtr<ID3D12Resource>        res;
    mutable D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    mutable bool                  dirty = false;
};
struct Net12 : Network
{
    std::unique_ptr<D3D12Graph> graph;
    Buf12                       in, out[2];  // out[i] in the graph's output order
    std::unique_ptr<Tex12>      tex[2];
    uint32_t                    pitch = 0;
};

class Dev12;
struct Pass12 : Pass
{
    Dev12*                      dev = nullptr;
    std::string                 name;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso;
    std::vector<ShaderBinding>  table;  // in descriptor order
    std::vector<uint32_t>       cbvs;
    bool Dispatch(const Bind* binds, uint32_t count, uint32_t gx, uint32_t gy) override;
};

class Dev12 : public Device
{
public:
    ComPtr<ID3D12Device>               d;
    Caps                               c;
    ComPtr<ID3D12DescriptorHeap>       heap, cpuHeap;
    ComPtr<ID3D12Resource>             upload, token;
    uint8_t*                           map = nullptr;
    volatile uint32_t*                 tokens = nullptr;
    uint32_t                           slotValue[kSlots] = {}, next = 0, slot = 0, used = 0, descUsed = 0, inc = 0;
    ComPtr<ID3D12GraphicsCommandList2> list;
    std::vector<std::unique_ptr<Tex12>> wrapped;
    std::vector<const Buf12*>          touched;  // buffers to return to COMMON
    std::vector<D3D12_RESOURCE_BARRIER> pending;
    struct Read { ComPtr<ID3D12Resource> buf; std::shared_ptr<std::vector<uint8_t>> out; uint32_t pitch, row, rows, slot, value; };
    std::vector<Read>                  reads;

    const Caps& caps() const override { return c; }

    bool Init(ID3D12Device* device)
    {
        d = device;
        ComPtr<IDXGIFactory4> f;
        ComPtr<IDXGIAdapter1> a;
        DXGI_ADAPTER_DESC1    ad = {};
        const auto create = reinterpret_cast<HRESULT(WINAPI*)(REFIID, void**)>(GetProcAddress(LoadLibraryA("dxgi.dll"), "CreateDXGIFactory1"));
        if (create && SUCCEEDED(create(IID_PPV_ARGS(&f))) && SUCCEEDED(f->EnumAdapterByLuid(d->GetAdapterLuid(), IID_PPV_ARGS(&a))))
            a->GetDesc1(&ad);
        char name[128] = {};
        WideCharToMultiByte(CP_UTF8, 0, ad.Description, -1, name, 127, nullptr, nullptr);
        c.name = name, c.vendorId = ad.VendorId, c.deviceId = ad.DeviceId;
        for (D3D_SHADER_MODEL m : {D3D_SHADER_MODEL(0x69), D3D_SHADER_MODEL(0x68), D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_4, D3D_SHADER_MODEL_6_2, D3D_SHADER_MODEL_6_0})
        {
            D3D12_FEATURE_DATA_SHADER_MODEL sm = {m};
            if (SUCCEEDED(d->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm)))) { c.shaderModel = uint32_t(sm.HighestShaderModel); break; }
        }
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1 = {};
        D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4 = {};
        d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof(o1));
        d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof(o4));
        c.int64 = o1.Int64ShaderOps, c.native16Bit = o4.Native16BitShaderOpsSupported;
        c.atomic64 = c.int64 && c.shaderModel >= 0x66;  // raw buffers: part of shader model 6.6
        c.dot4 = c.shaderModel >= 0x64;
        c.cuda = c.vendorId == 0x10de && GetModuleHandleA("nvapi64.dll") != nullptr;
        if (c.shaderModel < 0x60) return Log("shader model %x: below 6.0", c.shaderModel), false;
        D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kSlots * kDescriptors, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        D3D12_DESCRIPTOR_HEAP_DESC cd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16};
        if (FAILED(d->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap))) || FAILED(d->CreateDescriptorHeap(&cd, IID_PPV_ARGS(&cpuHeap)))) return false;
        inc    = d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        upload = Raw(uint64_t(kUpload) * kSlots, D3D12_HEAP_TYPE_UPLOAD, false);
        token  = Raw(256, D3D12_HEAP_TYPE_READBACK, false);
        void *m1 = nullptr, *m2 = nullptr;
        if (!upload || !token || FAILED(upload->Map(0, nullptr, &m1)) || FAILED(token->Map(0, nullptr, &m2))) return false;
        map = static_cast<uint8_t*>(m1), tokens = static_cast<uint32_t*>(m2);
        Log("D3D12: %s, shader model %x, dot4 %d, int64 %d, 64-bit atomics %d, native 16-bit %d, nvapi %d", name, c.shaderModel, c.dot4, c.int64, c.atomic64, c.native16Bit, c.cuda);
        return true;
    }
    ComPtr<ID3D12Resource> Raw(uint64_t size, D3D12_HEAP_TYPE type, bool uav)
    {
        D3D12_HEAP_PROPERTIES hp = {type};
        D3D12_RESOURCE_DESC   r  = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, (size + 255) & ~255ull, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                  uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
        ComPtr<ID3D12Resource> out;
        d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
                                   : type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&out));
        return out;
    }
    std::unique_ptr<Tex12> MakeTex(DXGI_FORMAT view, uint32_t w, uint32_t h, bool typeless)
    {
        auto t = std::make_unique<Tex12>();
        D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC   r  = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, w, h, 1, 1, typeless ? Typeless(view) : view, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
        D3D12_FEATURE_DATA_FORMAT_SUPPORT fs = {view};
        d->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs));
        if (fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) r.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &r, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&t->res)))) return nullptr;
        t->view = view, t->width = w, t->height = h;
        return t;
    }
    std::unique_ptr<Texture> CreateTexture(Format f, uint32_t w, uint32_t h, bool uintView) override
    {
        auto t = MakeTex(Dxgi(f), w, h, uintView);
        if (t) t->format = f, t->uintView = uintView;
        return t;
    }
    std::unique_ptr<Texture> CreateTextureLike(const ExternalImage& e, uint32_t w, uint32_t h) override
    {
        auto* r = static_cast<ID3D12Resource*>(e.handle);
        auto  t = MakeTex(ViewFormat(e.format ? DXGI_FORMAT(e.format) : r->GetDesc().Format), w, h, false);
        if (t) t->format = Format::Native;
        return t;
    }
    std::unique_ptr<Buffer> CreateBuffer(uint64_t size) override
    {
        auto b = std::make_unique<Buf12>();
        b->res = Raw(size, D3D12_HEAP_TYPE_DEFAULT, true), b->size = size;
        return b->res ? std::move(b) : nullptr;
    }
    std::unique_ptr<Pass> CreatePass(const Shader& s) override
    {
        auto p  = std::make_unique<Pass12>();
        p->dev  = this, p->name = s.name;
        std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
        std::vector<D3D12_ROOT_PARAMETER>   params;
        for (uint32_t i = 0; i < s.bindingCount; ++i)
        {
            const ShaderBinding& b = s.bindings[i];
            if (b.kind == BindKind::Cbv)
            {
                D3D12_ROOT_PARAMETER rp = {D3D12_ROOT_PARAMETER_TYPE_CBV};
                rp.Descriptor.ShaderRegister = b.binding;
                params.push_back(rp), p->cbvs.push_back(b.binding);
                continue;
            }
            const bool uav = b.kind == BindKind::UavTexture || b.kind == BindKind::UavBuffer;
            ranges.push_back({uav ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, b.binding, 0, UINT(p->table.size())});
            p->table.push_back(b);
        }
        D3D12_ROOT_PARAMETER tp = {D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE};
        tp.DescriptorTable = {UINT(ranges.size()), ranges.data()};
        params.push_back(tp);
        D3D12_STATIC_SAMPLER_DESC ss[4] = {};
        for (uint32_t i = 0; i < 4; ++i)
        {
            ss[i].Filter   = i & 1 ? D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_POINT;
            ss[i].AddressU = ss[i].AddressV = ss[i].AddressW = i < 2 ? D3D12_TEXTURE_ADDRESS_MODE_CLAMP : D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            ss[i].BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
            ss[i].MaxLOD = 1000.0f, ss[i].MaxAnisotropy = 1, ss[i].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            ss[i].ShaderRegister = 1000 + i;
        }
        D3D12_ROOT_SIGNATURE_DESC rs = {UINT(params.size()), params.data(), 4, ss};
        ComPtr<ID3DBlob> blob, err;
        static const auto serialize = reinterpret_cast<PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>(GetProcAddress(LoadLibraryA("d3d12.dll"), "D3D12SerializeRootSignature"));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        if (FAILED(serialize(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) ||
            FAILED(d->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&p->root))))
            return Log("%s: root signature failed %s", s.name, err ? static_cast<char*>(err->GetBufferPointer()) : ""), nullptr;
        pd.pRootSignature = p->root.Get(), pd.CS = {s.dxil, s.dxilBytes};
        if (const HRESULT hr = d->CreateComputePipelineState(&pd, IID_PPV_ARGS(&p->pso)); FAILED(hr))
            return Log("%s: pipeline failed %lx", s.name, (unsigned long)hr), nullptr;
        return p;
    }
    std::unique_ptr<Network> CreateNetwork(uint32_t w, uint32_t h, std::string* error) override
    {
        auto n = std::make_unique<Net12>();
        const char* ml = Env("NFRUX_ML");
        const bool  dot4 = c.dot4 && !(ml && !strcmp(ml, "scalar"));
        n->graph = D3D12Graph::Create(d.Get(), DescribeModel(w, h), dot4, error);
        if (!n->graph) return nullptr;
        n->pitch  = (w / 2 * 4 + 255) & ~255u;
        n->in.res = Raw(uint64_t(w) * h * kInputChannels, D3D12_HEAP_TYPE_DEFAULT, true), n->in.size = uint64_t(w) * h * kInputChannels;
        const ModelBindings b = GetModelBindings();
        for (uint32_t i = 0; i < 2; ++i)
        {
            n->out[i].res  = Raw(uint64_t(n->pitch) * (h / 2), D3D12_HEAP_TYPE_DEFAULT, true);
            n->out[i].size = uint64_t(n->pitch) * (h / 2);
            n->tex[i]      = MakeTex(DXGI_FORMAT_R8G8B8A8_SNORM, w / 2, h / 2, false);
            if (!n->out[i].res || !n->tex[i]) return nullptr;
            n->tex[i]->format = Format::RGBA8Snorm;
            (n->graph->outputBinding(i) == b.params ? n->params : n->gate) = n->tex[i].get();
        }
        n->input = &n->in;
        static bool logged = false;
        if (!logged) Log("network %ux%u on D3D12 %s", w, h, dot4 ? "dot4 (shader model 6.4)" : "int32 (shader model 6.0)"), logged = true;
        return n;
    }

    // --- recording
    bool BeginFrame(void* commands) override
    {
        if (FAILED(static_cast<ID3D12GraphicsCommandList*>(commands)->QueryInterface(IID_PPV_ARGS(&list)))) return false;
        Poll();
        slot = next++ % kSlots;
        if (slotValue[slot] && tokens[slot] != slotValue[slot])  // still in flight: give it 2 s
            for (int i = 0; i < 2000 && tokens[slot] != slotValue[slot]; ++i) Sleep(1);
        slotValue[slot] = next;
        used = descUsed = 0;
        ID3D12DescriptorHeap* h[] = {heap.Get()};
        list->SetDescriptorHeaps(1, h);
        return true;
    }
    void EndFrame() override
    {
        for (auto& t : wrapped) Want(*t, t->home);
        for (const Buf12* b : touched) if (b->state != D3D12_RESOURCE_STATE_COMMON) Transition(b->res.Get(), b->state, D3D12_RESOURCE_STATE_COMMON), b->dirty = false;
        touched.clear();
        Flush();
        const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p = {token->GetGPUVirtualAddress() + slot * 4, slotValue[slot]};
        const D3D12_WRITEBUFFERIMMEDIATE_MODE      m = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
        list->WriteBufferImmediate(1, &p, &m);
        for (auto& t : wrapped) retired.push_back({std::move(t), slot, slotValue[slot]});
        wrapped.clear();
        list.Reset();
    }
    struct Retired { std::unique_ptr<Tex12> t; uint32_t slot, value; };
    std::vector<Retired> retired;
    void Poll()
    {
        auto done = [&](uint32_t s, uint32_t v) { return tokens[s] == v || slotValue[s] != v; };
        retired.erase(std::remove_if(retired.begin(), retired.end(), [&](const Retired& r) { return done(r.slot, r.value); }), retired.end());
    }
    Texture* Wrap(const ExternalImage& e, bool) override
    {
        auto* r = static_cast<ID3D12Resource*>(e.handle);
        for (auto& t : wrapped) if (t->res.Get() == r) return t.get();
        auto t   = std::make_unique<Tex12>();
        const D3D12_RESOURCE_DESC ds = r->GetDesc();
        t->res = r, t->view = ViewFormat(e.format ? DXGI_FORMAT(e.format) : ds.Format);
        t->width = e.width ? e.width : uint32_t(ds.Width), t->height = e.height ? e.height : ds.Height;
        t->state = t->home = D3D12_RESOURCE_STATES(e.state), t->format = Format::Native;
        wrapped.push_back(std::move(t));
        return wrapped.back().get();
    }
    void Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES& from, D3D12_RESOURCE_STATES to)
    {
        D3D12_RESOURCE_BARRIER b = {};
        b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
        pending.push_back(b);
        from = to;
    }
    template <class T> void Want(const T& t, D3D12_RESOURCE_STATES s)
    {
        if (t.state != s) Transition(t.res.Get(), t.state, s), t.dirty = false;
        else if (s == kUav && t.dirty)
        {
            D3D12_RESOURCE_BARRIER b = {D3D12_RESOURCE_BARRIER_TYPE_UAV};
            b.UAV.pResource = t.res.Get();
            pending.push_back(b);
            t.dirty = false;
        }
    }
    void Touch(const Buf12& b) { if (std::find(touched.begin(), touched.end(), &b) == touched.end()) touched.push_back(&b); }
    void Flush() { if (!pending.empty()) list->ResourceBarrier(UINT(pending.size()), pending.data()), pending.clear(); }
    D3D12_GPU_VIRTUAL_ADDRESS Constants(const void* data, size_t size)
    {
        const uint32_t start = (used + 255) & ~255u;
        if (start + size > kUpload) return 0;
        used = start + uint32_t(size);
        memcpy(map + uint64_t(slot) * kUpload + start, data, size);
        return upload->GetGPUVirtualAddress() + uint64_t(slot) * kUpload + start;
    }
    void CopyTexture(const Texture& s, const Texture& dst, uint32_t x, uint32_t y) override
    {
        auto &a = static_cast<const Tex12&>(s), &b = static_cast<const Tex12&>(dst);
        Want(a, D3D12_RESOURCE_STATE_COPY_SOURCE), Want(b, D3D12_RESOURCE_STATE_COPY_DEST), Flush();
        D3D12_TEXTURE_COPY_LOCATION ld = {b.res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, ls = {a.res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        const D3D12_BOX box = {0, 0, 0, std::min(a.width, b.width - x), std::min(a.height, b.height - y), 1};
        list->CopyTextureRegion(&ld, x, y, 0, &ls, &box);
    }
    void CopyRegion(const Texture& s, uint32_t x, uint32_t y, const Texture& dst) override
    {
        auto &a = static_cast<const Tex12&>(s), &b = static_cast<const Tex12&>(dst);
        Want(a, D3D12_RESOURCE_STATE_COPY_SOURCE), Want(b, D3D12_RESOURCE_STATE_COPY_DEST), Flush();
        D3D12_TEXTURE_COPY_LOCATION ld = {b.res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, ls = {a.res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        const D3D12_BOX box = {x, y, 0, x + b.width, y + b.height, 1};
        list->CopyTextureRegion(&ld, 0, 0, 0, &ls, &box);
    }
    void Clear(const Texture& t, float v) override
    {
        auto& a = static_cast<const Tex12&>(t);
        Want(a, kUav), Flush();
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {a.view, D3D12_UAV_DIMENSION_TEXTURE2D};
        const auto cpu = cpuHeap->GetCPUDescriptorHandleForHeapStart();
        d->CreateUnorderedAccessView(a.res.Get(), nullptr, &u, cpu);
        d->CreateUnorderedAccessView(a.res.Get(), nullptr, &u, Cpu(descUsed));
        const float f[4] = {v, v, v, v};
        list->ClearUnorderedAccessViewFloat(Gpu(descUsed++), cpu, a.res.Get(), f, 0, nullptr);
        a.dirty = true;
    }
    void Fill(const Buffer& b, uint32_t v) override
    {
        auto& a = static_cast<const Buf12&>(b);
        Touch(a), Want(a, kUav), Flush();
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {DXGI_FORMAT_R32_UINT, D3D12_UAV_DIMENSION_BUFFER};
        u.Buffer = {0, UINT(a.size / 4), 0, 0, D3D12_BUFFER_UAV_FLAG_NONE};
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = cpuHeap->GetCPUDescriptorHandleForHeapStart();
        cpu.ptr += inc;
        d->CreateUnorderedAccessView(a.res.Get(), nullptr, &u, cpu);
        d->CreateUnorderedAccessView(a.res.Get(), nullptr, &u, Cpu(descUsed));
        const UINT x[4] = {v, v, v, v};
        list->ClearUnorderedAccessViewUint(Gpu(descUsed++), cpu, a.res.Get(), x, 0, nullptr);
        a.dirty = true;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE Cpu(uint32_t i) { auto h = heap->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(slot * kDescriptors + i) * inc; return h; }
    D3D12_GPU_DESCRIPTOR_HANDLE Gpu(uint32_t i) { auto h = heap->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(slot * kDescriptors + i) * inc; return h; }
    void RunNetworks(Network* const* nets, uint32_t count) override
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            auto& n = static_cast<Net12&>(*nets[i]);
            Touch(n.in), Want(n.in, kSrv);
            for (auto& o : n.out) Touch(o), Want(o, kUav);
        }
        Flush();
        for (uint32_t i = 0; i < count; ++i)
        {
            auto& n = static_cast<Net12&>(*nets[i]);
            ID3D12Resource* o[2] = {n.out[0].res.Get(), n.out[1].res.Get()};
            n.graph->Record(list.Get(), n.in.res->GetGPUVirtualAddress(), o, n.pitch);
        }
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t k = 0; k < 2; ++k)
            {
                auto& n = static_cast<Net12&>(*nets[i]);
                Want(n.out[k], D3D12_RESOURCE_STATE_COPY_SOURCE), Want(*n.tex[k], D3D12_RESOURCE_STATE_COPY_DEST);
            }
        Flush();
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t k = 0; k < 2; ++k)
            {
                auto& n = static_cast<Net12&>(*nets[i]);
                D3D12_TEXTURE_COPY_LOCATION ld = {n.tex[k]->res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, ls = {n.out[k].res.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
                ls.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_SNORM, n.tex[k]->width, n.tex[k]->height, 1, n.pitch};
                list->CopyTextureRegion(&ld, 0, 0, 0, &ls, nullptr);
            }
    }
    void ReadBack(const Texture& t, std::shared_ptr<std::vector<uint8_t>> out) override
    {
        auto& a = static_cast<const Tex12&>(t);
        Want(a, D3D12_RESOURCE_STATE_COPY_SOURCE), Flush();
        D3D12_RESOURCE_DESC ds = a.res->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
        UINT rows; UINT64 row, total;
        d->GetCopyableFootprints(&ds, 0, 1, 0, &fp, &rows, &row, &total);
        Read r = {Raw(total, D3D12_HEAP_TYPE_READBACK, false), out, fp.Footprint.RowPitch, uint32_t(row), rows, slot, slotValue[slot]};
        D3D12_TEXTURE_COPY_LOCATION ls = {a.res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, ld = {r.buf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        ld.PlacedFootprint = fp;
        list->CopyTextureRegion(&ld, 0, 0, 0, &ls, nullptr);
        reads.push_back(r);
    }
    void Mark(const char*) override {}
    void WaitIdle() override
    {
        for (uint32_t s = 0; s < kSlots; ++s)
            for (int i = 0; i < 2000 && slotValue[s] && tokens[s] != slotValue[s]; ++i) Sleep(1);
        for (auto& r : reads)
        {
            uint8_t* p = nullptr;
            r.buf->Map(0, nullptr, reinterpret_cast<void**>(&p));
            r.out->resize(size_t(r.row) * r.rows);
            for (uint32_t y = 0; y < r.rows; ++y) memcpy(r.out->data() + size_t(y) * r.row, p + size_t(y) * r.pitch, r.row);
            r.buf->Unmap(0, nullptr);
        }
        reads.clear();
        Poll();
    }
};

bool Pass12::Dispatch(const Bind* binds, uint32_t count, uint32_t gx, uint32_t gy)
{
    auto find = [&](uint32_t b) -> const Bind* { for (uint32_t i = 0; i < count; ++i) if (binds[i].binding == b) return &binds[i]; return nullptr; };
    if (dev->descUsed + table.size() > kDescriptors) return Log("%s: out of descriptors", name.c_str()), false;
    const uint32_t first = dev->descUsed;
    dev->descUsed += uint32_t(table.size());
    for (size_t i = 0; i < table.size(); ++i)
    {
        const Bind* b = find(table[i].binding);
        const auto  h = dev->Cpu(first + uint32_t(i));
        const bool  uav = table[i].kind == BindKind::UavTexture || table[i].kind == BindKind::UavBuffer;
        if (!b || (!b->texture && !b->buffer)) return Log("%s: binding %u unbound", name.c_str(), table[i].binding), false;
        if (b->buffer)
        {
            auto& x = static_cast<const Buf12&>(*b->buffer);
            dev->Touch(x), dev->Want(x, uav ? kUav : kSrv);
            if (uav)
            {
                D3D12_UNORDERED_ACCESS_VIEW_DESC u = {DXGI_FORMAT_R32_TYPELESS, D3D12_UAV_DIMENSION_BUFFER};
                u.Buffer = {0, UINT(x.size / 4), 0, 0, D3D12_BUFFER_UAV_FLAG_RAW};
                dev->d->CreateUnorderedAccessView(x.res.Get(), nullptr, &u, h);
            }
            else
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC s = {DXGI_FORMAT_R32_TYPELESS, D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
                s.Buffer = {0, UINT(x.size / 4), 0, D3D12_BUFFER_SRV_FLAG_RAW};
                dev->d->CreateShaderResourceView(x.res.Get(), &s, h);
            }
            continue;
        }
        auto& x = static_cast<const Tex12&>(*b->texture);
        dev->Want(x, uav ? kUav : kSrv);
        const DXGI_FORMAT f = b->asUint ? DXGI_FORMAT_R32_UINT : x.view;
        if (uav)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC u = {f, D3D12_UAV_DIMENSION_TEXTURE2D};
            dev->d->CreateUnorderedAccessView(x.res.Get(), nullptr, &u, h);
        }
        else
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC s = {f, D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
            s.Texture2D.MipLevels = 1;
            dev->d->CreateShaderResourceView(x.res.Get(), &s, h);
        }
    }
    dev->Flush();
    auto* l = dev->list.Get();
    l->SetComputeRootSignature(root.Get());
    l->SetPipelineState(pso.Get());
    for (size_t i = 0; i < cbvs.size(); ++i)
    {
        const Bind* b = find(cbvs[i]);
        const auto  va = b && b->data ? dev->Constants(b->data, b->size) : 0;
        if (!va) return Log("%s: constants %u", name.c_str(), cbvs[i]), false;
        l->SetComputeRootConstantBufferView(UINT(i), va);
    }
    l->SetComputeRootDescriptorTable(UINT(cbvs.size()), dev->Gpu(first));
    l->Dispatch(gx, gy, 1);
    for (const ShaderBinding& t : table)
        if (t.kind == BindKind::UavTexture) static_cast<const Tex12&>(*find(t.binding)->texture).dirty = true;
        else if (t.kind == BindKind::UavBuffer) static_cast<const Buf12&>(*find(t.binding)->buffer).dirty = true;
    return true;
}
}  // namespace

std::unique_ptr<Device> CreateD3D12Device(ID3D12Device* device)
{
    auto d = std::make_unique<Dev12>();
    return d->Init(device) ? std::move(d) : nullptr;
}
}  // namespace nfrux