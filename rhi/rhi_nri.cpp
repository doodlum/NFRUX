// NFRUX's rhi on NVIDIA's NRI: one implementation for D3D12 and Vulkan, on the host's device and command buffer. Barriers,
// descriptor recycling and completion tokens are here; the network goes through native handles. SPDX-License-Identifier: MIT
#include "rhi_nri.h"

#include "cuda_vulkan.h"
#include "d3d12_network.h"
#include "log.h"
#include "marker.h"
#include "model.h"

#include <d3d12.h>
#include <vulkan/vulkan.h>

#include <NRI.h>
#include <Extensions/NRIHelper.h>
#include <Extensions/NRIWrapperD3D12.h>
#include <Extensions/NRIWrapperVK.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

namespace nfrux
{
namespace
{
thread_local bool t_recording = false;
}  // namespace
void LogBarriers(ID3D12GraphicsCommandList* list);
namespace
{
constexpr uint32_t kSlots = 32, kSets = 160, kConstants = 512, kConstantStride = 512;
#define NRI_OK(x) ((x) == nri::Result::SUCCESS)

nri::Format ToNri(Format f)
{
    switch (f)
    {
    case Format::R8Unorm: return nri::Format::R8_UNORM;
    case Format::RG16Float: return nri::Format::RG16_SFLOAT;
    case Format::R32Float: return nri::Format::R32_SFLOAT;
    case Format::R32Uint: return nri::Format::R32_UINT;
    case Format::RGBA8Unorm: return nri::Format::RGBA8_UNORM;
    case Format::RGBA8Snorm: return nri::Format::RGBA8_SNORM;
    case Format::BGRA8Unorm: return nri::Format::BGRA8_UNORM;
    case Format::RGBA16Float: return nri::Format::RGBA16_SFLOAT;
    case Format::RGB10A2Unorm: return nri::Format::R10_G10_B10_A2_UNORM;
    case Format::RG11B10Float: return nri::Format::R11_G11_B10_UFLOAT;
    default: return nri::Format::UNKNOWN;
    }
}
bool IsDepth(nri::Format f)
{
    return f == nri::Format::D16_UNORM || f == nri::Format::D24_UNORM_S8_UINT || f == nri::Format::D32_SFLOAT || f == nri::Format::D32_SFLOAT_S8_UINT;
}

const nri::AccessLayoutStage kSrv = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER};
const nri::AccessLayoutStage kUav = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
const nri::AccessLayoutStage kSrc = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
const nri::AccessLayoutStage kDst = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
// The host's images, in and out (DLSS-FG Programming Guide, section 6): D3D12 inputs NON_PIXEL_SHADER_RESOURCE, outputs UNORDERED_ACCESS. Vulkan GENERAL, or
// SHADER_READ_ONLY_OPTIMAL for an input the host declares not read-write.
const nri::AccessLayoutStage kHostVkRead = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::ALL};
const nri::AccessLayoutStage kHostVk = {nri::AccessBits::SHADER_RESOURCE | nri::AccessBits::SHADER_RESOURCE_STORAGE | nri::AccessBits::COPY_SOURCE | nri::AccessBits::COPY_DESTINATION,
                                        nri::Layout::GENERAL, nri::StageBits::ALL};
bool Same(const nri::AccessLayoutStage& a, const nri::AccessLayoutStage& b) { return a.access == b.access && a.layout == b.layout; }

class DevN;
struct TexN : Texture
{
    DevN*           dev = nullptr;
    nri::Texture*   tex = nullptr;
    nri::Descriptor *srv = nullptr, *uav = nullptr;
    nri::PlaneBits  planes = nri::PlaneBits::ALL;
    uint64_t        native = 0;
    mutable nri::AccessLayoutStage state = {nri::AccessBits::NONE, nri::Layout::UNDEFINED, nri::StageBits::ALL};
    ~TexN() override;
};
struct BufN : Buffer
{
    DevN*            dev = nullptr;
    nri::Buffer*     buf = nullptr;
    nri::Descriptor *srv = nullptr, *uav = nullptr;
    mutable nri::AccessStage state = {nri::AccessBits::NONE, nri::StageBits::NONE};
    ~BufN() override;
};
struct PassN : Pass
{
    DevN*                      dev = nullptr;
    std::string                name;
    std::vector<ShaderBinding> bindings;
    nri::PipelineLayout*       layout = nullptr;
    nri::Pipeline*             pipeline = nullptr;
    ~PassN() override;
    bool Dispatch(const Bind* binds, uint32_t count, uint32_t gx, uint32_t gy) override;
};
struct NetN : Network
{
    std::unique_ptr<CudaVulkanGraph> cudaVk;
    CudaVulkanGraph::Buffer          cudaBuf[3];  // input, outputs: with addresses
    ~NetN() override
    {
        if (!cudaVk) return;
        in.reset(), out[0].reset(), out[1].reset();
        for (auto& b : cudaBuf) cudaVk->Destroy(b);
    }
    std::unique_ptr<D3D12Graph>    d3d;
    std::unique_ptr<CudaGraph>     cuda;
    std::unique_ptr<ffx_ml::Graph> vk;
    std::unique_ptr<BufN>          in, out[2];  // params, gate
    std::unique_ptr<TexN>          tex[2];
    uint32_t                       pitch = 0, order[2] = {0, 1};  // D3D12: which of out[] the graph's output i is
};

class DevN : public Device
{
public:
    nri::Device*            dev = nullptr;
    nri::CoreInterface      n = {};
    nri::WrapperVKInterface nvk = {};
    nri::WrapperD3D12Interface nd3 = {};
    ID3D12Device*           d3d = nullptr;
    VulkanHost              vk;
    bool                    isVk = false;
    Caps                    c;
    nri::Descriptor*        samplers[4] = {};
    nri::DescriptorPool*    pools[kSlots] = {};
    nri::Buffer *           constants = nullptr, *tokenUp = nullptr, *tokenBack = nullptr, *ones = nullptr;
    uint64_t                onesSize = 0;
    std::vector<nri::Buffer*> old;
    std::vector<nri::Descriptor*> cbv;
    uint8_t*                map = nullptr;
    uint32_t*               tokenUpMap = nullptr;
    volatile uint32_t*      tokens = nullptr;
    uint32_t                slotValue[kSlots] = {}, next = 0, slot = 0, usedConstants = 0;
    nri::CommandBuffer*     cmd = nullptr;
    void*                   nativeCmd = nullptr;
    bool                    stalled = false;
    std::vector<std::unique_ptr<TexN>> wrapped;
    std::vector<nri::AccessLayoutStage> wrappedHome;
    std::vector<D3D12_RESOURCE_BARRIER> hostBack;  // host inputs that arrived in another state than the guide's: back to it at the end
    std::vector<nri::TextureBarrierDesc> tb;
    std::vector<nri::BufferBarrierDesc>  bb;
    std::vector<const BufN*>             used;  // buffers touched by this recording
    struct Retired { std::vector<std::unique_ptr<TexN>> t; nri::CommandBuffer* cmd; uint32_t slot, value; };
    std::vector<Retired>    retired;
    struct Read { nri::Buffer* buf; std::shared_ptr<std::vector<uint8_t>> out; uint32_t pitch, row, rows, slot, value; uint64_t size; };
    std::vector<Read>       reads;
    std::vector<std::pair<uint64_t, nri::Buffer*>> freeReads;  // small ones, kept
    std::vector<std::pair<uint32_t, nri::Buffer*>> dying;      // the recording they were finished in; destroyed 16 later

    ~DevN() override
    {
        for (auto& [name, t] : times) Log("gpu %-44s %.4f ms a batch, %u dispatches", name.c_str(), t.first / std::max(1u, timedFrames - 60), t.second / std::max(1u, timedFrames - 60));
        if (marks) n.DestroyQueryPool(marks);
        if (markBack) n.DestroyBuffer(markBack);
        for (auto& r : retired) if (r.cmd) n.DestroyCommandBuffer(r.cmd);
        wrapped.clear(), retired.clear(), scratch.clear();
        if (cudaState == 1 && !Lost()) MarkerOk(isVk ? cudaMarker : Rung());
        if (!vk.marker.empty()) MarkerOk(vk.marker);
        for (auto* x : cbv) n.DestroyDescriptor(x);
        for (auto* x : samplers) if (x) n.DestroyDescriptor(x);
        for (auto* x : pools) if (x) n.DestroyDescriptorPool(x);
        for (auto* x : {constants, tokenUp, tokenBack, ones}) if (x) n.DestroyBuffer(x);
        for (auto* x : old) n.DestroyBuffer(x);
        for (auto& x : reads) n.DestroyBuffer(x.buf);
        for (auto& x : freeReads) n.DestroyBuffer(x.second);
        for (auto& x : dying) n.DestroyBuffer(x.second);
        if (dev) nri::nriDestroyDevice(dev);
    }
    const Caps& caps() const override { return c; }
    static void NRI_CALL Message(nri::Message type, const char* file, uint32_t line, const char* message, void*)
    {
        if (type != nri::Message::INFO) Log("NRI: %s (%s:%u)", message, file, line);
    }

    nri::Buffer* Committed(uint64_t size, nri::MemoryLocation where, nri::BufferUsageBits usage, bool raw = false)
    {
        nri::BufferDesc d = {};
        d.size = (size + 255) & ~255ull, d.usage = usage, d.byteAddress = raw;
        nri::Buffer* b = nullptr;
        if (!NRI_OK(n.CreateCommittedBuffer(*dev, where, 0.0f, d, b))) return nullptr;
        if (Env("NFRUX_TEST_NAMES")) n.SetDebugName(b, "NFRUX buffer");
        return b;
    }
    bool Init()
    {
        if (!NRI_OK(nri::nriGetInterface(*dev, NRI_INTERFACE(nri::CoreInterface), &n))) return false;
        if (!(isVk ? NRI_OK(nri::nriGetInterface(*dev, NRI_INTERFACE(nri::WrapperVKInterface), &nvk))
                   : NRI_OK(nri::nriGetInterface(*dev, NRI_INTERFACE(nri::WrapperD3D12Interface), &nd3)))) return false;
        const nri::DeviceDesc& dd = n.GetDeviceDesc(*dev);
        c.name = dd.adapterDesc.name, c.vendorId = uint32_t(dd.adapterDesc.vendor), c.deviceId = dd.adapterDesc.deviceId;
        for (uint32_t i = 0; i < 4; ++i)
        {
            nri::SamplerDesc s = {};
            s.filters.min = s.filters.mag = i & 1 ? nri::Filter::LINEAR : nri::Filter::NEAREST;
            s.filters.mip = nri::Filter::NEAREST, s.mipMax = 1000.0f;
            s.addressModes.u = s.addressModes.v = s.addressModes.w = i < 2 ? nri::AddressMode::CLAMP_TO_EDGE : nri::AddressMode::CLAMP_TO_BORDER;
            if (!NRI_OK(n.CreateSampler(*dev, s, samplers[i]))) return false;
        }
        nri::DescriptorPoolDesc pd = {};
        pd.descriptorSetMaxNum = kSets, pd.samplerMaxNum = kSets * 4, pd.constantBufferMaxNum = kSets, pd.textureMaxNum = kSets * 8;
        pd.storageTextureMaxNum = kSets * 4, pd.structuredBufferMaxNum = kSets, pd.storageStructuredBufferMaxNum = kSets;
        for (auto& p : pools) if (!NRI_OK(n.CreateDescriptorPool(*dev, pd, p))) return false;
        constants = Committed(uint64_t(kSlots) * kConstants * kConstantStride, nri::MemoryLocation::HOST_UPLOAD, nri::BufferUsageBits::CONSTANT);
        tokenUp   = Committed(256, nri::MemoryLocation::HOST_UPLOAD, nri::BufferUsageBits::NONE);
        tokenBack = Committed(256, nri::MemoryLocation::HOST_READBACK, nri::BufferUsageBits::NONE);
        if (!constants || !tokenUp || !tokenBack) return false;
        map        = static_cast<uint8_t*>(n.MapBuffer(*constants, 0, nri::WHOLE_SIZE));
        tokenUpMap = static_cast<uint32_t*>(n.MapBuffer(*tokenUp, 0, nri::WHOLE_SIZE));
        tokens     = static_cast<uint32_t*>(n.MapBuffer(*tokenBack, 0, nri::WHOLE_SIZE));
        if (!map || !tokenUpMap || !tokens) return false;
        for (uint32_t i = 0; i < kSlots * kConstants; ++i)
        {
            nri::BufferViewDesc v = {};
            v.buffer = constants, v.type = nri::BufferView::CONSTANT_BUFFER, v.offset = uint64_t(i) * kConstantStride, v.size = kConstantStride;
            nri::Descriptor* d = nullptr;
            if (!NRI_OK(n.CreateBufferView(v, d))) return false;
            cbv.push_back(d);
        }
        return true;
    }
    bool InitD3D12(ID3D12Device* device)
    {
        d3d = device;
        nri::DeviceCreationD3D12Desc d = {};
        d.d3d12Device = device, d.disableD3D12EnhancedBarriers = true, d.disableNVAPIInitialization = true;
        d.callbackInterface.MessageCallback = Message;
        d.enableNRIValidation = EnvFlag("NFRUX_VALIDATE", false);
        if (!NRI_OK(nri::nriCreateDeviceFromD3D12Device(d, dev)) || !Init()) return false;
        for (int m : {0x69, 0x68, 0x66, 0x64, 0x62, 0x60})
        {
            D3D12_FEATURE_DATA_SHADER_MODEL sm = {D3D_SHADER_MODEL(m)};
            if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm)))) { c.shaderModel = uint32_t(sm.HighestShaderModel); break; }
        }
        LogD3D12Messages(device);
        c.dot4 = c.shaderModel >= 0x64;
        c.cuda  = c.vendorId == uint32_t(nri::Vendor::NVIDIA) && GetModuleHandleA("nvapi64.dll");
        Log("D3D12 (NRI): %s, shader model %x, dot4 %d", c.name.c_str(), c.shaderModel, c.dot4);
        return c.shaderModel >= 0x60;
    }
    bool InitVulkan(const VulkanHost& h)
    {
        vk = h, isVk = true;
        nri::DeviceCreationVKDesc d = {};
        d.vkInstance = h.instance, d.vkDevice = h.device, d.vkPhysicalDevice = h.physicalDevice, d.minorVersion = uint8_t(h.minorVersion);
        
        d.unknownFeatures = true;  // only what Vulkan 1.0 reports: the host's enabled set is not ours to assume
        d.vkExtensions.deviceExtensions = h.deviceExtensions, d.vkExtensions.deviceExtensionNum = h.deviceExtensionCount;
        d.callbackInterface.MessageCallback = Message;
        d.enableNRIValidation = EnvFlag("NFRUX_VALIDATE", false);
        if (!NRI_OK(nri::nriCreateDeviceFromVKDevice(d, dev)) || !Init()) return false;
        
        c.dot4 = (h.mlFeatures & ffx_ml::FEATURE_INTEGER_DOT_PRODUCT) != 0, c.cooperativeMatrix = (h.mlFeatures & ffx_ml::FEATURE_COOPERATIVE_MATRIX) != 0;
        Log("Vulkan (NRI): %s, dot4 %d, cooperative matrix %d", c.name.c_str(), c.dot4, c.cooperativeMatrix);
        return true;
    }

    std::unique_ptr<TexN> MakeTex(nri::Format f, uint32_t w, uint32_t h, bool storage)
    {
        auto t = std::make_unique<TexN>();
        t->dev = this, t->width = w, t->height = h;
        nri::TextureDesc d = {};
        d.type = nri::TextureType::TEXTURE_2D, d.format = f, d.width = nri::Dim_t(w), d.height = nri::Dim_t(h), d.depth = d.mipNum = d.layerNum = 1, d.sampleNum = 1;
        d.usage = nri::TextureUsageBits::SHADER_RESOURCE;
        const bool uav = storage && (uint32_t(n.GetFormatSupport(*dev, f)) & uint32_t(nri::FormatSupportBits::STORAGE_TEXTURE)) != 0;
        if (uav) d.usage |= nri::TextureUsageBits::SHADER_RESOURCE_STORAGE;
        if (!NRI_OK(n.CreateCommittedTexture(*dev, nri::MemoryLocation::DEVICE, 0.0f, d, t->tex))) return nullptr;
        if (Env("NFRUX_TEST_NAMES")) n.SetDebugName(t->tex, "NFRUX texture");
        return Views(*t, f, uav) ? std::move(t) : nullptr;
    }
    bool Views(TexN& t, nri::Format f, bool storage)
    {
        nri::TextureViewDesc v = {};
        v.texture = t.tex, v.type = nri::TextureView::TEXTURE, v.format = f, v.mipNum = v.layerNum = 1, v.planes = t.planes;
        if (!NRI_OK(n.CreateTextureView(v, t.srv))) return false;
        v.type = nri::TextureView::STORAGE_TEXTURE;
        return !storage || NRI_OK(n.CreateTextureView(v, t.uav));
    }
    std::unique_ptr<BufN> MakeBuf(uint64_t size)
    {
        auto b = std::make_unique<BufN>();
        b->dev = this, b->size = (size + 15) & ~15ull;
        b->buf = Committed(b->size, nri::MemoryLocation::DEVICE, nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE, true);
        nri::BufferViewDesc v = {};
        v.buffer = b->buf, v.type = nri::BufferView::BYTE_ADDRESS_BUFFER, v.size = b->size;
        if (!b->buf || !NRI_OK(n.CreateBufferView(v, b->srv))) return nullptr;
        v.type = nri::BufferView::STORAGE_BYTE_ADDRESS_BUFFER;
        return NRI_OK(n.CreateBufferView(v, b->uav)) ? std::move(b) : nullptr;
    }
    std::unique_ptr<BufN> WrapBuf(const CudaVulkanGraph::Buffer& x, uint64_t size)
    {
        auto b = std::make_unique<BufN>();
        b->dev = this, b->size = (size + 15) & ~15ull;
        nri::BufferVKDesc d = {};
        d.vkBuffer = uint64_t(x.buffer), d.size = b->size, d.structureStride = 4;
        if (!x.buffer || !NRI_OK(nvk.CreateBufferVK(*dev, d, b->buf))) return nullptr;
        nri::BufferViewDesc v = {};
        v.buffer = b->buf, v.type = nri::BufferView::BYTE_ADDRESS_BUFFER, v.size = b->size;
        if (!NRI_OK(n.CreateBufferView(v, b->srv))) return nullptr;
        v.type = nri::BufferView::STORAGE_BYTE_ADDRESS_BUFFER;
        return NRI_OK(n.CreateBufferView(v, b->uav)) ? std::move(b) : nullptr;
    }
    std::unique_ptr<Texture> CreateTexture(Format f, uint32_t w, uint32_t h, bool) override
    {
        auto t = MakeTex(ToNri(f), w, h, true);
        if (t) t->format = f;
        return t;
    }
    uint32_t Code(const ExternalImage& e) const { return e.format || isVk ? e.format : uint32_t(static_cast<ID3D12Resource*>(e.handle)->GetDesc().Format); }
    bool IsSrgb(const ExternalImage& e) const
    {
        const uint32_t f = Code(e);
        return isVk ? f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB : f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    }
    // Values as stored: sRGB and typeless as their UNORM / FLOAT sibling.
    nri::Format HostFormat(const ExternalImage& e)
    {
        const uint32_t f = Code(e);
        if (isVk)
            return f == VK_FORMAT_R8G8B8A8_SRGB ? nri::Format::RGBA8_UNORM : f == VK_FORMAT_B8G8R8A8_SRGB ? nri::Format::BGRA8_UNORM : nri::nriConvertVKFormatToNRI(f);
        switch (f)
        {
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: case DXGI_FORMAT_R8G8B8A8_TYPELESS: return nri::Format::RGBA8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: case DXGI_FORMAT_B8G8R8A8_TYPELESS: return nri::Format::BGRA8_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return nri::Format::R10_G10_B10_A2_UNORM;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return nri::Format::RGBA16_SFLOAT;
        default: return nri::nriConvertDXGIFormatToNRI(f);
        }
    }
    // sRGB images cannot always be viewed as UNORM (typed on D3D12, not MUTABLE on Vulkan): copied raw into one that is.
    struct Scratch { std::unique_ptr<TexN> t; nri::Format f; uint32_t last; uint64_t of; };
    std::vector<Scratch> scratch;
    Texture* Stored(const ExternalImage& e)
    {
        for (auto& s : scratch) if (s.last == next && s.of == uint64_t(e.handle)) return s.t.get();
        TexN* src = static_cast<TexN*>(WrapRaw(e, false));
        if (!src) return nullptr;
        const nri::Format f = HostFormat(e);
        Scratch* use = nullptr;
        for (auto& s : scratch) if (s.last != next && s.f == f && s.t->width == src->width && s.t->height == src->height) { use = &s; break; }
        if (!use)
        {
            for (size_t i = 0; i < scratch.size();)  // not used for longer than anything can be in flight
                if (next - scratch[i].last > 256) scratch.erase(scratch.begin() + i); else ++i;
            auto t = MakeTex(f, src->width, src->height, false);
            if (!t) return nullptr;
            t->format = Format::Native;
            scratch.push_back({std::move(t), f, 0, 0});
            use = &scratch.back();
        }
        use->last = next, use->of = uint64_t(e.handle);
        Copy(*src, 0, 0, *use->t, 0, 0, src->width, src->height);
        return use->t.get();
    }
    std::unique_ptr<Texture> CreateTextureLike(const ExternalImage& e, uint32_t w, uint32_t h) override
    {
        auto t = MakeTex(HostFormat(e), w, h, false);
        if (t) t->format = Format::Native;
        return t;
    }
    std::unique_ptr<Buffer> CreateBuffer(uint64_t size) override { return MakeBuf(size); }

    std::unique_ptr<Pass> CreatePass(const Shader& s) override
    {
        auto p = std::make_unique<PassN>();
        p->dev = this, p->name = s.name;
        p->bindings.assign(s.bindings, s.bindings + s.bindingCount);
        std::vector<nri::DescriptorRangeDesc> ranges;
        for (const ShaderBinding& b : p->bindings)
        {
            nri::DescriptorRangeDesc r = {};
            r.baseRegisterIndex = b.binding, r.descriptorNum = 1, r.shaderStages = nri::StageBits::COMPUTE_SHADER;
            switch (b.kind)
            {
            case BindKind::Cbv: r.descriptorType = nri::DescriptorType::CONSTANT_BUFFER; break;
            case BindKind::SrvTexture: r.descriptorType = nri::DescriptorType::TEXTURE; break;
            case BindKind::UavTexture: r.descriptorType = nri::DescriptorType::STORAGE_TEXTURE; break;
            case BindKind::SrvBuffer: r.descriptorType = nri::DescriptorType::STRUCTURED_BUFFER; break;
            case BindKind::UavBuffer: r.descriptorType = nri::DescriptorType::STORAGE_STRUCTURED_BUFFER; break;
            case BindKind::Sampler: r.descriptorType = nri::DescriptorType::SAMPLER; break;
            }
            ranges.push_back(r);
        }
        nri::DescriptorSetDesc sd = {0, ranges.data(), uint32_t(ranges.size())};
        nri::PipelineLayoutDesc ld = {};
        ld.rootRegisterSpace = 1, ld.descriptorSets = &sd, ld.descriptorSetNum = 1, ld.shaderStages = nri::StageBits::COMPUTE_SHADER;
        nri::ComputePipelineDesc cd = {};
        cd.shader.stage = nri::StageBits::COMPUTE_SHADER, cd.shader.entryPointName = "main";
        cd.shader.bytecode = isVk ? s.spirv : s.dxil, cd.shader.size = isVk ? s.spirvBytes : s.dxilBytes;
        if (!NRI_OK(n.CreatePipelineLayout(*dev, ld, p->layout))) return Log("%s: layout failed", s.name), nullptr;
        cd.pipelineLayout = p->layout;
        if (!NRI_OK(n.CreateComputePipeline(*dev, cd, p->pipeline))) return Log("%s: pipeline failed", s.name), nullptr;
        if (Env("NFRUX_TEST_NAMES")) n.SetDebugName(p->pipeline, "NFRUX pipeline");
        return p;
    }

    std::mutex making;
    bool        vkCuda = false;  // Vulkan: the last network made runs CUDA kernels
    std::string cudaMarker;
    bool lastFloat = false, noFloats = false;  // float cooperative matrices: the last made, refused
    int rung = -1, topRung = 0;  // 0 cooperative matrices / CUDA, 1 dot4, 2 integers: the last made, the highest allowed
    bool DropRung() override
    {
        std::lock_guard<std::mutex> lock(making);
        if (rung < 0 || rung >= 2) return false;
        if (rung == 0 && !isVk) cudaState = -1;  // its marker stays: not "ok"
        if (isVk && rung == 0 && vkCuda) { cudaState = -1, vkCuda = false, cudaMarker.clear(); return true; }  // its marker stays; cooperative matrices next
        if (isVk) vk.marker.clear();
        if (isVk && rung == 0 && lastFloat && !noFloats) { noFloats = true; return true; }  // the int8 matrices next, if there are any
        topRung = rung + 1;
        return true;
    }
    std::unique_ptr<Network> CreateNetwork(uint32_t w, uint32_t h, std::string* error) override
    {
        std::lock_guard<std::mutex> lock(making);  // worker threads
        auto net = std::make_unique<NetN>();
        const ModelBindings mb = GetModelBindings();
        const char* ml = Env("NFRUX_ML");
        std::string what;
        if (isVk)
        {
            const ffx_ml::DeviceContext dc = {vk.instance, vk.physicalDevice, vk.device, vk.gipa, vk.gdpa, vk.mlFeatures};
            ffx_ml::Path path = ffx_ml::Path::Auto;
            // Cooperative matrices: fp16 first on NVIDIA, int8 first elsewhere (Intel XMX, AMD WMMA); fp16 and bf16 are those whose sums are the same
            // integers while below 2^24 (bound computed here from the weights; tested at the extremes; and the self-check compares).
            ffx_ml::MatType type = ffx_ml::MatType::None;
            if (ml)
            {
                const std::string m = ml;
                if (m == "coopmat") path = ffx_ml::Path::CoopMat;
                else if (m == "coopmat_int8") path = ffx_ml::Path::CoopMat, type = ffx_ml::MatType::Int8;
                else if (m == "coopmat_fp16") path = ffx_ml::Path::CoopMat, type = ffx_ml::MatType::Fp16;
                else if (m == "coopmat_bf16") path = ffx_ml::Path::CoopMat, type = ffx_ml::MatType::Bf16;
                else if (m == "dot4") path = ffx_ml::Path::Dot4;
                else if (m == "scalar") path = ffx_ml::Path::Scalar;
                else if (m == "cuda") {}
                else { if (error) *error = "NFRUX_ML: unknown path"; return nullptr; }
            }
            // CUDA kernels first where they can be launched (NVIDIA tensor cores): the same as on D3D12.
            const bool forced = ml && !strcmp(ml, "cuda") && !topRung && cudaState >= 0;
            if (forced || (!ml && !topRung && !noFloats && cudaState >= 0 && c.vendorId == uint32_t(nri::Vendor::NVIDIA) && (vk.mlFeatures & kFeatureCudaLaunch)))
            {
                std::string why;
                if (cudaState == 0)
                {
                    char b[48];
                    snprintf(b, sizeof(b), "vulkan_cuda_%04x", c.deviceId);
                    cudaMarker = b, cudaState = forced || MarkerBegin(cudaMarker) ? 1 : -1;
                }
                if (cudaState > 0) net->cudaVk = CudaVulkanGraph::Create(dc, DescribeModel(w, h), &why);
                if (!net->cudaVk && cudaState > 0) Log("CUDA network not used: %s", why.c_str()), MarkerOk(cudaMarker), cudaState = -1;
                if (forced && !net->cudaVk) { if (error) *error = "NFRUX_ML=cuda: " + why; return nullptr; }
            }
            if (ml && !strcmp(ml, "cuda") && !net->cudaVk && cudaState < 0 && !topRung && rung < 0) { if (error) *error = "NFRUX_ML=cuda: unavailable"; return nullptr; }
            ffx_ml::Selection sel = ffx_ml::SelectPath(dc, path, type);
            if (topRung == 1) sel = ffx_ml::SelectPath(dc, ffx_ml::Path::Dot4, ffx_ml::MatType::None);
            if (topRung == 2 || (topRung == 1 && sel.path != ffx_ml::Path::Dot4)) sel = ffx_ml::SelectPath(dc, ffx_ml::Path::Scalar, ffx_ml::MatType::None);
            if (sel.path == ffx_ml::Path::CoopMat && sel.matType != ffx_ml::MatType::Int8 && ffx_ml::AccumulatorBound(DescribeModel(w, h)) >= (1 << 24))
                sel = ffx_ml::SelectPath(dc, ffx_ml::Path::Auto, ffx_ml::MatType::Int8);
            if (noFloats && topRung == 0 && sel.path == ffx_ml::Path::CoopMat && sel.matType != ffx_ml::MatType::Int8)
                sel = ffx_ml::SelectPath(dc, ffx_ml::Path::Auto, ffx_ml::MatType::Int8);
            if (sel.path == ffx_ml::Path::CoopMat) lastFloat = sel.matType != ffx_ml::MatType::Int8;
            else lastFloat = false;
            rung = sel.path == ffx_ml::Path::CoopMat ? 0 : sel.path == ffx_ml::Path::Dot4 ? 1 : 2;
            sel.rescaleFloat = false;  // exact
            if (path != ffx_ml::Path::Auto && !topRung && sel.path != path) { if (error) *error = "forced network path unavailable"; return nullptr; }
            if (net->cudaVk)
            {
                lastFloat = false, rung = 0, what = "CUDA kernels, int8 tensor cores (Vulkan)";
                for (uint32_t i = 0; i < 2; ++i) net->order[i] = net->cudaVk->outputBinding(i) == mb.params ? 0 : 1;
            }
            else
            {
                net->vk = ffx_ml::Graph::Create(dc, sel, DescribeModel(w, h), error);
                if (!net->vk) return nullptr;
                what = sel.Describe();
            }
            vkCuda = net->cudaVk != nullptr;
        }
        else
        {
            if (ml && strcmp(ml, "scalar") && strcmp(ml, "cuda") && (strcmp(ml, "dot4") || !c.dot4)) { if (error) *error = "NFRUX_ML: path unavailable on D3D12"; return nullptr; }
            const bool forced = ml && !strcmp(ml, "cuda") && !topRung;
            if (forced || (!ml && !topRung && c.vendorId == uint32_t(nri::Vendor::NVIDIA)))
            {
                std::string why;
                if (cudaState == 0) cudaState = forced || MarkerBegin(Rung()) ? 1 : -1;
                if (cudaState > 0) net->cuda = CudaGraph::Create(d3d, DescribeModel(w, h), &why);
                if (!net->cuda && cudaState > 0) Log("CUDA network not used: %s", why.c_str()), MarkerOk(Rung()), cudaState = -1;
                if (forced && !net->cuda) { if (error) *error = "NFRUX_ML=cuda: " + why; return nullptr; }
            }
            const bool dot4 = c.dot4 && topRung < 2 && !(ml && !strcmp(ml, "scalar"));
            if (!net->cuda) net->d3d = D3D12Graph::Create(d3d, DescribeModel(w, h), dot4, error);
            if (!net->d3d && !net->cuda) return nullptr;
            rung = net->cuda ? 0 : dot4 ? 1 : 2;
            what = net->cuda ? "CUDA kernels, int8 tensor cores" : dot4 ? "shader model 6.4 packed dot product" : "shader model 6.0 integers";
            for (uint32_t i = 0; i < 2; ++i) net->order[i] = (net->cuda ? net->cuda->outputBinding(i) : net->d3d->outputBinding(i)) == mb.params ? 0 : 1;
        }
        net->pitch = isVk ? w / 2 * 4 : (w / 2 * 4 + 255) & ~255u;
        auto make = [&](uint32_t i, uint64_t size) {
            if (!net->cudaVk) return MakeBuf(size);
            net->cudaBuf[i] = net->cudaVk->MakeBuffer(size, false);
            return WrapBuf(net->cudaBuf[i], size);
        };
        net->in    = make(0, uint64_t(w) * h * kInputChannels);
        bool ok = net->in != nullptr;
        for (uint32_t i = 0; i < 2; ++i)
        {
            net->out[i] = make(1 + i, uint64_t(net->pitch) * (h / 2));
            net->tex[i] = MakeTex(nri::Format::RGBA8_SNORM, w / 2, h / 2, false);
            ok = ok && net->out[i] && net->tex[i];
            if (ok) net->tex[i]->format = Format::RGBA8Snorm;
        }
        if (!ok) { if (error) *error = "out of memory"; return nullptr; }
        net->input = net->in.get(), net->params = net->tex[0].get(), net->gate = net->tex[1].get();
        static std::mutex logMutex;
        static std::string logged;
        std::lock_guard<std::mutex> logLock(logMutex);
        if (logged != what) Log("network %ux%u: %s", w, h, what.c_str()), logged = what;
        return net;
    }

    // --- recording
    bool Done(uint32_t s, uint32_t v) const { return tokens[s] == v || (slotValue[s] != v); }  // a slot is only taken again once its recording finished
    void Poll()
    {
        for (auto it = retired.begin(); it != retired.end();)
            // Our work in the host's command buffer has finished, but the buffer may still count as pending (its tail, its fence): kept 16 recordings more.
            if (next - it->value >= 16 && Done(it->slot, it->value)) { if (it->cmd) n.DestroyCommandBuffer(it->cmd); ++finished, it = retired.erase(it); } else ++it;
        for (auto it = dying.begin(); it != dying.end();)
            if (next - it->first >= 16) n.DestroyBuffer(it->second), it = dying.erase(it); else ++it;
        for (auto it = reads.begin(); it != reads.end();)
            if (Done(it->slot, it->value)) { Finish(*it); it = reads.erase(it); } else ++it;
    }
    void Finish(Read& r)
    {
        if (auto* p = static_cast<uint8_t*>(n.MapBuffer(*r.buf, 0, nri::WHOLE_SIZE)))
        {
            r.out->resize(size_t(r.row) * r.rows);
            for (uint32_t y = 0; y < r.rows; ++y) memcpy(r.out->data() + size_t(y) * r.row, p + size_t(y) * r.pitch, r.row);
            n.UnmapBuffer(*r.buf);
        }
        if (r.size <= 4096 && freeReads.size() < 16) freeReads.push_back({r.size, r.buf});
        else dying.push_back({next, r.buf});  // the token is written before the host's list ends: not free yet
    }
    std::atomic<int> cudaState = 0;  // 0 not tried, 1 in use, -1 not
    uint32_t    finished = 0;
    bool        marked = false, ran = false;
    uint32_t    firstRun = 0;  // finished when the network first ran
    std::string Rung() const { char b[64]; snprintf(b, sizeof(b), "cuda_%04x_%08lx%08lx", c.deviceId, (unsigned long)d3d->GetAdapterLuid().HighPart, (unsigned long)d3d->GetAdapterLuid().LowPart); return b; }
    bool Lost() const { return d3d && FAILED(d3d->GetDeviceRemovedReason()); }
    // A slot whose last recording has finished (its own value in its own token). Full of unfinished ones: wait once, 2 s, for any.
    bool TakeSlot()
    {
        for (int ms = 0; ms < (stalled ? 1 : 2000) && !Lost(); ++ms)
        {
            for (uint32_t i = 0; i < kSlots; ++i)
            {
                const uint32_t s = (next + i) % kSlots;
                if (slotValue[s] && tokens[s] != slotValue[s]) continue;
                slot = s, stalled = false;
                slotValue[s] = tokenUpMap[s] = ++next;
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!stalled) Log("no recording finished in 2 s: frame not generated");
        stalled = true;  // no waiting again until one does
        return false;
    }
    bool BeginFrame(void* commands) override
    {
        Poll();
        // Recordings from well after the network first ran have completed (those before it say nothing).
        if (cudaState == 1 && ran && finished >= firstRun + 40 && !marked) marked = true, MarkerOk(isVk ? cudaMarker : Rung());
        if (!vk.marker.empty() && ran && finished >= firstRun + 40) MarkerOk(vk.marker), vk.marker.clear();
        nativeCmd = commands, t_recording = true;
        if (!isVk) LogBarriers(static_cast<ID3D12GraphicsCommandList*>(commands));
        if (isVk)
        {
            nri::CommandBufferVKDesc d = {};
            d.vkCommandBuffer = commands, d.queueType = nri::QueueType::COMPUTE;
            if (!NRI_OK(nvk.CreateCommandBufferVK(*dev, d, cmd))) return false;
        }
        else
        {
            nri::CommandBufferD3D12Desc d = {};
            d.d3d12CommandList = static_cast<ID3D12GraphicsCommandList*>(commands);
            if (!NRI_OK(nd3.CreateCommandBufferD3D12(*dev, d, cmd))) return false;
        }
        if (!TakeSlot()) { n.DestroyCommandBuffer(cmd), cmd = nullptr; return false; }
        usedConstants    = 0;
        n.ResetDescriptorPool(*pools[slot]);
        n.CmdSetDescriptorPool(*cmd, *pools[slot]);
        return true;
    }
    void EndFrame() override
    {
        for (size_t i = 0; i < wrapped.size(); ++i) To(*wrapped[i], wrappedHome[i]);
        Flush();
        if (timed && markCount) { Mark("end"); n.CmdCopyQueries(*cmd, *marks, slot * kMarks, markCount, *markBack, uint64_t(slot) * kMarks * 8); }
        markCount = 0;
        if (!hostBack.empty()) static_cast<ID3D12GraphicsCommandList*>(nativeCmd)->ResourceBarrier(UINT(hostBack.size()), hostBack.data()), hostBack.clear();
        // Everything recorded before finished: then this recording's own value into its own slot, visible to the host.
        if (isVk)
        {
            auto* vc = static_cast<VkCommandBuffer>(nativeCmd);
            const auto barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(vk.gdpa(vk.device, "vkCmdPipelineBarrier"));
            const auto fill    = reinterpret_cast<PFN_vkCmdFillBuffer>(vk.gdpa(vk.device, "vkCmdFillBuffer"));
            VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT};
            barrier(vc, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
            fill(vc, VkBuffer(n.GetBufferNativeObject(tokenBack)), slot * 4, 4, slotValue[slot]);
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier(vc, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
        }
        else
        {
            ID3D12GraphicsCommandList2* l2 = nullptr;
            auto* r = reinterpret_cast<ID3D12Resource*>(n.GetBufferNativeObject(tokenBack));
            if (SUCCEEDED(static_cast<ID3D12GraphicsCommandList*>(nativeCmd)->QueryInterface(__uuidof(ID3D12GraphicsCommandList2), reinterpret_cast<void**>(&l2))))
            {
                const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p = {r->GetGPUVirtualAddress() + slot * 4, slotValue[slot]};
                const D3D12_WRITEBUFFERIMMEDIATE_MODE      m = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
                l2->WriteBufferImmediate(1, &p, &m);
                l2->Release();
            }
            for (const BufN* b : used) b->state = {nri::AccessBits::NONE, nri::StageBits::ALL};  // buffers decay to COMMON after the list
        }
        used.clear();
        retired.push_back({std::move(wrapped), cmd, slot, slotValue[slot]});
        wrapped.clear(), wrappedHome.clear();
        cmd = nullptr, t_recording = false;
    }
    Texture* Wrap(const ExternalImage& e, bool output) override
    {
        if (!output && IsSrgb(e)) return Stored(e);
        return WrapRaw(e, output);
    }
    Texture* WrapRaw(const ExternalImage& e, bool output)
    {
        for (auto& t : wrapped) if (t->native == uint64_t(e.handle)) return t.get();
        auto t = std::make_unique<TexN>();
        t->dev = this, t->native = uint64_t(e.handle), t->width = e.width, t->height = e.height, t->format = Format::Native;
        const nri::Format f = HostFormat(e);
        if (isVk)
        {
            nri::TextureVKDesc d = {};
            d.vkImage = uint64_t(e.handle), d.vkFormat = int32_t(e.format), d.vkImageType = VK_IMAGE_TYPE_2D;
            d.vkImageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            d.width = nri::Dim_t(e.width), d.height = nri::Dim_t(e.height), d.depth = d.mipNum = d.layerNum = 1, d.sampleNum = 1;
            if (!NRI_OK(nvk.CreateTextureVK(*dev, d, t->tex))) return nullptr;
        }
        else
        {
            auto* r = static_cast<ID3D12Resource*>(e.handle);
            const D3D12_RESOURCE_DESC ds = r->GetDesc();
            t->width = uint32_t(ds.Width), t->height = ds.Height;
            nri::TextureD3D12Desc d = {r, int32_t(nri::nriConvertNRIFormatToDXGI(f))};
            if (!NRI_OK(nd3.CreateTextureD3D12(*dev, d, t->tex))) return nullptr;
        }
        if (IsDepth(f)) t->planes = nri::PlaneBits::DEPTH;
        if (!IsSrgb(e) && !Views(*t, f, false)) return nullptr;
        const nri::AccessLayoutStage& home = isVk ? (!output && e.state == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? kHostVkRead : kHostVk) : output ? kUav : e.state == D3D12_RESOURCE_STATE_COPY_DEST ? kDst : kSrv;
        if (!isVk && !output && e.state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE && e.state != D3D12_RESOURCE_STATE_COPY_DEST)
        {
            // Another state: to the guide's and back, where this list can. A COMPUTE list cannot name states with pixel or attachment bits; one that has NON_PIXEL is readable as it is.
            auto* list = static_cast<ID3D12GraphicsCommandList*>(nativeCmd);
            const uint32_t compute = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_COPY_DEST | D3D12_RESOURCE_STATE_COPY_SOURCE;
            if (list->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT || (e.state & ~compute) == 0)
            {
                D3D12_RESOURCE_BARRIER b = {};
                b.Transition = {static_cast<ID3D12Resource*>(e.handle), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATES(e.state), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
                list->ResourceBarrier(1, &b);
                std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
                hostBack.push_back(b);
            }
        }
        t->state = home;
        wrapped.push_back(std::move(t)), wrappedHome.push_back(home);
        return wrapped.back().get();
    }
    void To(const TexN& t, const nri::AccessLayoutStage& s)
    {
        if (Same(t.state, s) && !Same(s, kUav)) return;
        nri::TextureBarrierDesc b = {};
        b.texture = t.tex, b.before = t.state, b.after = s, b.mipNum = nri::REMAINING, b.layerNum = nri::REMAINING;
        tb.push_back(b);
        t.state = s;
    }
    void To(const BufN& t, nri::AccessBits a, nri::StageBits s)
    {
        if (t.state.access == a && a != nri::AccessBits::SHADER_RESOURCE_STORAGE) return;
        if (std::find(used.begin(), used.end(), &t) == used.end()) used.push_back(&t);
        nri::BufferBarrierDesc b = {};
        b.buffer = t.buf, b.before = t.state, b.after = {a, s};
        bb.push_back(b);
        t.state = b.after;
    }
    void Flush()
    {
        if (tb.empty() && bb.empty()) return;
        nri::BarrierDesc b = {};
        static const bool full = EnvFlag("NFRUX_FULL_BARRIERS", false);  // diagnosis
        nri::GlobalBarrierDesc g = {{nri::AccessBits::SHADER_RESOURCE_STORAGE | nri::AccessBits::COPY_DESTINATION, nri::StageBits::ALL}, {nri::AccessBits::SHADER_RESOURCE_STORAGE | nri::AccessBits::SHADER_RESOURCE | nri::AccessBits::COPY_SOURCE | nri::AccessBits::COPY_DESTINATION, nri::StageBits::ALL}};
        if (full) b.globals = &g, b.globalNum = 1;
        b.textures = tb.data(), b.textureNum = uint32_t(tb.size()), b.buffers = bb.data(), b.bufferNum = uint32_t(bb.size());
        n.CmdBarrier(*cmd, b);
        tb.clear(), bb.clear();
    }
    void Copy(const Texture& s, uint32_t sx, uint32_t sy, const Texture& d, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h)
    {
        auto &a = static_cast<const TexN&>(s), &b = static_cast<const TexN&>(d);
        To(a, kSrc), To(b, kDst), Flush();
        nri::TextureRegionDesc rs = {}, rd = {};
        rs.x = nri::Dim_t(sx), rs.y = nri::Dim_t(sy), rd.x = nri::Dim_t(dx), rd.y = nri::Dim_t(dy);
        rs.width = rd.width = nri::Dim_t(w), rs.height = rd.height = nri::Dim_t(h), rs.depth = rd.depth = 1;
        n.CmdCopyTexture(*cmd, *b.tex, &rd, *a.tex, &rs);
    }
    void CopyTexture(const Texture& s, const Texture& d, uint32_t x, uint32_t y) override
    {
        Copy(s, 0, 0, d, x, y, std::min(s.width, d.width - x), std::min(s.height, d.height - y));
    }
    void CopyRect(const Texture& s, uint32_t sx, uint32_t sy, const Texture& d, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h) override
    {
        if (sx >= s.width || sy >= s.height || dx >= d.width || dy >= d.height) return;
        Copy(s, sx, sy, d, dx, dy, std::min({w, s.width - sx, d.width - dx}), std::min({h, s.height - sy, d.height - dy}));
    }
    // A word into one of the host's buffers (D3D12: in COPY_DEST, as Streamline gives it).
    void WriteWord(void* buffer, uint32_t value) override
    {
        Flush();
        if (isVk)
        {
            const auto vc = static_cast<VkCommandBuffer>(nativeCmd);
            const auto barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(vk.gdpa(vk.device, "vkCmdPipelineBarrier"));
            const auto fill = reinterpret_cast<PFN_vkCmdFillBuffer>(vk.gdpa(vk.device, "vkCmdFillBuffer"));
            VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
            barrier(vc, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
            fill(vc, reinterpret_cast<VkBuffer>(buffer), 0, 4, value);
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            barrier(vc, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
            return;
        }
        auto* list = static_cast<ID3D12GraphicsCommandList*>(nativeCmd);
        auto* res = static_cast<ID3D12Resource*>(buffer);
        ID3D12GraphicsCommandList2* l2 = nullptr;
        if (FAILED(list->QueryInterface(__uuidof(ID3D12GraphicsCommandList2), reinterpret_cast<void**>(&l2)))) return;
        const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p = {res->GetGPUVirtualAddress(), value};
        l2->WriteBufferImmediate(1, &p, nullptr);
        l2->Release();
    }
    void CopyRegion(const Texture& s, uint32_t x, uint32_t y, const Texture& d) override { Copy(s, x, y, d, 0, 0, d.width, d.height); }
    // A host buffer of 0xff bytes: 1.0 in R8, ~0 in the scatter keys.
    bool Ones(uint64_t size)
    {
        if (ones && onesSize >= size) return true;
        if (ones) old.push_back(ones);  // maybe in flight: freed with the device
        ones = Committed(size, nri::MemoryLocation::HOST_UPLOAD, nri::BufferUsageBits::NONE);  // an older, smaller one is in flight: kept
        if (!ones) return false;
        onesSize = size;
        if (void* p = n.MapBuffer(*ones, 0, nri::WHOLE_SIZE)) memset(p, 0xff, size), n.UnmapBuffer(*ones);
        return true;
    }
    void Clear(const Texture& t, float) override
    {
        auto& a = static_cast<const TexN&>(t);
        const uint32_t pitch = (t.width * TexelSize(t.format) + 255) & ~255u;
        if (!Ones(uint64_t(pitch) * t.height)) return;
        To(a, kDst), Flush();
        nri::TextureDataLayoutDesc l = {0, pitch, pitch * t.height};
        nri::TextureRegionDesc r = {};
        r.width = nri::WHOLE_SIZE, r.height = nri::WHOLE_SIZE, r.depth = 1;
        n.CmdUploadBufferToTexture(*cmd, *a.tex, r, *ones, l);
    }
    void Fill(const Buffer& b, uint32_t) override
    {
        auto& a = static_cast<const BufN&>(b);
        if (!Ones(a.size)) return;
        To(a, nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY), Flush();
        n.CmdCopyBuffer(*cmd, *a.buf, 0, *ones, 0, a.size);
    }
    void RunNetworks(Network* const* nets, uint32_t count) override
    {
        if (!ran) ran = true, firstRun = finished;
        for (uint32_t i = 0; i < count; ++i)
        {
            auto& x = static_cast<NetN&>(*nets[i]);
            To(*x.in, x.cuda ? nri::AccessBits::SHADER_RESOURCE_STORAGE : nri::AccessBits::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER);
            for (auto& o : x.out) To(*o, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER);
        }
        Flush();
        if (isVk)
        {
            std::vector<CudaVulkanGraph*> cg;
            std::vector<uint64_t>         ci;
            std::vector<std::array<uint64_t, 2>> co;
            uint32_t pitch = 0;
            for (uint32_t i = 0; i < count; ++i)
            {
                auto& x = static_cast<NetN&>(*nets[i]);
                if (!x.cudaVk) continue;
                cg.push_back(x.cudaVk.get()), ci.push_back(x.cudaBuf[0].address), co.push_back({x.cudaBuf[1 + x.order[0]].address, x.cudaBuf[1 + x.order[1]].address}), pitch = x.pitch;
            }
            if (!cg.empty()) CudaVulkanGraph::Record(static_cast<VkCommandBuffer>(nativeCmd), cg.data(), ci.data(), reinterpret_cast<const uint64_t(*)[2]>(co.data()), uint32_t(cg.size()), pitch);
            std::vector<Network*> rest;
            for (uint32_t i = 0; i < count; ++i) if (!static_cast<NetN&>(*nets[i]).cudaVk) rest.push_back(nets[i]);
            const uint32_t all = count;
            Network* const* allNets = nets;
            nets = rest.data(), count = uint32_t(rest.size());
            if (count)
            {
            const ModelBindings mb = GetModelBindings();
            std::vector<std::array<ffx_ml::ExternalTensor, 3>> t(count);
            std::vector<ffx_ml::Graph*> g(count);
            std::vector<const ffx_ml::ExternalTensor*> sets(count);
            std::vector<uint32_t> counts(count, 3);
            for (uint32_t i = 0; i < count; ++i)
            {
                auto& x = static_cast<NetN&>(*nets[i]);
                auto  b = [&](const BufN& f) { return VkBuffer(n.GetBufferNativeObject(f.buf)); };
                t[i][0] = {mb.input, b(*x.in), 0}, t[i][1] = {mb.params, b(*x.out[0]), 0}, t[i][2] = {mb.gate, b(*x.out[1]), 0};
                g[i] = x.vk.get(), sets[i] = t[i].data();
            }
            if (count == 1) g[0]->Record(static_cast<VkCommandBuffer>(nativeCmd), sets[0], 3);
            else ffx_ml::Graph::RecordInterleaved(static_cast<VkCommandBuffer>(nativeCmd), g.data(), sets.data(), counts.data(), count);
            }
            nets = allNets, count = all;
        }
        else
        {
            std::vector<CudaGraph*> cg;
            std::vector<D3D12_GPU_VIRTUAL_ADDRESS> ci;
            std::vector<std::array<ID3D12Resource*, 2>> co;
            uint32_t pitch = 0;
            for (uint32_t i = 0; i < count; ++i)
            {
                auto& x = static_cast<NetN&>(*nets[i]);
                if (!x.cuda) continue;
                auto r = [&](const BufN& f) { return reinterpret_cast<ID3D12Resource*>(n.GetBufferNativeObject(f.buf)); };
                cg.push_back(x.cuda.get()), ci.push_back(r(*x.in)->GetGPUVirtualAddress()), co.push_back({r(*x.out[x.order[0]]), r(*x.out[x.order[1]])}), pitch = x.pitch;
            }
            std::vector<ID3D12Resource* const*> op;
            for (auto& o : co) op.push_back(o.data());
            if (!cg.empty()) CudaGraph::Record(static_cast<ID3D12GraphicsCommandList*>(nativeCmd), cg.data(), ci.data(), op.data(), uint32_t(cg.size()), pitch);
            for (uint32_t i = 0; i < count; ++i)
            {
                auto& x = static_cast<NetN&>(*nets[i]);
                if (x.cuda) continue;
                auto  r = [&](const BufN& f) { return reinterpret_cast<ID3D12Resource*>(n.GetBufferNativeObject(f.buf)); };
                ID3D12Resource* o[2] = {r(*x.out[x.order[0]]), r(*x.out[x.order[1]])};
                x.d3d->Record(static_cast<ID3D12GraphicsCommandList*>(nativeCmd), r(*x.in)->GetGPUVirtualAddress(), o, x.pitch);
            }
        }
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t k = 0; k < 2; ++k)
            {
                auto& x = static_cast<NetN&>(*nets[i]);
                To(*x.out[k], nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY), To(*x.tex[k], kDst);
            }
        Flush();
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t k = 0; k < 2; ++k)
            {
                auto& x = static_cast<NetN&>(*nets[i]);
                nri::TextureDataLayoutDesc l = {0, x.pitch, x.pitch * x.tex[k]->height};
                nri::TextureRegionDesc r = {};
                r.width = nri::WHOLE_SIZE, r.height = nri::WHOLE_SIZE, r.depth = 1;
                n.CmdUploadBufferToTexture(*cmd, *x.tex[k]->tex, r, *x.out[k]->buf, l);
            }
    }
    void ReadBack(const Texture& t, std::shared_ptr<std::vector<uint8_t>> out) override
    {
        auto& a = static_cast<const TexN&>(t);
        const uint32_t row = t.width * TexelSize(t.format), pitch = (row + 255) & ~255u;
        Read r = {nullptr, out, pitch, row, t.height, slot, slotValue[slot], uint64_t(pitch) * t.height};
        for (auto it = freeReads.begin(); it != freeReads.end(); ++it)
            if (it->first == r.size) { r.buf = it->second, freeReads.erase(it); break; }
        if (!r.buf) r.buf = Committed(r.size, nri::MemoryLocation::HOST_READBACK, nri::BufferUsageBits::NONE);
        if (!r.buf) return;
        To(a, kSrc), Flush();
        nri::TextureDataLayoutDesc l = {0, pitch, pitch * t.height};
        nri::TextureRegionDesc rg = {};
        rg.width = nri::WHOLE_SIZE, rg.height = nri::WHOLE_SIZE, rg.depth = 1;
        n.CmdReadbackTextureToBuffer(*cmd, *r.buf, l, *a.tex, rg);
        reads.push_back(r);
    }
    // NFRUX_TEST_TIMES: the GPU's time after each pass; means by pass when the device ends.
    static constexpr uint32_t kMarks = 256;
    nri::QueryPool* marks = nullptr;
    nri::Buffer*    markBack = nullptr;
    bool            timed = EnvFlag("NFRUX_TEST_TIMES", false);
    uint32_t        markCount = 0;
    std::vector<std::string> markNames[kSlots];
    uint32_t        markValue[kSlots] = {};
    std::map<std::string, std::pair<double, uint32_t>> times;
    uint32_t        timedFrames = 0;
    void Mark(const char* label) override
    {
        if (!timed || markCount >= kMarks) return;
        if (!marks)
        {
            nri::QueryPoolDesc q = {};
            q.queryType = nri::QueryType::TIMESTAMP, q.capacity = kMarks * kSlots;
            if (!NRI_OK(n.CreateQueryPool(*dev, q, marks))) { timed = false; return; }
            markBack = Committed(uint64_t(kMarks) * kSlots * 8, nri::MemoryLocation::HOST_READBACK, nri::BufferUsageBits::NONE);
        }
        if (!markCount)
        {
            if (!markNames[slot].empty() && markBack)  // the slot's last recording has finished
                if (auto* v = static_cast<const uint64_t*>(n.MapBuffer(*markBack, 0, nri::WHOLE_SIZE)))
                {
                    const uint64_t* q = v + size_t(slot) * kMarks;
                    const double hz = double(n.GetDeviceDesc(*dev).other.timestampFrequencyHz), ms = hz > 0 ? 1000.0 / hz : 0;  // (a wrapped D3D12 device has no queue to ask: 0)
                    if (markNames[slot].size() > 8 && ++timedFrames > 60)
                        for (size_t i = 1; i < markNames[slot].size(); ++i) { auto& t = times[markNames[slot][i]]; t.first += double(q[i] - q[i - 1]) * ms, ++t.second; }
                    n.UnmapBuffer(*markBack);
                }
            markNames[slot].clear();
            n.CmdResetQueries(*cmd, *marks, slot * kMarks, kMarks);
        }
        markNames[slot].push_back(label);
        n.CmdEndQuery(*cmd, *marks, slot * kMarks + markCount++);
    }
    bool WaitIdle() override
    {
        bool busy = true;
        for (int ms = 0; ms < 2000 && busy && !Lost(); ++ms)
        {
            busy = false;
            for (uint32_t s = 0; s < kSlots; ++s) busy = busy || (slotValue[s] && tokens[s] != slotValue[s]);
            if (busy) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (busy) Log("work still in flight after 2 s%s", Lost() ? " (device removed)" : "");
        Poll();
        return !busy;
    }
};

TexN::~TexN() { if (srv) dev->n.DestroyDescriptor(srv); if (uav) dev->n.DestroyDescriptor(uav); if (tex) dev->n.DestroyTexture(tex); }
BufN::~BufN() { if (srv) dev->n.DestroyDescriptor(srv); if (uav) dev->n.DestroyDescriptor(uav); if (buf) dev->n.DestroyBuffer(buf); }
PassN::~PassN() { if (pipeline) dev->n.DestroyPipeline(pipeline); if (layout) dev->n.DestroyPipelineLayout(layout); }

bool PassN::Dispatch(const Bind* binds, uint32_t count, uint32_t gx, uint32_t gy)
{
    DevN& d = *dev;
    nri::DescriptorSet* set = nullptr;
    if (!NRI_OK(d.n.AllocateDescriptorSets(*d.pools[d.slot], *layout, 0, &set, 1, 0))) return Log("%s: out of descriptor sets", name.c_str()), false;
    std::vector<nri::Descriptor*>               ds(bindings.size());
    std::vector<nri::UpdateDescriptorRangeDesc> up(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i)
    {
        const ShaderBinding& b = bindings[i];
        const Bind* f = nullptr;
        for (uint32_t k = 0; k < count && !f; ++k) if (binds[k].binding == b.binding) f = &binds[k];
        if (b.kind == BindKind::Sampler) ds[i] = b.binding >= 1000 && b.binding < 1004 ? d.samplers[b.binding - 1000] : nullptr;
        else if (!f) return Log("%s: binding %u unbound", name.c_str(), b.binding), false;
        else if (b.kind == BindKind::Cbv)
        {
            if (!f->data || f->size > kConstantStride || d.usedConstants >= kConstants) return Log("%s: constants", name.c_str()), false;
            const uint32_t k = d.slot * kConstants + d.usedConstants++;
            memcpy(d.map + size_t(k) * kConstantStride, f->data, f->size);
            ds[i] = d.cbv[k];
        }
        else if (f->buffer)
        {
            auto& x = static_cast<const BufN&>(*f->buffer);
            const bool w = b.kind == BindKind::UavBuffer;
            d.To(x, w ? nri::AccessBits::SHADER_RESOURCE_STORAGE : nri::AccessBits::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER);
            ds[i] = w ? x.uav : x.srv;
        }
        else if (f->texture)
        {
            auto& x = static_cast<const TexN&>(*f->texture);
            const bool w = b.kind == BindKind::UavTexture;
            d.To(x, w ? kUav : kSrv);
            ds[i] = w ? x.uav : x.srv;
        }
        if (!ds[i]) return Log("%s: no view for binding %u", name.c_str(), b.binding), false;
        up[i] = {set, uint32_t(i), 0, &ds[i], 1};
    }
    d.n.UpdateDescriptorRanges(up.data(), uint32_t(up.size()));
    d.Flush();
    d.n.CmdSetPipelineLayout(*d.cmd, nri::BindPoint::COMPUTE, *layout);
    d.n.CmdSetPipeline(*d.cmd, *pipeline);
    nri::SetDescriptorSetDesc sd = {0, set, nri::BindPoint::COMPUTE};
    d.n.CmdSetDescriptorSet(*d.cmd, sd);
    if (d.timed && !d.markCount) d.Mark("begin");
    d.n.CmdDispatch(*d.cmd, {gx, gy, 1});
    if (d.timed) d.Mark(name.c_str());
    return true;
}
}  // namespace

uint32_t TexelSize(Format f)
{
    switch (f)
    {
    case Format::R8Unorm: return 1;
    case Format::RGBA16Float: return 8;
    default: return 4;
    }
}
// With NFRUX_TEST_D3D12_MESSAGES: every barrier on lists of this kind, the host's and NFRUX's.
using BarrierFn = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, const D3D12_RESOURCE_BARRIER*);
static BarrierFn g_barrier;
static void STDMETHODCALLTYPE LoggedBarrier(ID3D12GraphicsCommandList* l, UINT n, const D3D12_RESOURCE_BARRIER* b)
{
    static std::atomic<int> count;
    for (UINT i = 0; i < n && count < 4000; ++i)
        if (b[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION && count++ < 4000)
        {
            wchar_t name[128] = L"?", list[128] = L"?";
            UINT s = sizeof(name) - 2;
            if (b[i].Transition.pResource) b[i].Transition.pResource->GetPrivateData(WKPDID_D3DDebugObjectNameW, &s, name);
            s = sizeof(list) - 2;
            l->GetPrivateData(WKPDID_D3DDebugObjectNameW, &s, list);
            if (!wcsstr(list, L"dlssg") && !wcsstr(name, L"PreUI") && !wcsstr(name, L"GBufferMotion") && !wcsstr(name, L"clone.tag")) { --count; continue; }
            char an[128] = "?";
            s = sizeof(an) - 1;
            if (name[0] == L'?' && b[i].Transition.pResource) b[i].Transition.pResource->GetPrivateData(WKPDID_D3DDebugObjectName, &s, an);
            Log("barrier%s %p %ls: %ls %s %x -> %x", t_recording ? " (NFRUX)" : "", (void*)l, list, name, an, b[i].Transition.StateBefore, b[i].Transition.StateAfter);
        }
    g_barrier(l, n, b);
}
void LogBarriers(ID3D12GraphicsCommandList* list)
{
    if (g_barrier || !Env("NFRUX_TEST_D3D12_MESSAGES")) return;
    void** v = *reinterpret_cast<void***>(list);
    DWORD old;
    if (!VirtualProtect(&v[26], 8, PAGE_READWRITE, &old)) return;
    g_barrier = reinterpret_cast<BarrierFn>(v[26]), v[26] = reinterpret_cast<void*>(LoggedBarrier);
    VirtualProtect(&v[26], 8, old, &old);
}void LogD3D12Messages(ID3D12Device* device)
{
        if (Env("NFRUX_TEST_D3D12_MESSAGES"))  // the host runs the debug layer: its messages into the log, while NFRUX records or not
        {
            ID3D12InfoQueue1* q = nullptr;
            if (SUCCEEDED(device->QueryInterface(__uuidof(ID3D12InfoQueue1), reinterpret_cast<void**>(&q))))
            {
                for (auto s : {D3D12_MESSAGE_SEVERITY_CORRUPTION, D3D12_MESSAGE_SEVERITY_ERROR, D3D12_MESSAGE_SEVERITY_WARNING}) q->SetBreakOnSeverity(s, FALSE);
                DWORD cookie = 0;
                const HRESULT hr = q->RegisterMessageCallback([](D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY s, D3D12_MESSAGE_ID id, LPCSTR text, void*) {
                    static std::atomic<int> n[2048];
                    if (s <= D3D12_MESSAGE_SEVERITY_WARNING && uint32_t(id) < 2048 && n[id]++ < 6) Log("D3D12 %s %d%s: %s", s <= D3D12_MESSAGE_SEVERITY_ERROR ? "error" : "warning", int(id), t_recording ? " (in NFRUX)" : "", text);
                }, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie);
                Log("D3D12 messages are logged (%08lx)", hr);
                q->Release();
            }
            else Log("no D3D12 debug layer on the host's device");
        }
}
std::unique_ptr<Device> CreateNriD3D12Device(ID3D12Device* device)
{
    auto d = std::make_unique<DevN>();
    return d->InitD3D12(device) ? std::move(d) : nullptr;
}
std::unique_ptr<Device> CreateNriVulkanDevice(const VulkanHost& host)
{
    auto d = std::make_unique<DevN>();
    return d->InitVulkan(host) ? std::move(d) : nullptr;
}
}  // namespace nfrux
