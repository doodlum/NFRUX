// NFRUX's rhi on NVIDIA's NRI: one implementation for D3D12 and Vulkan, on the host's device and command buffer. Barriers,
// descriptor recycling and completion tokens are here; the network goes through native handles. SPDX-License-Identifier: MIT
#include "rhi_nri.h"

#include "d3d12_network.h"
#include "log.h"
#include "model.h"

#include <d3d12.h>
#include <vulkan/vulkan.h>

#include <NRI.h>
#include <Extensions/NRIHelper.h>
#include <Extensions/NRIWrapperD3D12.h>
#include <Extensions/NRIWrapperVK.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <map>
#include <thread>

namespace nfrux
{
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
// The host's images: D3D12 COMMON, Vulkan GENERAL, in and out.
const nri::AccessLayoutStage kHost = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::ALL};
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
    std::unique_ptr<D3D12Graph>    d3d;
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
    std::vector<nri::Descriptor*> cbv;
    uint8_t*                map = nullptr;
    uint32_t*               tokenUpMap = nullptr;
    volatile uint32_t*      tokens = nullptr;
    uint32_t                slotValue[kSlots] = {}, next = 0, slot = 0, usedConstants = 0;
    nri::CommandBuffer*     cmd = nullptr;
    void*                   nativeCmd = nullptr;
    std::vector<std::unique_ptr<TexN>> wrapped;
    std::vector<nri::AccessLayoutStage> wrappedHome;
    std::vector<nri::TextureBarrierDesc> tb;
    std::vector<nri::BufferBarrierDesc>  bb;
    struct Retired { std::vector<std::unique_ptr<TexN>> t; nri::CommandBuffer* cmd; uint32_t slot, value; };
    std::vector<Retired>    retired;
    struct Read { nri::Buffer* buf; std::shared_ptr<std::vector<uint8_t>> out; uint32_t pitch, row, rows; };
    std::vector<Read>       reads;

    ~DevN() override
    {
        wrapped.clear(), retired.clear();
        for (auto* x : cbv) n.DestroyDescriptor(x);
        for (auto* x : samplers) if (x) n.DestroyDescriptor(x);
        for (auto* x : pools) if (x) n.DestroyDescriptorPool(x);
        for (auto* x : {constants, tokenUp, tokenBack, ones}) if (x) n.DestroyBuffer(x);
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
        return NRI_OK(n.CreateCommittedBuffer(*dev, where, 0.0f, d, b)) ? b : nullptr;
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
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1 = {};
        device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof(o1));
        c.int64 = o1.Int64ShaderOps, c.atomic64 = c.int64 && c.shaderModel >= 0x66, c.dot4 = c.shaderModel >= 0x64;
        c.cuda  = c.vendorId == uint32_t(nri::Vendor::NVIDIA) && GetModuleHandleA("nvapi64.dll");
        Log("D3D12 (NRI): %s, shader model %x, dot4 %d, 64-bit atomics %d", c.name.c_str(), c.shaderModel, c.dot4, c.atomic64);
        return c.shaderModel >= 0x60;
    }
    bool InitVulkan(const VulkanHost& h)
    {
        vk = h, isVk = true;
        nri::QueueFamilyVKDesc q = {1, nri::QueueType::COMPUTE, h.queueFamily};
        nri::DeviceCreationVKDesc d = {};
        d.vkInstance = h.instance, d.vkDevice = h.device, d.vkPhysicalDevice = h.physicalDevice, d.minorVersion = uint8_t(h.minorVersion);
        d.queueFamilies = &q, d.queueFamilyNum = 1;
        d.vkExtensions.deviceExtensions = h.deviceExtensions, d.vkExtensions.deviceExtensionNum = h.deviceExtensionCount;
        d.callbackInterface.MessageCallback = Message;
        d.enableNRIValidation = EnvFlag("NFRUX_VALIDATE", false);
        if (!NRI_OK(nri::nriCreateDeviceFromVKDevice(d, dev)) || !Init()) return false;
        c.atomic64 = h.atomic64, c.int64 = h.atomic64;
        c.dot4 = (h.mlFeatures & ffx_ml::FEATURE_INTEGER_DOT_PRODUCT) != 0, c.cooperativeMatrix = (h.mlFeatures & ffx_ml::FEATURE_COOPERATIVE_MATRIX) != 0;
        Log("Vulkan (NRI): %s, dot4 %d, cooperative matrix %d, 64-bit atomics %d", c.name.c_str(), c.dot4, c.cooperativeMatrix, c.atomic64);
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
    std::unique_ptr<Texture> CreateTexture(Format f, uint32_t w, uint32_t h, bool) override
    {
        auto t = MakeTex(ToNri(f), w, h, true);
        if (t) t->format = f;
        return t;
    }
    nri::Format HostFormat(const ExternalImage& e)
    {
        if (isVk) return nri::nriConvertVKFormatToNRI(e.format);
        return nri::nriConvertDXGIFormatToNRI(e.format ? e.format : uint32_t(static_cast<ID3D12Resource*>(e.handle)->GetDesc().Format));
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
        return p;
    }

    std::unique_ptr<Network> CreateNetwork(uint32_t w, uint32_t h, std::string* error) override
    {
        auto net = std::make_unique<NetN>();
        const ModelBindings mb = GetModelBindings();
        const char* ml = Env("NFRUX_ML");
        std::string what;
        if (isVk)
        {
            const ffx_ml::DeviceContext dc = {vk.instance, vk.physicalDevice, vk.device, vk.gipa, vk.gdpa, vk.mlFeatures};
            ffx_ml::Path path = ffx_ml::Path::Auto;
            if (ml) path = !strcmp(ml, "coopmat") ? ffx_ml::Path::CoopMat : !strcmp(ml, "dot4") ? ffx_ml::Path::Dot4 : ffx_ml::Path::Scalar;
            ffx_ml::Selection sel = ffx_ml::SelectPath(dc, path, path == ffx_ml::Path::CoopMat ? ffx_ml::MatType::Int8 : ffx_ml::MatType::None);
            sel.rescaleFloat = false;  // exact
            if (path != ffx_ml::Path::Auto && sel.path != path) { if (error) *error = "forced network path unavailable"; return nullptr; }
            net->vk = ffx_ml::Graph::Create(dc, sel, DescribeModel(w, h), error);
            if (!net->vk) return nullptr;
            what = sel.Describe();
        }
        else
        {
            const bool dot4 = c.dot4 && !(ml && !strcmp(ml, "scalar"));
            net->d3d = D3D12Graph::Create(d3d, DescribeModel(w, h), dot4, error);
            if (!net->d3d) return nullptr;
            what = dot4 ? "shader model 6.4 packed dot product" : "shader model 6.0 integers";
            for (uint32_t i = 0; i < 2; ++i) net->order[i] = net->d3d->outputBinding(i) == mb.params ? 0 : 1;
        }
        net->pitch = isVk ? w / 2 * 4 : (w / 2 * 4 + 255) & ~255u;
        net->in    = MakeBuf(uint64_t(w) * h * kInputChannels);
        bool ok = net->in != nullptr;
        for (uint32_t i = 0; i < 2; ++i)
        {
            net->out[i] = MakeBuf(uint64_t(net->pitch) * (h / 2));
            net->tex[i] = MakeTex(nri::Format::RGBA8_SNORM, w / 2, h / 2, false);
            ok = ok && net->out[i] && net->tex[i];
            if (ok) net->tex[i]->format = Format::RGBA8Snorm;
        }
        if (!ok) { if (error) *error = "out of memory"; return nullptr; }
        net->input = net->in.get(), net->params = net->tex[0].get(), net->gate = net->tex[1].get();
        static std::string logged;
        if (logged != what) Log("network %ux%u: %s", w, h, what.c_str()), logged = what;
        return net;
    }

    // --- recording
    bool Done(uint32_t s, uint32_t v) const { return tokens[s] == v || slotValue[s] != v; }
    void Poll()
    {
        for (auto it = retired.begin(); it != retired.end();)
            if (Done(it->slot, it->value)) { if (it->cmd) n.DestroyCommandBuffer(it->cmd); it = retired.erase(it); } else ++it;
    }
    void WaitSlot(uint32_t s)
    {
        for (int i = 0; i < 2000 && slotValue[s] && tokens[s] != slotValue[s]; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    bool BeginFrame(void* commands) override
    {
        Poll();
        nativeCmd = commands;
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
        slot = next++ % kSlots;
        WaitSlot(slot);
        slotValue[slot]  = next;
        tokenUpMap[slot] = next;
        usedConstants    = 0;
        n.ResetDescriptorPool(*pools[slot]);
        n.CmdSetDescriptorPool(*cmd, *pools[slot]);
        return true;
    }
    void EndFrame() override
    {
        for (size_t i = 0; i < wrapped.size(); ++i) To(*wrapped[i], wrappedHome[i]);
        Flush();
        n.CmdCopyBuffer(*cmd, *tokenBack, slot * 4, *tokenUp, slot * 4, 4);
        retired.push_back({std::move(wrapped), cmd, slot, slotValue[slot]});
        wrapped.clear(), wrappedHome.clear();
        cmd = nullptr;
    }
    Texture* Wrap(const ExternalImage& e, bool) override
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
        if (!Views(*t, f, false)) return nullptr;
        t->state = kHost;
        wrapped.push_back(std::move(t)), wrappedHome.push_back(kHost);
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
    void CopyRegion(const Texture& s, uint32_t x, uint32_t y, const Texture& d) override { Copy(s, x, y, d, 0, 0, d.width, d.height); }
    // A host buffer of 0xff bytes: 1.0 in R8, ~0 in the scatter keys.
    bool Ones(uint64_t size)
    {
        if (ones && onesSize >= size) return true;
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
        for (uint32_t i = 0; i < count; ++i)
        {
            auto& x = static_cast<NetN&>(*nets[i]);
            To(*x.in, nri::AccessBits::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER);
            for (auto& o : x.out) To(*o, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER);
        }
        Flush();
        if (isVk)
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
        else
            for (uint32_t i = 0; i < count; ++i)
            {
                auto& x = static_cast<NetN&>(*nets[i]);
                auto  r = [&](const BufN& f) { return reinterpret_cast<ID3D12Resource*>(n.GetBufferNativeObject(f.buf)); };
                ID3D12Resource* o[2] = {r(*x.out[x.order[0]]), r(*x.out[x.order[1]])};
                x.d3d->Record(static_cast<ID3D12GraphicsCommandList*>(nativeCmd), r(*x.in)->GetGPUVirtualAddress(), o, x.pitch);
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
        Read r = {Committed(uint64_t(pitch) * t.height, nri::MemoryLocation::HOST_READBACK, nri::BufferUsageBits::NONE), out, pitch, row, t.height};
        if (!r.buf) return;
        To(a, kSrc), Flush();
        nri::TextureDataLayoutDesc l = {0, pitch, pitch * t.height};
        nri::TextureRegionDesc rg = {};
        rg.width = nri::WHOLE_SIZE, rg.height = nri::WHOLE_SIZE, rg.depth = 1;
        n.CmdReadbackTextureToBuffer(*cmd, *r.buf, l, *a.tex, rg);
        reads.push_back(r);
    }
    void Mark(const char*) override {}
    void WaitIdle() override
    {
        for (uint32_t s = 0; s < kSlots; ++s) WaitSlot(s);
        for (auto& r : reads)
        {
            if (auto* p = static_cast<uint8_t*>(n.MapBuffer(*r.buf, 0, nri::WHOLE_SIZE)))
            {
                r.out->resize(size_t(r.row) * r.rows);
                for (uint32_t y = 0; y < r.rows; ++y) memcpy(r.out->data() + size_t(y) * r.row, p + size_t(y) * r.pitch, r.row);
                n.UnmapBuffer(*r.buf);
            }
            n.DestroyBuffer(r.buf);
        }
        reads.clear();
        Poll();
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
    d.n.CmdDispatch(*d.cmd, {gx, gy, 1});
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
