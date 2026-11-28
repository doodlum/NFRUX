// SPDX-License-Identifier: MIT
#include "d3d12_network.h"

#include "nfrux_shaders.h"

#include <algorithm>
#include <cstring>
#include <map>

using Microsoft::WRL::ComPtr;

namespace nfrux
{
namespace
{
ComPtr<ID3D12Resource> MakeBuffer(ID3D12Device* device, uint64_t size, D3D12_HEAP_TYPE heap, bool uav)
{
    D3D12_HEAP_PROPERTIES hp = {heap};
    D3D12_RESOURCE_DESC   d  = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width     = (size + 255) & ~uint64_t(255);
    d.Height = d.DepthOrArraySize = d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags            = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> r;
    device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
                                    heap == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON,
                                    nullptr, IID_PPV_ARGS(&r));
    return r;
}
}  // namespace

std::unique_ptr<D3D12Graph> D3D12Graph::Create(ID3D12Device* device, const ffx_ml::GraphDesc& desc, bool dot4, std::string* error)
{
    auto fail = [&](const std::string& m) -> std::unique_ptr<D3D12Graph> {
        if (error)
            *error = m;
        return nullptr;
    };
    ffx_ml::tosa::Graph  parsed;
    ffx_ml::LoweredGraph g;
    if (!ffx_ml::tosa::Parse(desc, parsed, error) || !ffx_ml::Lower(parsed, g, error))
        return nullptr;
    auto out = std::unique_ptr<D3D12Graph>(new D3D12Graph());

    // b0 the layer, t0..t3 and u0 as root views (all buffers).
    D3D12_ROOT_PARAMETER rp[6] = {};
    rp[0].ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[0].Constants.Num32BitValues = 20;
    for (uint32_t i = 0; i < 4; ++i)
        rp[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV, rp[1 + i].Descriptor.ShaderRegister = i;
    rp[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC rs = {6, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob>          blob, err;
    static const auto serialize = reinterpret_cast<PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>(
        GetProcAddress(LoadLibraryA("d3d12.dll"), "D3D12SerializeRootSignature"));
    if (!serialize || FAILED(serialize(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) ||
        FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&out->m_root))))
        return fail("root signature");

    std::vector<uint8_t> constants;
    auto align = [&] { constants.resize((constants.size() + 255) & ~size_t(255)); };
    const ffx_ml::StoredTensor* input = nullptr;
    std::map<uint32_t, int32_t> slot;  // stored tensor -> Layer::src / dst
    for (uint32_t binding : parsed.outputBindings)
        out->m_outputBindings.push_back(binding);
    for (uint32_t i = 0; i < g.tensors.size(); ++i)
    {
        const auto& t = g.tensors[i];
        if (!t.external)
        {
            slot[i] = int32_t(out->m_tensors.size());
            out->m_tensors.push_back(MakeBuffer(device, uint64_t(t.h) * t.w * t.channelStride, D3D12_HEAP_TYPE_DEFAULT, true));
            if (!out->m_tensors.back())
                return fail("out of memory");
            continue;
        }
        const auto it = std::find(parsed.outputBindings.begin(), parsed.outputBindings.end(), t.binding);
        const bool isInput = std::find(parsed.inputBindings.begin(), parsed.inputBindings.end(), t.binding) != parsed.inputBindings.end();
        if (isInput)
            slot[i] = -1, input = &t;
        else if (it != parsed.outputBindings.end())
            slot[i] = -2 - int32_t(it - parsed.outputBindings.begin());
        else
            return fail("unknown external tensor");
    }
    if (!input)
        return fail("no input");

    std::map<const uint32_t*, ComPtr<ID3D12PipelineState>> psos;
    for (size_t li = 0; li < g.convs.size(); ++li)
    {
        const ffx_ml::FusedConv&   fc = g.convs[li];
        const ffx_ml::tosa::Node&  c = *fc.conv;
        const ffx_ml::tosa::Node&  r = *fc.rescale;
        const ffx_ml::ConvSource&  s0 = fc.sources[0];
        const ffx_ml::StoredTensor &t0 = g.tensors[s0.tensor], &dst = g.tensors[fc.output];
        const bool two = fc.sources.size() == 2;
        const std::string at = " (layer " + std::to_string(li) + ")";
        const int  ic0 = t0.channelStride / 4, ic1 = two ? g.tensors[fc.sources[1].tensor].channelStride / 4 : 0;
        // What conv2d.hlsl implements; anything else is refused, not run inexactly.
        bool ok = fc.sources.size() <= 2 && c.kernelH == c.kernelW && c.stride[0] == c.stride[1] && c.dilation[0] == 1 && c.dilation[1] == 1 &&
                  c.weightZp == 0 && r.scale32 && !r.doubleRound && r.rescaleInZp == 0 && t0.c % 4 == 0 && t0.c == t0.channelStride &&
                  dst.channelStride % 4 == 0 && (!two || (!fc.sources[1].resize && g.tensors[fc.sources[1].tensor].c % 4 == 0 &&
                  g.tensors[fc.sources[1].tensor].c == g.tensors[fc.sources[1].tensor].channelStride)) && fc.inC == (ic0 + ic1) * 4;
        for (int32_t s : r.shift)
            ok = ok && s >= 2 && s <= 62;
        if (s0.resize)
            ok = ok && s0.scale[0] > 0 && s0.scale[2] > 0;
        else
            ok = ok && t0.h == fc.inH && t0.w == fc.inW;
        if (!ok)
            return fail("unsupported layer" + at);
        const int32_t key[9] = {c.kernelH, c.stride[0], ic0, ic1, fc.outC, s0.resize ? 1 : 0, r.perChannel ? 1 : 0, fc.table ? 1 : 0, 0};
        const shaders::ConvShader* sh = nullptr;
        for (const auto& k : shaders::kConvs)
            if (k.dot4 == dot4 && !memcmp(k.key, key, sizeof(key)))
                sh = &k;
        if (!sh)
            return fail("no kernel for the layer" + at);
        Layer l = {};
        auto& pso = psos[sh->dxil];
        if (!pso)
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = out->m_root.Get();
            pd.CS             = {sh->dxil, sh->bytes};
            if (FAILED(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso))))
                return fail("pipeline" + at);
        }
        l.pso = pso;
        const uint32_t icw = uint32_t(ic0 + ic1), ic = icw * 4, oc = uint32_t(fc.outC), taps = uint32_t(c.kernelH * c.kernelW), ocb = sh->ocb;
        const uint32_t blocks = (oc + ocb - 1) / ocb;
        align();
        l.weights = constants.size();
        constants.resize(constants.size() + size_t(blocks) * taps * icw * ocb * 4);
        for (uint32_t o = 0; o < oc; ++o)
            for (uint32_t tap = 0; tap < taps; ++tap)
                for (uint32_t i = 0; i < ic; ++i)
                    constants[l.weights + (((size_t(o / ocb) * taps + tap) * icw + i / 4) * ocb + o % ocb) * 4 + (i & 3)] =
                        uint8_t(c.weights[(size_t(o) * taps + tap) * ic + i]);
        align();
        l.params = constants.size();
        const uint32_t stride = (oc + 3) & ~3u;
        std::vector<int32_t> p(stride * 3 + 64, 0);
        for (uint32_t o = 0; o < oc; ++o)
        {
            int64_t sum = 0;
            for (uint32_t i = 0; i < taps * ic; ++i)
                sum += c.weights[size_t(o) * taps * ic + i];
            p[o] = int32_t(int64_t(c.bias[o]) - int64_t(c.inputZp) * sum);
            const uint32_t k = r.perChannel ? o : 0;
            p[stride + o]     = r.multiplier[k];
            p[2 * stride + o] = r.shift[k];
        }
        if (fc.table)
            memcpy(&p[3 * stride], fc.table->table.data(), 256);
        constants.insert(constants.end(), reinterpret_cast<uint8_t*>(p.data()), reinterpret_cast<uint8_t*>(p.data() + p.size()));
        const int32_t k[20] = {fc.inW, fc.inH, fc.outW, fc.outH, t0.w, t0.h, c.pad[2], c.pad[0], s0.scale[0], s0.scale[1], s0.scale[2], s0.scale[3],
                               s0.offset[0], s0.offset[1], r.rescaleOutZp, c.inputZp, dst.channelStride / 4, fc.outW * (dst.channelStride / 4)};
        memcpy(l.constants, k, sizeof(k));
        l.src[0] = slot[s0.tensor];
        l.src[1] = two ? slot[fc.sources[1].tensor] : l.src[0];
        l.dst    = slot[fc.output];
        l.groups[0] = (uint32_t(fc.outW) + sh->px * 8 - 1) / (sh->px * 8);
        l.groups[1] = (uint32_t(fc.outH) + 7) / 8;
        l.groups[2] = blocks;
        out->m_layers.push_back(l);
    }
    out->m_constants = MakeBuffer(device, constants.size(), D3D12_HEAP_TYPE_DEFAULT, false);
    out->m_upload    = MakeBuffer(device, constants.size(), D3D12_HEAP_TYPE_UPLOAD, false);
    void* map = nullptr;
    if (!out->m_constants || !out->m_upload || FAILED(out->m_upload->Map(0, nullptr, &map)))
        return fail("out of memory");
    memcpy(map, constants.data(), constants.size());
    out->m_upload->Unmap(0, nullptr);
    return out;
}

void D3D12Graph::Record(ID3D12GraphicsCommandList* list, D3D12_GPU_VIRTUAL_ADDRESS input, ID3D12Resource* const* outputs, uint32_t outputPitch) const
{
    const D3D12Graph* self = this;
    RecordBatch(list, &self, &input, &outputs, 1, outputPitch);
}

void D3D12Graph::RecordBatch(ID3D12GraphicsCommandList* list, const D3D12Graph* const* graphs, const D3D12_GPU_VIRTUAL_ADDRESS* inputs,
                             ID3D12Resource* const* const* outputs, uint32_t count, uint32_t outputPitch)
{
    auto transition = [](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
        D3D12_RESOURCE_BARRIER x = {};
        x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
        return x;
    };
    std::vector<D3D12_RESOURCE_BARRIER> b;
    for (uint32_t g = 0; g < count; ++g)
        if (!graphs[g]->m_uploaded)
        {
            list->CopyResource(graphs[g]->m_constants.Get(), graphs[g]->m_upload.Get());  // COMMON promotes to COPY_DEST
            b.push_back(transition(graphs[g]->m_constants.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
            graphs[g]->m_uploaded = true;
        }
    if (!b.empty())
        list->ResourceBarrier(UINT(b.size()), b.data()), b.clear();
    // The tensors decay to COMMON between lists and go back to it here: explicit states within the recording.
    std::vector<std::vector<D3D12_RESOURCE_STATES>> state(count);
    size_t layers = 0;
    for (uint32_t g = 0; g < count; ++g)
        state[g].assign(graphs[g]->m_tensors.size(), D3D12_RESOURCE_STATE_COMMON), layers = std::max(layers, graphs[g]->m_layers.size());
    ID3D12RootSignature* root = nullptr;
    for (size_t li = 0; li < layers; ++li)
    {
        for (uint32_t g = 0; g < count; ++g)
        {
            if (li >= graphs[g]->m_layers.size())
                continue;
            const Layer& l = graphs[g]->m_layers[li];
            auto want = [&](int32_t t, D3D12_RESOURCE_STATES s) {
                if (t < 0 || state[g][t] == s)
                    return;
                b.push_back(transition(graphs[g]->m_tensors[t].Get(), state[g][t], s));
                state[g][t] = s;
            };
            want(l.src[0], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            want(l.src[1], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            want(l.dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (!b.empty())
            list->ResourceBarrier(UINT(b.size()), b.data()), b.clear();
        for (uint32_t g = 0; g < count; ++g)
        {
            const D3D12Graph& x = *graphs[g];
            if (li >= x.m_layers.size())
                continue;
            const Layer& l = x.m_layers[li];
            if (root != x.m_root.Get())
                list->SetComputeRootSignature(root = x.m_root.Get());
            const uint64_t base = x.m_constants->GetGPUVirtualAddress();
            auto va = [&](int32_t t) { return t >= 0 ? x.m_tensors[t]->GetGPUVirtualAddress() : t == -1 ? inputs[g] : outputs[g][-2 - t]->GetGPUVirtualAddress(); };
            list->SetPipelineState(l.pso.Get());
            list->SetComputeRoot32BitConstants(0, 20, l.constants, 0);
            if (l.dst < -1 && outputPitch)
                list->SetComputeRoot32BitConstant(0, outputPitch / 4, 17);
            list->SetComputeRootShaderResourceView(1, va(l.src[0]));
            list->SetComputeRootShaderResourceView(2, va(l.src[1]));
            list->SetComputeRootShaderResourceView(3, base + l.weights);
            list->SetComputeRootShaderResourceView(4, base + l.params);
            list->SetComputeRootUnorderedAccessView(5, va(l.dst));
            list->Dispatch(l.groups[0], l.groups[1], l.groups[2]);
        }
    }
    for (uint32_t g = 0; g < count; ++g)
        for (size_t i = 0; i < graphs[g]->m_tensors.size(); ++i)
            if (state[g][i] != D3D12_RESOURCE_STATE_COMMON)
                b.push_back(transition(graphs[g]->m_tensors[i].Get(), state[g][i], D3D12_RESOURCE_STATE_COMMON));
    if (!b.empty())
        list->ResourceBarrier(UINT(b.size()), b.data());
}
}  // namespace nfrux
