// The network on D3D12 compute (nfrux_conv.hlsl), exact integer results. SPDX-License-Identifier: MIT
#pragma once
#include "ffx_vk_ml_graph.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <memory>
#include <string>
#include <vector>

namespace nfrux
{
class D3D12Graph
{
public:
    // dot4: packed dot products (shader model 6.4), else shader model 6.0 integers. Null (and error) when a layer is
    // outside what the kernels implement.
    static std::unique_ptr<D3D12Graph> Create(ID3D12Device* device, const ffx_ml::GraphDesc& desc, bool dot4, std::string* error);

    // The input (tight NHWC int8) and outputs (four channels a pixel) as buffers: input in UNORDERED_ACCESS or a shader
    // read state, outputs in UNORDERED_ACCESS before and after.
    void Record(ID3D12GraphicsCommandList* list, D3D12_GPU_VIRTUAL_ADDRESS input, ID3D12Resource* const* outputs) const;
    // Weights: uploaded by the first recording (the upload buffer lives as long as the graph).
    uint32_t                      outputCount() const { return uint32_t(m_outputBindings.size()); }
    uint32_t                      outputBinding(uint32_t i) const { return m_outputBindings[i]; }

private:
    struct Layer
    {
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pso;
        uint32_t constants[20];
        uint64_t weights, params;
        int32_t  src[2], dst;  // internal tensor index; -1 the input; -(2 + i) output i
        uint32_t groups[3];
    };
    Microsoft::WRL::ComPtr<ID3D12RootSignature>         m_root;
    Microsoft::WRL::ComPtr<ID3D12Resource>              m_constants, m_upload;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_tensors;
    std::vector<Layer>                                  m_layers;
    std::vector<uint32_t>                               m_outputBindings;
    mutable bool                                        m_uploaded = false;
};
}  // namespace nfrux