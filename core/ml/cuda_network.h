// The network on D3D12 through CUDA kernels (NVAPI), int8 tensor-core products (mma.sync), exact integer results. No CUDA toolkit: the
// PTX is written here for the network's size and the driver's nvcuda.dll links it to a cubin. SPDX-License-Identifier: MIT
#pragma once
#include "tosa_graph.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <memory>
#include <string>
#include <vector>

namespace nfrux
{
class CudaGraph
{
public:
    // Null (and why) when the GPU, driver or a layer does not fit.
    static std::unique_ptr<CudaGraph> Create(ID3D12Device* device, const ffx_ml::GraphDesc& desc, std::string* error);
    ~CudaGraph();
    // Input and outputs in UNORDERED_ACCESS. Outputs: four channels a pixel, outputPitch bytes a row (as given to Create's caller).
    void     Record(ID3D12GraphicsCommandList* list, D3D12_GPU_VIRTUAL_ADDRESS input, ID3D12Resource* const* outputs, uint32_t outputPitch);
    uint32_t outputBinding(uint32_t i) const { return m_outputBindings[i]; }

private:
    struct Layer
    {
        void*    fn = nullptr;
        uint64_t weights = 0, params = 0;
        int32_t  src[2] = {}, dst = 0;  // internal tensor; -1 the input; -(2 + i) output i
        uint32_t grid[2] = {};
        bool     external = false;
        uint32_t row = 0;  // bytes an output row (internal)
    };
    Microsoft::WRL::ComPtr<ID3D12Device>                m_device;
    Microsoft::WRL::ComPtr<ID3D12Resource>              m_constants, m_upload;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_tensors;
    std::vector<Layer>                                  m_layers;
    std::vector<uint32_t>                               m_outputBindings;
    void*                                               m_module = nullptr;
    bool                                                m_uploaded = false;
};
}  // namespace nfrux
