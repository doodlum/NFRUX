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
bool HasTensorCores(const char* name);  // an sm_75 device, by its name
// What both APIs launch: the kernels linked for the GPU, the constants and the layers.
struct CudaPlan
{
    struct Layer
    {
        uint64_t weights = 0, params = 0;
        int32_t  src[2] = {}, dst = 0;  // internal tensor; -1 the input; -(2 + i) output i
        uint32_t grid[2] = {};
        bool     external = false;
    };
    std::vector<uint8_t>  cubin, constants;
    std::vector<uint64_t> tensors;  // bytes
    std::vector<Layer>    layers;   // layerN
    std::vector<uint32_t> outputBindings;
};
bool MakeCudaPlan(const ffx_ml::GraphDesc& desc, LUID luid, CudaPlan& out, std::string* error);
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
    struct Layer : CudaPlan::Layer { void* fn = nullptr; };
    Microsoft::WRL::ComPtr<ID3D12Device>                m_device;
    Microsoft::WRL::ComPtr<ID3D12Resource>              m_constants, m_upload;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_tensors;
    std::vector<Layer>                                  m_layers;
    std::vector<uint32_t>                               m_outputBindings;
    void*                                               m_module = nullptr;
    bool                                                m_uploaded = false;
};
}  // namespace nfrux
