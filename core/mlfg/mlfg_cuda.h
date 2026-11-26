// The FSR FG4 runtime's CUDA tier (D3D12, NVIDIA RTX): fp16 tensor-core kernels, see mlfg_cuda.cpp. SPDX-License-Identifier: MIT
#pragma once
#include "rhi.h"

#include <memory>
#include <string>

struct ID3D12Device;
namespace nfrux::mlfg
{
#pragma pack(push, 4)
struct CudaParams
{
    uint64_t s0, s1, w, b, add, out;  // GPU virtual addresses
    int32_t  inW, inH, p0, ch0, p1, ch1m, split, cin, outW, outH, po, cho, cout, writeEnd, act, fp8, pa, cha, kc, ntp;
};
#pragma pack(pop)
static_assert(sizeof(CudaParams) == 128, "the kernels' parameter block");

class CudaKernels
{
public:
    static std::unique_ptr<CudaKernels> Create(Device& dev, std::string* error);
    ~CudaKernels();
    // shape 0 3x3/1, 1 2x2/2, 2 1x1, 3 transposed, 4 / 5: 3x3 / transposed with one n tile a warp; list: the D3D12 command list (Device::NativeAccess). 32 threads a group.
    bool Launch(void* list, int shape, const uint32_t grid[3], const CudaParams& p) const;

private:
    CudaKernels() = default;
    ID3D12Device* m_device = nullptr;
    void*         m_module = nullptr;
    void*         m_fn[6] = {};
};
std::string CudaPtx();  // tests
}  // namespace nfrux::mlfg
