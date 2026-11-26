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
#pragma pack(push, 4)
struct FnParams  // a fused FasterNet block (C 48 / 64)
{
    uint64_t x, y, wp, we, wj, bp, be, bj;  // tensors; the pconv / expand / project weights and biases
    int32_t  w, h;
};
#pragma pack(pop)
static_assert(sizeof(FnParams) == 72, "the fused kernel's parameter block");

class CudaKernels
{
public:
    static std::unique_ptr<CudaKernels> Create(Device& dev, std::string* error);
    ~CudaKernels();
    // shape 0 3x3/1, 1 2x2/2, 2 1x1, 3 transposed, 4 / 5: 3x3 / transposed with one n tile a warp; list: the D3D12 command list (Device::NativeAccess). 32 threads a group.
    bool Launch(void* list, int shape, const uint32_t grid[3], const CudaParams& p) const;
    // A whole FasterNet block (no up), 16 pixels of a row a warp: grid (W / 16 rounded up, H, 1).
    bool LaunchFn(void* list, uint32_t C, const uint32_t grid[3], const FnParams& p) const;

private:
    CudaKernels() = default;
    ID3D12Device* m_device = nullptr;
    void*         m_module = nullptr;
    void*         m_fn[8] = {};
};
std::string CudaPtx();  // tests
}  // namespace nfrux::mlfg
