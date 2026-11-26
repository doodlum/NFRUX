// The FSR FG4 runtime's CUDA tier (NVIDIA RTX, D3D12 and Vulkan): tensor-core kernels, see mlfg_cuda.cpp. SPDX-License-Identifier: MIT
#pragma once
#include "rhi.h"

#include <memory>
#include <string>

struct ID3D12Device;
namespace nfrux::mlfg
{
// Warps a CUDA block, each one row (grid y = rows / this, rounded up): one-warp blocks would cap an SM at 24 warps.
constexpr uint32_t kCudaWarps = 4;
#pragma pack(push, 4)
struct CudaParams
{
    uint64_t s0, s1, w, b, add, out;  // GPU virtual addresses
    int32_t  inW, inH, p0, ch0, p1, ch1m, split, cin, outW, outH, po, cho, cout, writeEnd, act, fp8, pa, cha, kc, ntp;
};
#pragma pack(pop)
static_assert(sizeof(CudaParams) == 128, "the kernels' parameter block");
#pragma pack(push, 4)
struct SplitParams  // split-K (3x3, 8 n tiles): the layer's, the partial sums' buffer, chunks a split, splits
{
    CudaParams c;
    uint64_t   part;
    int32_t    per, splits;
};
#pragma pack(pop)
static_assert(sizeof(SplitParams) == 144, "split-K's parameter block");
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
    // shape 0 3x3/1, 1 2x2/2, 2 1x1, 3 transposed; nb n tiles (8 channels) a warp: 1, 2, 3, 4, 6, 8 (transposed up to 4); bufs: the buffers it touches (Device::CudaLaunch).
    bool Launch(const Buffer* const* bufs, uint32_t n, int shape, uint32_t nb, const uint32_t grid[3], const CudaParams& p) const;
    // shape 0-2, 8 n tiles, 32 pixels of a row a warp: grid (W / 32 rounded up, rows / kCudaWarps, n tiles / 8).
    // staged (3x3 only): the shared-memory-staged kernel (one source, word-aligned channel pairs).
    bool LaunchM(const Buffer* const* bufs, uint32_t n, int shape, const uint32_t grid[3], const CudaParams& p, bool staged = false) const;
    // A whole FasterNet block (no up), 16 pixels of a row a warp: grid (W / 16 rounded up, H, 1).
    bool LaunchFn(const Buffer* const* bufs, uint32_t n, uint32_t C, bool fp8, const uint32_t grid[3], const FnParams& p) const;  // fp8: the e4m3 kernel (FP8 weights)
    // Split-K 3x3 (8 n tiles, 16 pixels a warp): grid z = n-tile groups x splits; then the fixed-order reduction + epilogue.
    bool LaunchSplit(const Buffer* const* bufs, uint32_t n, const uint32_t grid[3], const SplitParams& p, uint32_t pixels) const;
    bool Fp8() const { return m_fn8[0] != nullptr; }  // sm_89+: e4m3 tensor cores

private:
    explicit CudaKernels(Device& d) : m_dev(d) {}
    Device&       m_dev;
    void*         m_module = nullptr;
    void*         m_conv[4][6] = {};
    void*         m_fnk[2] = {};
    void*         m_fn8[2] = {};
    void*         m_convM[3] = {};  // 3x3 / 2x2 / 1x1, 8 n tiles, 32 pixels a warp
    void*         m_convS = nullptr;  // 3x3 staged in shared memory
    void*         m_splitK = nullptr, *m_reduce = nullptr;
};
std::string CudaPtx(bool fp8);  // tests
int         Fp8Channel(int kpos);  // an fp8 k chunk's position -> input channel within its 32 (weights' packing)
}  // namespace nfrux::mlfg
