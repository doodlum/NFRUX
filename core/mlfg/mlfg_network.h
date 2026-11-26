// The ML runtime of NFRUX's FSR Frame Generation 4 interpolator (AMD FidelityFX SDK v2.4.0, MIT; docs/FFX_MLFG.md): the
// 105 layers of AMD's network (core/model/amd_mlfg_weights.h, tools/mlfg_weights.py) over NHWC fp16 tensors
// (core/shaders/mlfg/mlfg_constants.h), with mlfg_net's 'amd' numerics: fp16 activations, fp32 sums, fp8 e4m3 rounding
// of the stage-2 activations. Tiers (the ladder: the fastest the GPU and API allow, forced by nfrux.ini [D3D12] /
// [Vulkan] AmdNetwork or NFRUX_MLFG):
//   generic   HLSL kernels (kMlfgConv*), both APIs, any D3D12 FL12 / Vulkan 1.1 GPU; strict: the same bytes everywhere
//   coopmat   Vulkan cooperative matrices fp16 x fp16 -> fp32 (VK_KHR_cooperative_matrix), every layer
//   cuda      D3D12 on NVIDIA RTX: CUDA kernels, mma.sync fp16 -> fp32 (NVAPI launch, as the int8 network's)
// A tier that is not the generic one gives the generic tier's results within fp32 summation-order differences.
// SPDX-License-Identifier: MIT
#pragma once
#include "mlfg_cuda.h"
#include "rhi.h"

#include <memory>
#include <string>
#include <vector>

namespace nfrux::mlfg
{
enum class Tier { Generic, CoopMat, Cuda };
const char* TierName(Tier t);

// The weights: NFRUX's own (embedded), or a file tools/mlfg_weights.py --bin made from a retrained npz.
struct LayerWeights
{
    uint32_t kind, k, stride, cin, cout, fp8, weights, bias;  // kind 0 conv, 1 transposed 2x2/2; byte offsets
};
struct Weights
{
    std::vector<uint8_t>      file;
    const uint8_t*            data = nullptr;
    size_t                    size = 0;
    std::vector<LayerWeights> layers;
    float W(uint32_t layer, size_t i) const;  // element i in the blob's layout (conv [co][ci][k][k], transposed [ci][co][2][2])
    float B(uint32_t layer, uint32_t o) const;
    static bool Load(Weights& w, std::string* error);  // nfrux.ini [General] AmdWeights / NFRUX_AMD_WEIGHTS, else embedded
};

// A tensor buffer and the view of it a layer reads or writes: size, channel pitch (halves), first channel.
struct View
{
    const Buffer* buf = nullptr;
    uint32_t      w = 0, h = 0, pitch = 0, ch = 0;
    explicit operator bool() const { return buf != nullptr; }
    View At(uint32_t c) const { View v = *this; v.ch = c; return v; }
};
struct Tensor
{
    std::unique_ptr<Buffer> buf;
    uint32_t                w = 0, h = 0, pitch = 0;
    View v(uint32_t ch = 0) const { return {buf.get(), w, h, pitch, ch}; }
};

class Network
{
public:
    enum Act { None = 0, Leaky = 1, Relu = 2 };
    // l0: the stage-1 level-0 tensor size (mlfg_sizes 'l0'); s2: the stage-2 tensor size ('s2').
    static std::unique_ptr<Network> Create(Device& dev, uint32_t l0w, uint32_t l0h, uint32_t s2w, uint32_t s2h, std::string* error);
    ~Network();
    Tier tier() const { return m_tier; }
    std::string Describe() const;

    bool Prepare();  // the weights' upload, recorded into the first frame
    void FrameDone();  // a frame was recorded: the crash marker is cleared after the first 120
    // Passes 2-16: rgb0 / rgb1 (ch 0-2 / 3-5 of net_in) -> features into ch 6-11 / 12-17.
    bool Encoders(const View& netIn);
    // One flow block (0-3) of stage 1: in (its size S) -> out13 (13 channels, size S).
    bool FlowBlock(uint32_t block, const View& in, const View& out13);
    // Stage 2 (passes 154-188 up to the softmax): the 24-channel input -> 5 fp16 logits.
    bool Stage2(const View& in, const View& logits);
    // One layer (exposed for tests): out = round(act(conv(src | src1) + bias + add)).
    struct Conv
    {
        uint32_t layer = 0;
        View     src, src1, add, out;
        uint32_t split = ~0u;  // input channels from src1 from here on
        Act      act = None;
        bool     fp8 = false;
        uint32_t writeEnd = 0;  // channels written (zeros beyond cout); 0: cout rounded up to even
    };
    bool Run(const Conv& c);

private:
    Network(Device& d) : m_dev(d) {}
    bool Make(uint32_t l0w, uint32_t l0h, uint32_t s2w, uint32_t s2h, std::string* error);
    bool SelectTier(std::string* error);
    bool FasterNet(uint32_t first, uint32_t C, uint32_t P, const View& x, const View& y, const View& pc, const View& h, const View* skip, const View* upOut);
    Device&                 m_dev;
    Tier                    m_tier = Tier::Generic;
    Weights                 m_weights;
    std::vector<uint32_t>   m_offset, m_biasOffset, m_nb;  // the layer's weights in the tier's buffers (floats / halves)
    std::vector<uint8_t>    m_blob, m_biasBlob;  // the tier's weights until uploaded
    std::unique_ptr<Buffer> m_gpuWeights, m_gpuBias;
    std::string             m_marker;  // crash marker (cooperative matrices off NVIDIA), cleared after 120 frames
    uint32_t                m_frames = 0;
    std::unique_ptr<Pass>   m_conv[4][2];  // 3x3, 2x2/2, 1x1, transposed; large / small tile
    std::unique_ptr<Pass>   m_mat[4], m_matN1[2], m_fn[2];  // coopmat: layers; narrow; fused FasterNet 48 / 64
    std::unique_ptr<CudaKernels> m_cuda;
    bool                    m_fused = true;
    bool                    m_m2 = true;  // CUDA: 32 pixels a warp (NFRUX_MLFG_NO_M2=1: 16, tests)  // fused FasterNet blocks (NFRUX_MLFG_UNFUSED=1: the layers one by one, tests)
    bool                    m_uploaded = false;
    Tensor                  m_a, m_b, m_c, m_d;                                 // stage-1 scratch
    Tensor                  m_t32, m_x48, m_y48, m_skip, m_pc16, m_h96, m_x64, m_y64, m_pc32, m_h128, m_u32;  // stage 2
};
}  // namespace nfrux::mlfg
