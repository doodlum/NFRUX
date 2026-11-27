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

namespace nfrux::shaders { struct MlfgMatSet; }
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

// The int8 network (AmdPrecision=int8): tools/mlfg_weights.py --int8 from an export_amd int8 export (blob_int8's layout);
// semantics tools/ffx_extract/mlfg_int8.py. Embedded, or nfrux.ini [General] AmdWeightsInt8 / NFRUX_AMD_WEIGHTS_INT8.
struct QLayerWeights
{
    uint32_t kind, k, stride, cin, cout, flags, act, wq, ws, bias, fold, qin, qpre, qout, qres, pad;  // byte offsets; q*: act indices
    static constexpr uint32_t kFold = 1, kF16Out = 2, kNone = 0xFFFFFFFFu;
};
struct QWeights
{
    std::vector<uint8_t>       file;
    const uint8_t*             data = nullptr;
    size_t                     size = 0;
    std::vector<QLayerWeights> layers;
    std::vector<float>         act;  // per-tensor scales
    float F(size_t byteOffset) const;
    int   W(uint32_t layer, size_t i) const { return int(int8_t(data[layers[layer].wq + i])); }  // the blob's layout
    float S(uint32_t index) const { return index == QLayerWeights::kNone ? 0.0f : act[index]; }
    static bool Load(QWeights& w, std::string* error);
};

// A tensor buffer and the view of it a layer reads or writes: size, channel pitch (halves; int8 tensors: bytes), first channel.
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
    bool int8() const { return m_int8; }
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
    // One int8 layer: src int8 (or fp16 quantised per channel: srcF16, block entries), res int8, out int8 / fp16 (the
    // layer's fp16 flag, or outF16: dequantised); output channels from copyFrom on copy src requantised (FasterNet).
    struct QConv
    {
        uint32_t layer = 0;
        View     src, res, out;
        bool     srcF16 = false, outF16 = false;
        uint32_t copyFrom = ~0u, writeEnd = 0;
    };
    bool RunQ(const QConv& c);

private:
    Network(Device& d) : m_dev(d) {}
    bool Make(uint32_t l0w, uint32_t l0h, uint32_t s2w, uint32_t s2h, std::string* error);
    bool SelectTier(std::string* error);
    bool SelectPrecision(std::string* error);
    bool MakeInt8(std::string* error);
    bool EncodersQ(const View& netIn);
    bool FlowBlockQ(uint32_t block, const View& in, const View& out13);
    bool Stage2Q(const View& in, const View& logits);
    bool FasterNetQ(uint32_t first, uint32_t C, const View& x, const View& y, const View& hin, const View& h, const View* skip, const View* upOut);
    bool                    m_int8 = false, m_dot4 = false, m_q8cuda = false;  // m_q8cuda: int8 CUDA kernels (m_cuda) for int8-source layers
    std::vector<uint32_t>   m_qcOffset, m_qcMb;  // int8 CUDA: per layer weight words (m_gpuW8), m / b floats (m_gpuBias)
    QWeights                m_q;
    std::vector<uint32_t>   m_qOffset;           // per layer: weight words, then m / b / rf
    std::unique_ptr<Pass>   m_q8[4][3];          // int8: 3x3, 2x2/2, 1x1, transposed; large / small tile / 8x8 by 32 channels
    Tensor                  m_hin48, m_hin64;    // int8: the FasterNet expand inputs
    bool FasterNet(uint32_t first, uint32_t C, uint32_t P, const View& x, const View& y, const View& pc, const View& h, const View* skip, const View* upOut);
    Device&                 m_dev;
    Tier                    m_tier = Tier::Generic;
    Weights                 m_weights;
    std::vector<uint32_t>   m_offset, m_biasOffset, m_nb, m_off8;  // m_off8: FP8 FasterNet weights (bytes)
    std::vector<uint8_t>    m_blob8;
    std::unique_ptr<Buffer> m_gpuW8;  // the layer's weights in the tier's buffers (floats / halves)
    std::vector<uint8_t>    m_blob, m_biasBlob;  // the tier's weights until uploaded
    std::unique_ptr<Buffer> m_gpuWeights, m_gpuBias;
    std::string             m_marker;  // crash marker (cooperative matrices off NVIDIA), cleared after 120 frames
    uint32_t                m_frames = 0;
    std::unique_ptr<Pass>   m_conv[4][2];  // 3x3, 2x2/2, 1x1, transposed; large / small tile
    std::unique_ptr<Pass>   m_mat[4], m_matN1[2];  // coopmat: layers; narrow (one 16-output tile)
    std::unique_ptr<Pass>   m_matN2, m_mat1n3;
    const shaders::MlfgMatSet* m_matSet = nullptr;  // coopmat: the variant set (fragment shape, subgroup size)  // coopmat: 3x3 with 17-32 outputs (2 tiles); 1x1 with 48 / 96 outputs (3 tiles)
    std::unique_ptr<CudaKernels> m_cuda;
    bool                    m_fused = true;
    bool                    m_m2 = true;
    static constexpr uint32_t kPartialFloats = 2u << 20;  // split-K's fp32 partial sums (8 MB)
    std::unique_ptr<Buffer> m_partial;  // CUDA split-K: the fp32 partial sums
    bool                    m_uploaded = false;
    Tensor                  m_a, m_b, m_c, m_d;                                 // stage-1 scratch
    Tensor                  m_t32, m_x48, m_y48, m_skip, m_pc16, m_h96, m_x64, m_y64, m_pc32, m_h128, m_u32;  // stage 2
};
}  // namespace nfrux::mlfg
