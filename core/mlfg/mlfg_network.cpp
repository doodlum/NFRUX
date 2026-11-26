// SPDX-License-Identifier: MIT
#include "mlfg_network.h"

#include "amd_mlfg_weights.h"
#include "log.h"
#include "marker.h"
#include "mlfg_coopmat.h"
#include "mlfg_cuda.h"
#include "nfrux_shaders.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

namespace nfrux::mlfg
{
namespace
{
constexpr uint32_t kLayers = 105;
constexpr uint32_t kEncA = 0, kEncB = 4, kBlock0 = 8, kS2Inp = 60, kS2D1 = 61, kS2B1 = 62, kS2D2 = 74, kS2B2 = 75, kS2B3 = 91, kS2Up = 103, kS2Head = 104;
const uint32_t kBlocks[4][3] = {{18, 96, 192}, {32, 64, 128}, {32, 48, 96}, {32, 24, 48}};  // cin, c1, c2

// The layer list the runtime runs: kind, k, stride, cin, cout (a weights file must have exactly these).
std::vector<std::array<uint32_t, 5>> Expected()
{
    std::vector<std::array<uint32_t, 5>> v;
    auto conv  = [&](uint32_t k, uint32_t s, uint32_t ci, uint32_t co) { v.push_back({0, k, s, ci, co}); };
    auto convT = [&](uint32_t ci, uint32_t co) { v.push_back({1, 2, 2, ci, co}); };
    for (int e = 0; e < 2; ++e) conv(2, 2, 3, 32), conv(3, 1, 32, 32), conv(3, 1, 32, 32), convT(32, 6);
    for (const auto& b : kBlocks)
    {
        conv(2, 2, b[0], b[1]), conv(3, 1, b[1], b[1]), conv(2, 2, b[1], b[2]);
        for (int r = 0; r < 8; ++r) conv(3, 1, b[2], b[2]);
        convT(b[2], b[1]), convT(b[1], 13);
    }
    auto fn = [&](uint32_t C, uint32_t P) { conv(3, 1, P, P), conv(1, 1, C, 2 * C), conv(1, 1, 2 * C, C); };
    conv(3, 1, 24, 32), conv(2, 2, 32, 48);
    for (int j = 0; j < 4; ++j) fn(48, 16);
    conv(2, 2, 48, 64);
    for (int j = 0; j < 5; ++j) fn(64, 32);
    convT(64, 48);
    for (int j = 0; j < 4; ++j) fn(48, 16);
    convT(48, 32), conv(3, 1, 32, 5);
    return v;
}

float Half(uint16_t h)
{
    const uint32_t s = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    float f;
    if (e == 0) { f = float(m) * 5.9604644775390625e-8f; return s ? -f : f; }
    const uint32_t u = e == 31 ? (s | 0x7F800000u | (m << 13)) : (s | ((e + 112u) << 23) | (m << 13));
    memcpy(&f, &u, 4);
    return f;
}
// The CUDA kernels' n tiles (8 channels) a warp: 8 (transposed 4), padded; 1 for the 3x3 / transposed layers with at most
// 16 outputs. Measured on the RTX 4070: the divisor that pads nothing (6 for 96 outputs, 3 for 24 ...) is slower (stage 1
// at 4K 2.8 -> 3.4 ms; the warps are latency-bound and want 8 independent accumulators). The fused FasterNet kernels
// rely on this rule.
uint32_t CudaNb(const LayerWeights& L)
{
    if (L.cout <= 16 && (L.kind == 1 || L.k == 3)) return 1;
    return L.kind == 1 ? 4 : 8;
}uint16_t HalfBits(float f)  // round to nearest even (the weights are fp16 / fp8 values: exact)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u, e = (u >> 23) & 0xFFu;
    uint32_t       m = u & 0x7FFFFFu;
    if (e == 0xFF) return uint16_t(s | 0x7C00u | (m ? 0x200u : 0u));
    const int E = int(e) - 112;
    if (E >= 31) return uint16_t(s | 0x7C00u);
    if (E <= 0)
    {
        if (E < -10) return uint16_t(s);
        m |= 0x800000u;
        const uint32_t shift = uint32_t(14 - E), rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
        uint32_t       q = m >> shift;
        if (rem > half || (rem == half && (q & 1u))) ++q;
        return uint16_t(s | q);
    }
    uint32_t       q   = (uint32_t(E) << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) ++q;
    return uint16_t(s | q);
}
float Fp8(uint8_t b)
{
    const uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
    float f = e == 0 ? float(m) * 0.001953125f : (e == 15 && m == 7) ? NAN : ldexpf(1.0f + float(m) / 8.0f, int(e) - 7);
    return (b & 0x80) ? -f : f;
}
}  // namespace

const char* TierName(Tier t) { return t == Tier::CoopMat ? "coopmat" : t == Tier::Cuda ? "cuda" : "generic"; }

float Weights::W(uint32_t l, size_t i) const
{
    const LayerWeights& L = layers[l];
    if (L.fp8) return Fp8(data[L.weights + i]);
    uint16_t h;
    memcpy(&h, data + L.weights + 2 * i, 2);
    return Half(h);
}
float Weights::B(uint32_t l, uint32_t o) const
{
    float f;
    memcpy(&f, data + layers[l].bias + 4 * size_t(o), 4);
    return f;
}

bool Weights::Load(Weights& w, std::string* error)
{
    auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
    w.data = reinterpret_cast<const uint8_t*>(kWeights), w.size = kWeightBytes;
    if (const char* path = Env("NFRUX_AMD_WEIGHTS"))
    {
        FILE* f = fopen(path, "rb");
        if (!f) return fail(std::string("cannot read the weights file ") + path);
        fseek(f, 0, SEEK_END);
        w.file.resize(size_t(ftell(f)));
        fseek(f, 0, SEEK_SET);
        const size_t got = fread(w.file.data(), 1, w.file.size(), f);
        fclose(f);
        if (got != w.file.size()) return fail(std::string("cannot read the weights file ") + path);
        w.data = w.file.data(), w.size = w.file.size();
        Log("FSR FG4 network: weights from %s", path);
    }
    uint32_t head[4];
    if (w.size < 16) return fail("weights: too short");
    memcpy(head, w.data, 16);
    const auto want = Expected();
    if (head[0] != 0x47464C4Du || head[1] != 1 || head[2] != want.size() || head[3] != w.size || w.size < 16 + 32 * want.size()) return fail("weights: not a tools/mlfg_weights.py blob of this network");
    w.layers.resize(want.size());
    memcpy(w.layers.data(), w.data + 16, 32 * want.size());
    for (size_t i = 0; i < want.size(); ++i)
    {
        const LayerWeights& L = w.layers[i];
        const uint64_t n = uint64_t(L.k) * L.k * L.cin * L.cout;
        if (L.kind != want[i][0] || L.k != want[i][1] || L.stride != want[i][2] || L.cin != want[i][3] || L.cout != want[i][4] ||
            L.weights + n * (L.fp8 ? 1 : 2) > w.size || L.bias + 4ull * L.cout > w.size)
            return fail("weights: layer " + std::to_string(i) + " is not the expected one");
    }
    return true;
}

std::unique_ptr<Network> Network::Create(Device& dev, uint32_t l0w, uint32_t l0h, uint32_t s2w, uint32_t s2h, std::string* error)
{
    std::unique_ptr<Network> n(new Network(dev));
    if (!Weights::Load(n->m_weights, error) || !n->Make(l0w, l0h, s2w, s2h, error))
        return nullptr;
    return n;
}
Network::~Network()
{
    if (!m_marker.empty() && m_frames) MarkerOk(m_marker);  // an orderly end after frames were made: as the int8 runtime's markers
}

std::string Network::Describe() const
{
    return m_tier == Tier::CoopMat ? "cooperative matrices fp16 (Vulkan)" : m_tier == Tier::Cuda ? std::string(m_gpuW8 ? "CUDA kernels, fp16 tensor cores, FP8 for stage 2's FasterNet blocks" : "CUDA kernels, fp16 tensor cores") + (m_dev.caps().vulkan ? " (Vulkan)" : " (D3D12)") : "generic HLSL kernels (fp32, strict)";
}

bool Network::SelectTier(std::string* error)
{
    const Caps& caps = m_dev.caps();
    const bool  mat  = caps.vulkan && caps.coopMatF16 && (caps.subgroupSize == 16 || caps.subgroupSize == 32 || caps.subgroupSize == 64);
    const char* env  = Env("NFRUX_MLFG");  // tests: strict, no markers
    const char* ini  = env ? nullptr : Option(caps.vulkan ? "vulkan" : "d3d12", "amdnetwork");
    const char* want = env ? env : ini;
    m_tier = Tier::Generic;
    if (want && !strcmp(want, "generic")) return true;
    if (want && strcmp(want, "coopmat") && strcmp(want, "cuda"))
    {
        if (env) { if (error) *error = std::string("NFRUX_MLFG=") + want + ": not generic, coopmat or cuda"; return false; }
        Log("nfrux.ini: AmdNetwork=%s is not generic, coopmat or cuda: the automatic choice", want);
        want = nullptr;
    }
    // NVIDIA RTX: CUDA kernels on the tensor cores (D3D12: NVAPI launch; Vulkan: VK_NVX_binary_import), behind a crash marker
    // as the int8 runtime's CUDA; else coopmat / generic.
    const bool cudaWanted = want ? !strcmp(want, "cuda") : caps.nvidia;
    if (cudaWanted && m_dev.CudaAvailable())
    {
        std::string why;
        char        b[64];
        snprintf(b, sizeof(b), "%s_mlfg_cuda_%04x", caps.vulkan ? "vulkan" : "d3d12", caps.deviceId);
        if (env || MarkerBegin(b))
        {
            if ((m_cuda = CudaKernels::Create(m_dev, &why)))
            {
                if (!env) m_marker = b;
                m_tier = Tier::Cuda;
                return true;
            }
            if (!env) MarkerOk(b);
        }
        else why = "crash marker";
        if (env) { if (error) *error = "NFRUX_MLFG=cuda: " + why; return false; }
        Log("FSR FG4 network: CUDA not used (%s)", why.c_str());
        if (want) want = nullptr;
    }
    else if (want && !strcmp(want, "cuda"))
    {
        if (env) { if (error) *error = "NFRUX_MLFG=cuda: no CUDA launch on this device"; return false; }
        Log("nfrux.ini: AmdNetwork=cuda: no CUDA launch on this device: the automatic choice");
        want = nullptr;
    }
    if (want && !mat)
    {
        if (env) { if (error) *error = "NFRUX_MLFG=coopmat: no fp16 16x16x16 cooperative matrices here"; return false; }
        Log("nfrux.ini: AmdNetwork=coopmat cannot be used here (no fp16 16x16x16 cooperative matrices): generic");
        return true;
    }
    if (!mat) return true;
    // Off NVIDIA the cooperative matrices are untested (no such GPU here): a crash marker of their own, as the int8 runtime's.
    if (!env && !caps.nvidia)
    {
        char b[64];
        snprintf(b, sizeof(b), "vulkan_mlfg_coopmat_%04x_%04x", caps.vendorId, caps.deviceId);
        if (!MarkerBegin(b)) return true;
        m_marker = b;
    }
    m_tier = Tier::CoopMat;
    return true;
}

void Network::FrameDone()
{
    if (!m_marker.empty() && ++m_frames >= 120) MarkerOk(m_marker), m_marker.clear();
}

bool Network::Make(uint32_t l0w, uint32_t l0h, uint32_t s2w, uint32_t s2h, std::string* error)
{
    auto fail = [&](const char* m) { if (error) *error = m; return false; };
    if (l0w % 32 || l0h % 32 || s2w % 4 || s2h % 4) return fail("network sizes");
    if (!SelectTier(error)) return false;
    m_fused = !EnvFlag("NFRUX_MLFG_UNFUSED", false);
    m_m2 = !EnvFlag("NFRUX_MLFG_NO_M2", false);
    // The fused FasterNet kernels of the coopmat and generic tiers: slower than the layers on the RTX 4070 (coopmat +1.0 ms at
    // 4K, generic D3D12 +7.7 / Vulkan -0.3 ms at 1080p: one-subgroup groups, shared-memory round trips, register pressure);
    // untested elsewhere. Off unless NFRUX_MLFG_FUSED=1. CUDA's is on (NFRUX_MLFG_UNFUSED=1 turns every one off).
    m_fusedMat = m_fused && EnvFlag("NFRUX_MLFG_FUSED", false);
    m_offset.resize(kLayers), m_biasOffset.resize(kLayers), m_nb.resize(kLayers), m_off8.resize(kLayers);
    if (m_tier == Tier::Cuda)
    {
        // mma fragments: per layer [kc][tap][ntp][lane] words, word = (W[k = 8 kc + 2q][n], W[k + 1][n]), n = 8 nt + g, lane = 4g + q;
        // ntp = the n tiles rounded up to the kernel's 8 (transposed: 4). Biases fp32 [coPad64].
        std::vector<uint32_t> h;
        std::vector<float>    bias;
        for (uint32_t l = 0; l < kLayers; ++l)
        {
            const LayerWeights& L = m_weights.layers[l];
            m_nb[l] = CudaNb(L);
            const uint32_t taps = L.k * L.k, kc = (L.cin + 7) / 8, nb = m_nb[l], nt = (L.cout + 7) / 8, ntp = (nt + nb - 1) / nb * nb;
            const uint32_t coP = (L.cout + 63) & ~63u;
            m_offset[l] = uint32_t(h.size()), m_biasOffset[l] = uint32_t(bias.size());
            h.resize(h.size() + size_t(kc) * taps * ntp * 32, 0);
            bias.resize(bias.size() + std::max(coP, ntp * 8), 0.0f);
            uint32_t* w = h.data() + m_offset[l];
            auto at = [&](uint32_t t, uint32_t i, uint32_t o) -> uint16_t {
                if (i >= L.cin || o >= L.cout) return 0;
                return HalfBits(m_weights.W(l, L.kind == 0 ? (size_t(o) * L.cin + i) * taps + t : (size_t(i) * L.cout + o) * taps + t));
            };
            for (uint32_t c = 0; c < kc; ++c)
                for (uint32_t t = 0; t < taps; ++t)
                    for (uint32_t n = 0; n < ntp; ++n)
                        for (uint32_t lane = 0; lane < 32; ++lane)
                        {
                            const uint32_t g = lane >> 2, q = lane & 3, k = c * 8 + 2 * q, o = n * 8 + g;
                            w[((size_t(c) * taps + t) * ntp + n) * 32 + lane] = at(t, k, o) | (uint32_t(at(t, k + 1, o)) << 16);
                        }
            for (uint32_t o = 0; o < L.cout; ++o) bias[m_biasOffset[l] + o] = m_weights.B(l, o);
        }
        // FP8 tensor cores: the FasterNet layers' weights again, as e4m3 B fragments of 32-channel k chunks ([chunk][n][lane]
        // two words; channel order Fp8Channel; pconv P = 16: two taps a chunk).
        if (m_cuda->Fp8())
        {
            for (uint32_t l = kS2B1; l < kS2Up; ++l)
            {
                if (l == kS2D2 || l == kS2B2 + 15) continue;  // d2, up: not in a fused block
                const LayerWeights& L = m_weights.layers[l];
                const uint32_t taps = L.k * L.k, nt = L.cout / 8;
                const bool     pconv = L.k == 3, two = pconv && L.cin == 16;
                const uint32_t chunks = pconv ? (two ? 5 : 9) : (L.cin + 31) / 32;
                m_off8[l] = uint32_t(m_blob8.size());
                m_blob8.resize(m_blob8.size() + size_t(chunks) * nt * 32 * 8, 0);
                for (uint32_t c = 0; c < chunks; ++c)
                    for (uint32_t n = 0; n < nt; ++n)
                        for (uint32_t lane = 0; lane < 32; ++lane)
                            for (uint32_t i = 0; i < 8; ++i)
                            {
                                const uint32_t kp = (i < 4 ? 4 * (lane & 3) + i : 16 + 4 * (lane & 3) + i - 4), lc = uint32_t(Fp8Channel(int(kp)));
                                uint32_t t = 0, ci = 32 * c + lc;
                                if (pconv) t = two ? 2 * c + lc / 16 : c, ci = two ? lc % 16 : lc;
                                if (t >= taps || ci >= L.cin || !L.fp8) continue;
                                const uint32_t o = 8 * n + (lane >> 2);
                                m_blob8[m_off8[l] + ((size_t(c) * nt + n) * 32 + lane) * 8 + i] = m_weights.data[L.weights + (size_t(o) * L.cin + ci) * taps + t];
                            }
            }
            if (!(m_gpuW8 = m_dev.CreateBuffer(m_blob8.size(), true))) return fail("weights buffer");
        }
        m_blob.resize(h.size() * 4), memcpy(m_blob.data(), h.data(), m_blob.size());
        m_biasBlob.resize(bias.size() * 4), memcpy(m_biasBlob.data(), bias.data(), m_biasBlob.size());
        m_gpuWeights = m_dev.CreateBuffer(m_blob.size(), true), m_gpuBias = m_dev.CreateBuffer(m_biasBlob.size(), true);
        if (!m_gpuWeights || !m_gpuBias) return fail("weights buffer");
    }
    else if (m_tier == Tier::CoopMat)
    {
        // fp16 [tap][ciPad16][coPad64], biases fp32 [coPad64].
        std::vector<uint16_t> h;
        std::vector<float>    bias;
        for (uint32_t l = 0; l < kLayers; ++l)
        {
            const LayerWeights& L = m_weights.layers[l];
            const uint32_t taps = L.k * L.k, ciP = (L.cin + 15) & ~15u, coP = (L.cout + 63) & ~63u;
            m_offset[l] = uint32_t(h.size()), m_biasOffset[l] = uint32_t(bias.size());
            h.resize(h.size() + size_t(taps) * ciP * coP, 0);
            bias.resize(bias.size() + coP, 0.0f);
            uint16_t* w = h.data() + m_offset[l];
            for (uint32_t o = 0; o < L.cout; ++o)
                for (uint32_t i = 0; i < L.cin; ++i)
                    for (uint32_t t = 0; t < taps; ++t)
                    {
                        const size_t src = L.kind == 0 ? (size_t(o) * L.cin + i) * taps + t : (size_t(i) * L.cout + o) * taps + t;
                        w[(size_t(t) * ciP + i) * coP + o] = HalfBits(m_weights.W(l, src));
                    }
            for (uint32_t o = 0; o < L.cout; ++o) bias[m_biasOffset[l] + o] = m_weights.B(l, o);
        }
        m_blob.resize(h.size() * 2), memcpy(m_blob.data(), h.data(), m_blob.size());
        m_biasBlob.resize(bias.size() * 4), memcpy(m_biasBlob.data(), bias.data(), m_biasBlob.size());
        m_gpuWeights = m_dev.CreateBuffer(m_blob.size()), m_gpuBias = m_dev.CreateBuffer(m_biasBlob.size());
        const uint32_t sg = m_dev.caps().subgroupSize;
        const Shader* sh[4] = {sg == 16 ? &shaders::kMlfgMat3_16 : sg == 32 ? &shaders::kMlfgMat3_32 : &shaders::kMlfgMat3_64,
                               sg == 16 ? &shaders::kMlfgMat2_16 : sg == 32 ? &shaders::kMlfgMat2_32 : &shaders::kMlfgMat2_64,
                               sg == 16 ? &shaders::kMlfgMat1_16 : sg == 32 ? &shaders::kMlfgMat1_32 : &shaders::kMlfgMat1_64,
                               sg == 16 ? &shaders::kMlfgMatT_16 : sg == 32 ? &shaders::kMlfgMatT_32 : &shaders::kMlfgMatT_64};
        bool made = m_gpuWeights && m_gpuBias;
        for (int i = 0; i < 4; ++i) made = made && (m_mat[i] = m_dev.CreatePass(*sh[i]));
        made = made && (m_fn[0] = m_dev.CreatePass(sg == 16 ? shaders::kMlfgFn48_16 : sg == 32 ? shaders::kMlfgFn48_32 : shaders::kMlfgFn48_64));
        made = made && (m_fn[1] = m_dev.CreatePass(sg == 16 ? shaders::kMlfgFn64_16 : sg == 32 ? shaders::kMlfgFn64_32 : shaders::kMlfgFn64_64));
        made = made && (m_matN1[0] = m_dev.CreatePass(sg == 16 ? shaders::kMlfgMat3n1_16 : sg == 32 ? shaders::kMlfgMat3n1_32 : shaders::kMlfgMat3n1_64));
        made = made && (m_matN1[1] = m_dev.CreatePass(sg == 16 ? shaders::kMlfgMatTn1_16 : sg == 32 ? shaders::kMlfgMatTn1_32 : shaders::kMlfgMatTn1_64));
        if (!made) return fail("cooperative-matrix passes or weights buffer");
    }
    else
    {
    // Generic tier: fp32 weights, [tap][cin][coPad] then the bias [coPad], coPad = cout rounded up to 8.
    std::vector<float> f;
    for (uint32_t l = 0; l < kLayers; ++l)
    {
        const LayerWeights& L = m_weights.layers[l];
        const uint32_t taps = L.k * L.k, coP = (L.cout + 7) & ~7u;
        m_offset[l] = uint32_t(f.size());
        f.resize(f.size() + size_t(taps) * L.cin * coP + coP, 0.0f);
        float* w = f.data() + m_offset[l];
        for (uint32_t o = 0; o < L.cout; ++o)
            for (uint32_t i = 0; i < L.cin; ++i)
                for (uint32_t t = 0; t < taps; ++t)
                {
                    const size_t src = L.kind == 0 ? (size_t(o) * L.cin + i) * taps + t : (size_t(i) * L.cout + o) * taps + t;
                    w[(size_t(t) * L.cin + i) * coP + o] = m_weights.W(l, src);
                }
        for (uint32_t o = 0; o < L.cout; ++o) w[size_t(taps) * L.cin * coP + o] = m_weights.B(l, o);
    }
    f.resize(f.size() + 64, 0.0f);  // the last chunk's loads stay inside
    m_blob.resize(f.size() * 4);
    memcpy(m_blob.data(), f.data(), m_blob.size());
    m_gpuWeights = m_dev.CreateBuffer(m_blob.size());
    const Shader* sh[4][2] = {{&shaders::kMlfgConv3, &shaders::kMlfgConv3s}, {&shaders::kMlfgConv2, &shaders::kMlfgConv2s},
                              {&shaders::kMlfgConv1, &shaders::kMlfgConv1s}, {&shaders::kMlfgConvT, &shaders::kMlfgConvTs}};
    bool made = m_gpuWeights != nullptr;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) made = made && (m_conv[i][j] = m_dev.CreatePass(*sh[i][j]));
    made = made && (m_genFn[0] = m_dev.CreatePass(shaders::kMlfgConvFn48)) && (m_genFn[1] = m_dev.CreatePass(shaders::kMlfgConvFn64));
    if (!made) return fail("network passes or weights buffer");
    }
    bool ok = true;
    auto tensor = [&](Tensor& t, uint32_t w, uint32_t h, uint32_t pitch) {
        t.w = w, t.h = h, t.pitch = pitch;
        t.buf = m_dev.CreateBuffer(uint64_t(w) * h * pitch * 2, m_tier == Tier::Cuda);
        ok = ok && t.buf;
    };
    // Stage-1 scratch, used at many sizes (views): the largest of each role.
    tensor(m_a, l0w / 2, l0h / 2, 48), tensor(m_b, l0w / 2, l0h / 2, 48);
    tensor(m_c, l0w / 4, l0h / 4, 96), tensor(m_d, l0w / 4, l0h / 4, 96);
    tensor(m_t32, s2w, s2h, 32);
    tensor(m_x48, s2w / 2, s2h / 2, 48), tensor(m_y48, s2w / 2, s2h / 2, 48), tensor(m_pc16, s2w / 2, s2h / 2, 16), tensor(m_h96, s2w / 2, s2h / 2, 96);
    tensor(m_x64, s2w / 4, s2h / 4, 64), tensor(m_y64, s2w / 4, s2h / 4, 64), tensor(m_pc32, s2w / 4, s2h / 4, 32), tensor(m_h128, s2w / 4, s2h / 4, 128);
    return ok || fail("out of memory");
}

bool Network::Prepare()
{
    if (m_uploaded) return true;
    if (!m_dev.Upload(*m_gpuWeights, m_blob.data(), m_blob.size())) return false;
    if (m_gpuBias && !m_dev.Upload(*m_gpuBias, m_biasBlob.data(), m_biasBlob.size())) return false;
    if (m_gpuW8 && !m_dev.Upload(*m_gpuW8, m_blob8.data(), m_blob8.size())) return false;
    std::vector<uint8_t>().swap(m_blob8);
    m_uploaded = true;
    std::vector<uint8_t>().swap(m_blob), std::vector<uint8_t>().swap(m_biasBlob);
    return true;
}

bool Network::Run(const Conv& c)
{
    const LayerWeights& L = m_weights.layers[c.layer];
    char label[48] = "";
    if (m_dev.Timing()) snprintf(label, sizeof(label), "L%03u %s%ux%u %u->%u", c.layer, L.kind ? "T" : "k", L.k, L.stride, L.cin, L.cout);
    if (label[0] && m_tier != Tier::Cuda) m_dev.Label(label);  // CUDA launches mark themselves
    const uint32_t writeEnd = c.writeEnd ? c.writeEnd : (L.cout + 1) & ~1u;
    if (!c.src || !c.out || (writeEnd & 1) || (c.out.ch & 1) || c.out.ch + writeEnd > c.out.pitch) return false;
    const View& s1 = c.src1 ? c.src1 : c.src;
    if (m_tier == Tier::Cuda)
    {
        const uint32_t split = std::min(c.split, L.cin), nb = m_nb[c.layer], nt = (std::max(L.cout, writeEnd) + 7) / 8, ntp = ((L.cout + 7) / 8 + nb - 1) / nb * nb;
        if (nt > ntp || (split < L.cin && split % 8)) return false;
        const Buffer* bufs[] = {c.src.buf, s1.buf, m_gpuWeights.get(), m_gpuBias.get(), c.add ? c.add.buf : c.src.buf, c.out.buf};
        CudaParams p = {m_dev.NativeAddress(*bufs[0]), m_dev.NativeAddress(*bufs[1]), m_dev.NativeAddress(*bufs[2]) + uint64_t(m_offset[c.layer]) * 4,
                        m_dev.NativeAddress(*bufs[3]) + uint64_t(m_biasOffset[c.layer]) * 4, m_dev.NativeAddress(*bufs[4]), m_dev.NativeAddress(*bufs[5]),
                        int32_t(c.src.w), int32_t(c.src.h), int32_t(c.src.pitch), int32_t(c.src.ch), int32_t(s1.pitch), int32_t(s1.ch) - int32_t(split), int32_t(split), int32_t(L.cin),
                        int32_t(c.out.w), int32_t(c.out.h), int32_t(c.out.pitch), int32_t(c.out.ch), int32_t(L.cout), int32_t(writeEnd), int32_t(c.act), c.fp8 ? 1 : 0,
                        c.add ? int32_t(c.add.pitch) : 0, int32_t(c.add.ch), int32_t((L.cin + 7) / 8), int32_t(ntp)};
        const int      shape = L.kind == 1 ? 3 : L.k == 3 ? 0 : L.k == 2 ? 1 : 2;
        const uint32_t grid[3] = {L.kind == 1 ? (c.src.w + 15) / 16 : (c.out.w + 15) / 16, ((L.kind == 1 ? c.src.h : c.out.h) + kCudaWarps - 1) / kCudaWarps, ntp / nb};
        // 32 pixels a warp (each B fragment twice) for the 8-tile convolutions without a second source on 3x3 / 2x2.
        // Only where that still gives 256 warps or more (measured: the 30- and 60-pixel-wide stage-1 layers get slower).
        const bool m2 = nb == 8 && L.kind == 0 && m_m2 && (L.k == 1 || split >= L.cin) && uint64_t((c.out.w + 31) / 32) * c.out.h * (ntp / nb) >= 256;
        const uint32_t gridM[3] = {(c.out.w + 31) / 32, grid[1], grid[2]};
        const bool ok = m2 ? m_cuda->LaunchM(bufs, 6, shape, gridM, p) : m_cuda->Launch(bufs, 6, shape, nb, grid, p);
        if (label[0]) m_dev.Mark(label);
        return ok;
    }
    if (m_tier == Tier::CoopMat)
    {
        const uint32_t split = std::min(c.split, L.cin);
        const bool     aligned = L.cin % 16 == 0 && split % 16 == 0 && c.src.pitch % 8 == 0 && c.src.ch % 8 == 0 && s1.pitch % 8 == 0 && (s1.ch + 16 - split % 16) % 8 == 0 && s1.ch % 8 == 0;
        struct { int32_t in[4], in2[4], out[4], cfg[4], add[4], x[4]; } k = {
            {int32_t(c.src.w), int32_t(c.src.h), int32_t(c.src.pitch), int32_t(std::min(c.split, L.cin))},
            {int32_t(c.src.ch), int32_t(s1.pitch), int32_t(s1.ch), int32_t(L.cin)},
            {int32_t(c.out.w), int32_t(c.out.h), int32_t(c.out.pitch), int32_t(c.out.ch)},
            {int32_t(L.cout), int32_t(c.act), c.fp8 ? 1 : 0, int32_t(writeEnd)},
            {c.add ? int32_t(c.add.pitch) : 0, int32_t(c.add.ch), int32_t(m_offset[c.layer]), int32_t((L.cout + 63) & ~63u)},
            {int32_t(m_biasOffset[c.layer]), int32_t((L.cin + 15) & ~15u), aligned ? 1 : 0, 0}};
        const int shape = L.kind == 1 ? 3 : L.k == 3 ? 0 : L.k == 2 ? 1 : 2;
        const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Buf(1, *c.src.buf), Bind::Buf(2, *s1.buf), Bind::Buf(3, *m_gpuWeights),
                          Bind::Buf(4, c.add ? *c.add.buf : *c.src.buf), Bind::Buf(5, *c.out.buf), Bind::Buf(6, *m_gpuBias)};
        const bool narrow = L.cout <= 16 && (L.kind == 1 || L.k == 3);  // one 16-channel tile a group
        if (L.kind == 1) return (narrow ? m_matN1[1] : m_mat[3])->Dispatch(b, 7, (c.src.w + 15) / 16, c.src.h, narrow ? 1 : (L.cout + 31) / 32);
        return (narrow ? m_matN1[0] : m_mat[shape])->Dispatch(b, 7, (c.out.w + 15) / 16, c.out.h, narrow ? 1 : (L.cout + 63) / 64);
    }
    const uint32_t gsplit = std::min(c.split, L.cin);
    const bool vec = c.src.pitch % 8 == 0 && c.src.ch % 8 == 0 && (gsplit >= L.cin || (gsplit % 8 == 0 && s1.pitch % 8 == 0 && s1.ch % 8 == 0));
    struct { int32_t in[4], in2[4], out[4], cfg[4], add[4], x[4]; } k = {
        {int32_t(c.src.w), int32_t(c.src.h), int32_t(c.src.pitch), int32_t(std::min(c.split, L.cin))},
        {int32_t(c.src.ch), int32_t(s1.pitch), int32_t(s1.ch), int32_t(L.cin)},
        {int32_t(c.out.w), int32_t(c.out.h), int32_t(c.out.pitch), int32_t(c.out.ch)},
        {int32_t(L.cout), int32_t(c.act), c.fp8 ? 1 : 0, int32_t(writeEnd)},
        {c.add ? int32_t(c.add.pitch) : 0, int32_t(c.add.ch), int32_t(m_offset[c.layer]), int32_t((L.cout + 7) & ~7u)}, {vec ? 1 : 0, 0, 0, 0}};
    // Shape, and tile: 16 x 16 (16 x 8 for 2x2/2) pixels a group, or 4 x 4 with the channels in 4 slices where the large
    // tile would give few groups (the low-resolution layers). Chosen from sizes only: the same on every API.
    const int      shape = L.kind == 1 ? 3 : L.k == 3 ? 0 : L.k == 2 ? 1 : 2;
    const uint32_t zs = (std::max(L.cout, writeEnd) + 7) / 8, th = shape == 1 ? 8 : 16;
    const bool     small = uint64_t((c.out.w + 15) / 16) * ((c.out.h + th - 1) / th) * zs < 768;
    Pass*          p = m_conv[shape][small ? 1 : 0].get();
    const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Buf(1, *c.src.buf), Bind::Buf(2, *s1.buf), Bind::Buf(3, *m_gpuWeights),
                      Bind::Buf(4, c.add ? *c.add.buf : *c.src.buf), Bind::Buf(5, *c.out.buf)};
    return small ? p->Dispatch(b, 6, (c.out.w + 3) / 4, (c.out.h + 3) / 4, zs) : p->Dispatch(b, 6, (c.out.w + 15) / 16, (c.out.h + th - 1) / th, zs);
}

namespace
{
View Sized(const Tensor& t, uint32_t w, uint32_t h, uint32_t pitch) { return {t.buf.get(), w, h, pitch, 0}; }
}

bool Network::Encoders(const View& netIn)
{
    const uint32_t w = netIn.w / 2, h = netIn.h / 2;
    const View a = Sized(m_a, w, h, 32), b = Sized(m_b, w, h, 32);
    for (uint32_t e = 0; e < 2; ++e)
    {
        const uint32_t first = e ? kEncB : kEncA;
        Conv c;
        c.layer = first, c.src = netIn.At(e ? 3 : 0), c.out = a, c.act = Leaky;
        if (!Run(c)) return false;
        c = {}, c.layer = first + 1, c.src = a, c.out = b, c.act = Leaky;
        if (!Run(c)) return false;
        c = {}, c.layer = first + 2, c.src = b, c.out = a, c.act = Leaky;
        if (!Run(c)) return false;
        c = {}, c.layer = first + 3, c.src = a, c.out = netIn.At(e ? 12 : 6), c.writeEnd = 6;
        if (!Run(c)) return false;
    }
    return true;
}

bool Network::FlowBlock(uint32_t block, const View& in, const View& out13)
{
    const uint32_t c1 = kBlocks[block][1], c2 = kBlocks[block][2], first = kBlock0 + 13 * block;
    const View a = Sized(m_a, in.w / 2, in.h / 2, c1), b = Sized(m_b, in.w / 2, in.h / 2, c1);
    const View x = Sized(m_c, in.w / 4, in.h / 4, c2), y = Sized(m_d, in.w / 4, in.h / 4, c2);
    Conv c;
    c.layer = first, c.src = in, c.out = a, c.act = Leaky;
    if (!Run(c)) return false;
    c = {}, c.layer = first + 1, c.src = a, c.out = b, c.act = Leaky;
    if (!Run(c)) return false;
    c = {}, c.layer = first + 2, c.src = b, c.out = x, c.act = Leaky;
    if (!Run(c)) return false;
    for (uint32_t r = 0; r < 8; ++r)  // lrelu(conv(x) + x), ping-pong: ends in x
    {
        const View& s = r % 2 ? y : x;
        const View& d = r % 2 ? x : y;
        c = {}, c.layer = first + 3 + r, c.src = s, c.add = s, c.out = d, c.act = Leaky;
        if (!Run(c)) return false;
    }
    c = {}, c.layer = first + 11, c.src = x, c.out = a, c.act = Leaky;
    if (!Run(c)) return false;
    c = {}, c.layer = first + 12, c.src = a, c.out = out13, c.writeEnd = 16;
    return Run(c);
}

// AMD's FasterNet block: y = q8(project(q8(relu(expand(cat(q8(pconv(x[:P])), x[P:]))))) + x); with up: q8(up(y) + skip).
bool Network::FasterNet(uint32_t first, uint32_t C, uint32_t P, const View& x, const View& y, const View& pc, const View& h, const View* skip, const View* upOut)
{
    Conv c;
    if (m_tier == Tier::Cuda && m_fused && x.pitch == C && y.pitch == C && !x.ch && !y.ch)
    {
        // One kernel for pconv, expand and project: the same products, order and roundings as the three layers.
        const bool    f8 = m_gpuW8 != nullptr;
        const Buffer* bufs[] = {x.buf, y.buf, f8 ? m_gpuW8.get() : m_gpuWeights.get(), m_gpuBias.get()};
        const uint64_t wa = m_dev.NativeAddress(*bufs[2]), ba = m_dev.NativeAddress(*m_gpuBias);
        auto wo = [&](uint32_t l) { return f8 ? wa + m_off8[l] : wa + uint64_t(m_offset[l]) * 4; };
        const FnParams p = {m_dev.NativeAddress(*x.buf), m_dev.NativeAddress(*y.buf), wo(first), wo(first + 1), wo(first + 2), ba + uint64_t(m_biasOffset[first]) * 4, ba + uint64_t(m_biasOffset[first + 1]) * 4,
                            ba + uint64_t(m_biasOffset[first + 2]) * 4, int32_t(x.w), int32_t(x.h)};
        const uint32_t grid[3] = {(x.w + 15) / 16, (x.h + kCudaWarps - 1) / kCudaWarps, 1};
        if (!m_cuda->LaunchFn(bufs, 4, C, f8, grid, p)) return false;
        if (m_dev.Timing()) m_dev.Mark(C == 48 ? (f8 ? "FasterNet 48 (fused, fp8)" : "FasterNet 48 (fused)") : (f8 ? "FasterNet 64 (fused, fp8)" : "FasterNet 64 (fused)"));
        if (!skip) return true;
        c.layer = first + 3, c.src = y, c.add = *skip, c.out = *upOut, c.fp8 = true;
        return Run(c);
    }
    if (m_tier == Tier::Generic && m_fusedMat && x.pitch == C && y.pitch == C && !x.ch && !y.ch)
    {
        auto pad = [&](uint32_t l) { return int32_t((m_weights.layers[l].cout + 7) & ~7u); };
        const int32_t k[12] = {int32_t(x.w), int32_t(x.h), 0, 0, int32_t(m_offset[first]), int32_t(m_offset[first + 1]), int32_t(m_offset[first + 2]), 0,
                               pad(first), pad(first + 1), pad(first + 2), 0};
        const Bind b[] = {Bind::Data(0, k, sizeof(k)), Bind::Buf(1, *x.buf), Bind::Buf(3, *m_gpuWeights), Bind::Buf(5, *y.buf)};
        if (m_dev.Timing()) m_dev.Label(C == 48 ? "FasterNet 48 (fused)" : "FasterNet 64 (fused)");
        if (!m_genFn[C == 48 ? 0 : 1]->Dispatch(b, 4, (x.w + 7) / 8, (x.h + 7) / 8, 1)) return false;
        if (!skip) return true;
        c.layer = first + 3, c.src = y, c.add = *skip, c.out = *upOut, c.fp8 = true;
        return Run(c);
    }
    if (m_tier == Tier::CoopMat && m_fusedMat && x.pitch == C && y.pitch == C && !x.ch && !y.ch)
    {
        auto pad = [&](uint32_t l) { return int32_t((m_weights.layers[l].cout + 63) & ~63u); };
        const int32_t k[16] = {int32_t(x.w), int32_t(x.h), 0, 0, int32_t(m_offset[first]), int32_t(m_offset[first + 1]), int32_t(m_offset[first + 2]), 0,
                               int32_t(m_biasOffset[first]), int32_t(m_biasOffset[first + 1]), int32_t(m_biasOffset[first + 2]), 0, pad(first), pad(first + 1), pad(first + 2), 0};
        const Bind b[] = {Bind::Data(0, k, sizeof(k)), Bind::Buf(1, *x.buf), Bind::Buf(3, *m_gpuWeights), Bind::Buf(5, *y.buf), Bind::Buf(6, *m_gpuBias)};
        if (m_dev.Timing()) m_dev.Label(C == 48 ? "FasterNet 48 (fused)" : "FasterNet 64 (fused)");
        if (!m_fn[C == 48 ? 0 : 1]->Dispatch(b, 5, (x.w + 15) / 16, x.h, 1)) return false;
        if (!skip) return true;
        c.layer = first + 3, c.src = y, c.add = *skip, c.out = *upOut, c.fp8 = true;
        return Run(c);
    }
    c.layer = first, c.src = x, c.out = pc, c.fp8 = true;
    if (!Run(c)) return false;
    c = {}, c.layer = first + 1, c.src = pc, c.src1 = x.At(P), c.split = P, c.out = h, c.act = Relu, c.fp8 = true;
    if (!Run(c)) return false;
    c = {}, c.layer = first + 2, c.src = h, c.add = x, c.out = y, c.fp8 = true;
    if (!Run(c)) return false;
    if (!skip) return true;
    c = {}, c.layer = first + 3, c.src = y, c.add = *skip, c.out = *upOut, c.fp8 = true;
    return Run(c);
}

bool Network::Stage2(const View& in, const View& logits)
{
    const uint32_t W = in.w, H = in.h;
    const View t32 = Sized(m_t32, W, H, 32);
    const View x48 = Sized(m_x48, W / 2, H / 2, 48), y48 = Sized(m_y48, W / 2, H / 2, 48), pc16 = Sized(m_pc16, W / 2, H / 2, 16), h96 = Sized(m_h96, W / 2, H / 2, 96);
    const View x64 = Sized(m_x64, W / 4, H / 4, 64), y64 = Sized(m_y64, W / 4, H / 4, 64), pc32 = Sized(m_pc32, W / 4, H / 4, 32), h128 = Sized(m_h128, W / 4, H / 4, 128);
    Conv c;
    c.layer = kS2Inp, c.src = in, c.out = t32, c.act = Relu;  // fp16
    if (!Run(c)) return false;
    c = {}, c.layer = kS2D1, c.src = t32, c.out = x48, c.fp8 = true;
    if (!Run(c)) return false;
    for (uint32_t j = 0; j < 4; ++j)  // ends in x48: the skip
        if (!FasterNet(kS2B1 + 3 * j, 48, 16, j % 2 ? y48 : x48, j % 2 ? x48 : y48, pc16, h96, nullptr, nullptr)) return false;
    c = {}, c.layer = kS2D2, c.src = x48, c.out = x64, c.fp8 = true;
    if (!Run(c)) return false;
    for (uint32_t j = 0; j < 4; ++j)  // ends in x64
        if (!FasterNet(kS2B2 + 3 * j, 64, 32, j % 2 ? y64 : x64, j % 2 ? x64 : y64, pc32, h128, nullptr, nullptr)) return false;
    if (!FasterNet(kS2B2 + 12, 64, 32, x64, y64, pc32, h128, &x48, &y48)) return false;  // up + skip -> y48
    for (uint32_t j = 0; j < 4; ++j)  // ends in y48
        if (!FasterNet(kS2B3 + 3 * j, 48, 16, j % 2 ? x48 : y48, j % 2 ? y48 : x48, pc16, h96, nullptr, nullptr)) return false;
    c = {}, c.layer = kS2Up, c.src = y48, c.out = t32;  // fp16
    if (!Run(c)) return false;
    c = {}, c.layer = kS2Head, c.src = t32, c.out = logits;  // fp16 logits (the softmax pass follows)
    return Run(c);
}
}  // namespace nfrux::mlfg
