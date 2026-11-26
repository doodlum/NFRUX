// SPDX-License-Identifier: MIT
#include "mlfg_network.h"

#include "amd_mlfg_weights.h"
#include "log.h"
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
float Fp8(uint8_t b)
{
    const uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
    float f = e == 0 ? float(m) * 0.001953125f : (e == 15 && m == 7) ? NAN : ldexpf(1.0f + float(m) / 8.0f, int(e) - 7);
    return (b & 0x80) ? -f : f;
}
}  // namespace

const char* TierName(Tier t) { return t == Tier::CoopMat ? "coopmat" : "generic"; }

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
Network::~Network() = default;

std::string Network::Describe() const
{
    return m_tier == Tier::CoopMat ? "cooperative matrices fp16 (Vulkan)" : "generic HLSL kernels (fp32, strict)";
}

bool Network::Make(uint32_t l0w, uint32_t l0h, uint32_t s2w, uint32_t s2h, std::string* error)
{
    auto fail = [&](const char* m) { if (error) *error = m; return false; };
    if (l0w % 32 || l0h % 32 || s2w % 4 || s2h % 4) return fail("network sizes");
    // Generic tier: fp32 weights, [tap][cin][coPad] then the bias [coPad], coPad = cout rounded up to 8.
    std::vector<float> f;
    m_offset.resize(kLayers);
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
    m_conv3 = m_dev.CreatePass(shaders::kMlfgConv3), m_conv2 = m_dev.CreatePass(shaders::kMlfgConv2);
    m_conv1 = m_dev.CreatePass(shaders::kMlfgConv1), m_convT = m_dev.CreatePass(shaders::kMlfgConvT);
    if (!m_gpuWeights || !m_conv3 || !m_conv2 || !m_conv1 || !m_convT) return fail("network passes or weights buffer");
    bool ok = true;
    auto tensor = [&](Tensor& t, uint32_t w, uint32_t h, uint32_t pitch) {
        t.w = w, t.h = h, t.pitch = pitch;
        t.buf = m_dev.CreateBuffer(uint64_t(w) * h * pitch * 2);
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
    m_uploaded = true;
    std::vector<uint8_t>().swap(m_blob);
    return true;
}

bool Network::Run(const Conv& c)
{
    const LayerWeights& L = m_weights.layers[c.layer];
    const uint32_t writeEnd = c.writeEnd ? c.writeEnd : (L.cout + 1) & ~1u;
    if (!c.src || !c.out || (writeEnd & 1) || (c.out.ch & 1) || c.out.ch + writeEnd > c.out.pitch) return false;
    const View& s1 = c.src1 ? c.src1 : c.src;
    struct { int32_t in[4], in2[4], out[4], cfg[4], add[4]; } k = {
        {int32_t(c.src.w), int32_t(c.src.h), int32_t(c.src.pitch), int32_t(std::min(c.split, L.cin))},
        {int32_t(c.src.ch), int32_t(s1.pitch), int32_t(s1.ch), int32_t(L.cin)},
        {int32_t(c.out.w), int32_t(c.out.h), int32_t(c.out.pitch), int32_t(c.out.ch)},
        {int32_t(L.cout), int32_t(c.act), c.fp8 ? 1 : 0, int32_t(writeEnd)},
        {c.add ? int32_t(c.add.pitch) : 0, int32_t(c.add.ch), int32_t(m_offset[c.layer]), int32_t((L.cout + 7) & ~7u)}};
    Pass* p = L.kind == 1 ? m_convT.get() : L.k == 3 ? m_conv3.get() : L.k == 2 ? m_conv2.get() : m_conv1.get();
    const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Buf(1, *c.src.buf), Bind::Buf(2, *s1.buf), Bind::Buf(3, *m_gpuWeights),
                      Bind::Buf(4, c.add ? *c.add.buf : *c.src.buf), Bind::Buf(5, *c.out.buf)};
    return p->Dispatch(b, 6, (c.out.w + 7) / 8, (c.out.h + 7) / 8, (std::max(L.cout, writeEnd) + 7) / 8);
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
