// NFRUX's second interpolator: AMD FSR Frame Generation 4's ML frame interpolation (AMD FidelityFX SDK v2.4.0, MIT;
// docs/FFX_MLFG.md), behind the same interface as the NFRU one. Spec: tools/ffx_extract/mlfg_net.py (AmdMlfgNet,
// precision 'amd', sampler 'point' = AMD's nearest sampler, flow_t 'scale').
//
// Per rendered frame: prepass (motion-vector dilation of the current frame, the splat of its vectors onto the previous
// frame, the previous frame's forward motion from the camera matrices), then stage 1 once, at t = 0.5 (the timestep
// AMD's network was trained at): pass 0, the two encoders, four coarse-to-fine flow blocks with the glue passes between.
// Per generated frame at t = (i + 1) / (n + 1): the final flows scaled by 2t (to frame 0) and 2(1 - t) (to frame 1)
// (mlfg_net flow_t 'scale', best at 4x), the two splats projected to t, the stage-2 input, stage 2 and its softmax, the
// composite, and NFRUX's UI rules (kMlfgFinish). History: the previous colour, the previous frame's dilated vectors
// and depth, and the last two view-projection matrices.
// SPDX-License-Identifier: MIT
#include "interpolator.h"
#include "log.h"
#include "mlfg_constants.h"
#include "mlfg_network.h"
#include "nfrux_shaders.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nfrux
{
namespace
{
using Tex     = std::unique_ptr<Texture>;
using PassPtr = std::unique_ptr<Pass>;
uint32_t G(uint32_t n) { return (n + 7) / 8; }
// A float's bits in an int constant (the passes' uniform quotients, correctly rounded here: IEEE division).
int32_t Bits(float f) { int32_t i; memcpy(&i, &f, 4); return i; }

// Column-major 4x4 (m[col * 4 + row]), as the NFRU interpolator's; inverse in double.
bool Inverse(const float m[16], float out[16])
{
    double a[4][8] = {};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) a[r][c] = m[c * 4 + r], a[r][4 + r] = 1;
    for (int i = 0; i < 4; ++i)
    {
        int p = i;
        for (int r = i + 1; r < 4; ++r) if (fabs(a[r][i]) > fabs(a[p][i])) p = r;
        if (fabs(a[p][i]) < 1e-30) return false;
        for (int c = 0; c < 8; ++c) std::swap(a[i][c], a[p][c]);
        const double d = a[i][i];
        for (int c = 0; c < 8; ++c) a[i][c] /= d;
        for (int r = 0; r < 4; ++r)
            if (r != i)
            {
                const double f = a[r][i];
                for (int c = 0; c < 8; ++c) a[r][c] -= f * a[i][c];
            }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out[c * 4 + r] = float(a[r][4 + c]);
    return true;
}
void Multiply(const float* a, const float* b, float* out)
{
    for (uint32_t col = 0; col < 4; ++col)
        for (uint32_t row = 0; row < 4; ++row)
        {
            double sum = 0;
            for (uint32_t k = 0; k < 4; ++k) sum += double(a[k * 4 + row]) * b[col * 4 + k];
            out[col * 4 + row] = float(sum);
        }
}
// to-clip-of-b from clip-of-a for column vectors (VP_b * inv(VP_a)); its column-major numbers are the rows of AMD's
// row-vector matrix (clip' = [x y z 1] * M), which is how kMlfgForwardMv takes it.
void ClipToClip(const float* vpA, const float* vpB, float* out)
{
    float inv[16];
    if (!Inverse(vpA, inv)) { memset(out, 0, 64); for (int i = 0; i < 4; ++i) out[i * 5] = 1; return; }
    Multiply(vpB, inv, out);
}

struct Sizes
{
    uint32_t W, H, l0w, l0h, livew, liveh, s2w, s2h, prew, preh;
    Sizes(uint32_t w, uint32_t h)
        : W(w), H(h), l0w((w + 255) / 256 * 32), l0h((h + 255) / 256 * 32), livew(w / 8), liveh(h / 8), s2w((w + 15) / 16 * 4), s2h((h + 15) / 16 * 4),
          prew(std::max(1u, w / 3)), preh(std::max(1u, h / 3)) {}
};
}  // namespace

class MlfgInterpolator final : public Interpolator
{
public:
    MlfgInterpolator(Device& d, const Settings& s) : dev(d), set(s), sz(s.displayWidth, s.displayHeight) {}
    bool Create(std::string* error);
    bool Dispatch(const FrameInputs& in, uint32_t count, uint32_t total) override;
    const Texture&  Output(uint32_t i) const override { return *outs[i]; }
    const Settings& settings() const override { return set; }
    std::vector<std::pair<std::string, const Texture*>> Intermediates() const override
    {
        return {{"mv", mv[hc ^ 1].get()}, {"depth", depth[hc ^ 1].get()}, {"forward_mv", fwd.get()}, {"flow0", img0.get()}, {"flow1", img1.get()}, {"output", outs[0].get()}};
    }
    std::vector<TensorInfo> Tensors() const override
    {
        std::vector<TensorInfo> v;
        for (auto& [n, t] : std::initializer_list<std::pair<const char*, const mlfg::Tensor*>>{
                 {"net_in", &netIn}, {"f54", &f54}, {"x54", &x54}, {"f83", &f83}, {"x83", &x83}, {"f112", &f112}, {"x112", &x112}, {"f149", &f149},
                 {"flowout", &flowout}, {"s2in", &s2in}, {"logits", &logits}, {"weights", &weights}})
            v.push_back({n, t->buf.get(), t->w, t->h, t->pitch});
        return v;
    }

private:
    bool Frame(const FrameInputs& in, uint32_t count, uint32_t total);
    bool Splat(const Texture& flow, const Texture& dep, float proj, float sign, const Texture& key, const Texture& word);
    Device&   dev;
    Settings  set;
    Sizes     sz;
    std::unique_ptr<mlfg::Network> net;
    PassPtr   dilate, splatKey, splatFlow, forwardMv, netInput, downsample, refine, flowOut, s2input, softmax, weightsH, composite, finish, clear;
    Tex       mv[2], depth[2], key10, word10, fwd, key0, word0, key1, word1, img0, img1, colorTm1[2], outs[kMaxGenerated], gen;
    mlfg::Tensor netIn, inq, f54, x54, flow54, f83, x83, flow83, f112, x112, flow112, f149, flowout, s2in, logits, weights, rows;  // rows: kMlfgWeightsH's horizontal upsample
    uint32_t  hc = 0;  // mv / depth [hc]: this frame's; [hc ^ 1] the last frame's
    uint64_t  frame = 0;
    float     vp0[16] = {}, vpm1[16] = {};  // frame 0's (the last rendered) and frame -1's view-projection
    float     lastJitter[2] = {};
};

bool MlfgInterpolator::Create(std::string* error)
{
    auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
    if (sz.W < 64 || sz.H < 64 || sz.W > 8192 || sz.H > 4320 || !set.renderWidth || !set.renderHeight) return fail("unsupported size");
    const bool inv = set.depthInverted;
    auto pass = [&](PassPtr& p, const Shader& s) { p = dev.CreatePass(s); return p != nullptr; };
    bool ok = pass(dilate, inv ? shaders::kMlfgDilateInv : shaders::kMlfgDilate) && pass(splatKey, inv ? shaders::kMlfgSplatKeyInv : shaders::kMlfgSplatKey) &&
              pass(splatFlow, inv ? shaders::kMlfgSplatFlowInv : shaders::kMlfgSplatFlow) && pass(forwardMv, shaders::kMlfgForwardMv) &&
              pass(netInput, shaders::kMlfgNetInput) && pass(downsample, shaders::kMlfgDownsample) && pass(refine, shaders::kMlfgRefine) &&
              pass(flowOut, shaders::kMlfgFlowOut) && pass(s2input, shaders::kMlfgStage2Input) && pass(softmax, shaders::kMlfgSoftmax5) && pass(weightsH, shaders::kMlfgWeightsH) && pass(clear, shaders::kMlfgClear);
    switch (set.outputFormat)
    {
    case Format::RGBA8Unorm: case Format::BGRA8Unorm: ok = ok && pass(composite, shaders::kMlfgCompositeRgba8) && pass(finish, shaders::kMlfgFinishRgba8); break;
    case Format::RGB10A2Unorm: ok = ok && pass(composite, shaders::kMlfgCompositeRgb10a2) && pass(finish, shaders::kMlfgFinishRgb10a2); break;
    case Format::RGBA16Float: ok = ok && pass(composite, shaders::kMlfgCompositeRgba16f) && pass(finish, shaders::kMlfgFinishRgba16f); break;
    default: return fail("unsupported output format");
    }
    if (!ok) return fail("FSR FG4 passes");
    auto tex = [&](Tex& t, Format f, uint32_t w, uint32_t h) { t = dev.CreateTexture(f, w, h, f == Format::R32Uint); ok = ok && t; };
    for (int i = 0; i < 2; ++i) tex(mv[i], Format::RG16Float, sz.prew, sz.preh), tex(depth[i], Format::R32Float, sz.prew, sz.preh);
    tex(key10, Format::R32Uint, sz.prew, sz.preh), tex(word10, Format::R32Uint, sz.prew, sz.preh), tex(fwd, Format::RG16Float, sz.prew, sz.preh);
    tex(key0, Format::R32Uint, sz.prew, sz.preh), tex(word0, Format::R32Uint, sz.prew, sz.preh), tex(key1, Format::R32Uint, sz.prew, sz.preh), tex(word1, Format::R32Uint, sz.prew, sz.preh);
    tex(img0, Format::RG16Float, sz.livew, sz.liveh), tex(img1, Format::RG16Float, sz.livew, sz.liveh);
    for (auto& t : colorTm1) ok = ok && (t = dev.CreateTextureLike(set.colorLike, sz.W, sz.H));
    for (auto& t : outs) tex(t, set.outputFormat, sz.W, sz.H);
    auto tensor = [&](mlfg::Tensor& t, uint32_t w, uint32_t h, uint32_t pitch) {
        t.w = w, t.h = h, t.pitch = pitch, t.buf = dev.CreateBuffer(uint64_t(w) * h * pitch * 2, true);  // the network's tensors: CUDA kernels may read them
        ok = ok && t.buf;
    };
    const uint32_t L = sz.l0w, Lh = sz.l0h;
    tensor(netIn, L, Lh, 32), tensor(inq, L / 4, Lh / 4, 32), tensor(f54, L / 4, Lh / 4, 16), tensor(x54, L / 2, Lh / 2, 32), tensor(flow54, L, Lh, 4);
    tensor(f83, L / 2, Lh / 2, 16), tensor(x83, L, Lh, 32), tensor(flow83, L, Lh, 4), tensor(f112, L, Lh, 16), tensor(x112, L, Lh, 32), tensor(flow112, L, Lh, 4);
    tensor(f149, L, Lh, 16), tensor(flowout, L, Lh, 4);
    tensor(s2in, sz.s2w, sz.s2h, 32), tensor(logits, sz.s2w, sz.s2h, 16), tensor(weights, sz.s2w, sz.s2h, 8), tensor(rows, sz.W, sz.s2h, 8);
    if (!ok) return fail("out of memory");
    net = mlfg::Network::Create(dev, L, Lh, sz.s2w, sz.s2h, error);
    if (!net) return false;
    memcpy(vp0, set.initialViewProjection, 64), memcpy(vpm1, set.initialViewProjection, 64);
    Log("FSR FG4 interpolator %ux%u (render %ux%u): stage 1 %ux%u, stage 2 %ux%u, network: %s", sz.W, sz.H, set.renderWidth, set.renderHeight, L, Lh, sz.s2w, sz.s2h,
        net->Describe().c_str());
    return true;
}

bool MlfgInterpolator::Splat(const Texture& flow, const Texture& dep, float proj, float sign, const Texture& key, const Texture& word)
{
    const int32_t c[4] = {int32_t(sz.prew), int32_t(sz.preh), 0, 0};
    const Bind cb[] = {Bind::Data(0, c, sizeof(c)), Bind::Tex(1, key), Bind::Tex(2, word)};
    if (!clear->Dispatch(cb, 3, G(sz.prew), G(sz.preh))) return false;
    mlfg::SplatConstants k = {{int32_t(sz.prew), int32_t(sz.preh), int32_t(flow.width), int32_t(flow.height)}, {proj, sign, 0, 0}, {0, 0, 1, 1}};
    const Bind kb[] = {Bind::Data(0, &k, sizeof(k)), Bind::Tex(1, flow), Bind::Tex(2, dep), Bind::Tex(4, key)};
    if (!splatKey->Dispatch(kb, 4, G(sz.prew), G(sz.preh))) return false;
    const Bind fb[] = {Bind::Data(0, &k, sizeof(k)), Bind::Tex(1, flow), Bind::Tex(2, dep), Bind::Tex(3, key), Bind::Tex(5, word)};
    return splatFlow->Dispatch(fb, 5, G(sz.prew), G(sz.preh));
}

bool MlfgInterpolator::Frame(const FrameInputs& in, uint32_t count, uint32_t total)
{
    const Texture *pc = in.colorTexture ? in.colorTexture : dev.Wrap(in.color, false), *pd = in.depthTexture ? in.depthTexture : dev.Wrap(in.depth, false);
    const Texture *pm = in.motionTexture ? in.motionTexture : dev.Wrap(in.motion, false);
    if (!pc || !pd || !pm || !net->Prepare())
        return false;
    const Texture &color = *pc, &prev = *colorTm1[frame % 2], &backup = *colorTm1[(frame + 1) % 2];
    const bool     reset = in.reset || frame == 0;
    ++frame;
    // Vectors that contain the sample positions' jitter (render pixels): its change taken out, as the NFRU interpolator does.
    const bool  jittered = in.motionJittered && !reset;
    const float cancel[2] = {jittered ? (in.jitter[0] - lastJitter[0]) / float(set.renderWidth) : 0.0f, jittered ? (in.jitter[1] - lastJitter[1]) / float(set.renderHeight) : 0.0f};
    lastJitter[0] = in.jitter[0], lastJitter[1] = in.jitter[1];
    // pipe191 of this frame: the history of the next.
    {
        mlfg::DilateConstants k = {{int32_t(set.renderWidth), int32_t(set.renderHeight), int32_t(sz.prew), int32_t(sz.preh)},
                                   {in.mvScale[0] * float(set.renderWidth) / float(sz.W), in.mvScale[1] * float(set.renderHeight) / float(sz.H), cancel[0], cancel[1]}, {0, 0, 1, 1}, {0, 0, 1, 1}};
        const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Tex(1, *pm), Bind::Tex(2, *pd), Bind::Tex(3, *mv[hc]), Bind::Tex(4, *depth[hc])};
        if (!dilate->Dispatch(b, 5, G(sz.prew), G(sz.preh))) return false;
    }
    if (reset)
    {
        dev.CopyTexture(color, backup);
        memcpy(vp0, in.viewProjection, 64), memcpy(vpm1, in.viewProjection, 64);
        hc ^= 1;
        return true;
    }
    const Texture &mv1 = *mv[hc], &d1 = *depth[hc], &mv0 = *mv[hc ^ 1], &d0 = *depth[hc ^ 1];
    // pipe192 (frame 1 -> frame 0) and pipe193: frame 0's forward motion.
    if (!Splat(mv1, d1, 1.0f, 1.0f, *key10, *word10)) return false;
    {
        mlfg::ForwardMvConstants k = {};
        ClipToClip(vp0, in.viewProjection, k.prevPstoPs);
        ClipToClip(vp0, vpm1, k.prevPsToPrev2Ps);
        k.sizes[0] = int32_t(sz.W), k.sizes[1] = int32_t(sz.H), k.sizes[2] = int32_t(sz.prew), k.sizes[3] = int32_t(sz.preh);
        k.depthSize[0] = int32_t(d0.width), k.depthSize[1] = int32_t(d0.height);
        const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Tex(1, mv0), Bind::Tex(2, d0), Bind::Tex(3, *key10), Bind::Tex(4, *word10), Bind::Tex(5, *fwd)};
        if (!forwardMv->Dispatch(b, 6, G(sz.prew), G(sz.preh))) return false;
    }
    dev.Mark("mlfg prepass");
    // Stage 1, once, at t = 0.5.
    const uint32_t L = sz.l0w, Lh = sz.l0h;
    {
        mlfg::NetInputConstants k = {{int32_t(L), int32_t(Lh), int32_t(L), int32_t(Lh)}, {int32_t(sz.W), int32_t(sz.H), 0, 0}, {0, 0, 1, 1}, {0, 0, 0, 0}};
        const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Tex(1, prev), Bind::Tex(2, color), Bind::Buf(3, *netIn.buf)};
        if (!netInput->Dispatch(b, 4, G(L), G(Lh))) return false;
    }
    if (!net->Encoders(netIn.v())) return false;
    {
        mlfg::DownsampleConstants k = {{int32_t(L), int32_t(L), int32_t(Lh), 0}, {int32_t(inq.w), int32_t(inq.h), int32_t(inq.w), int32_t(inq.h)}, {4.0f, 0, 0, 0}};
        const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Buf(1, *netIn.buf), Bind::Buf(2, *inq.buf)};
        if (!downsample->Dispatch(b, 3, G(inq.w), G(inq.h))) return false;
    }
    auto refineRun = [&](int32_t mode, const mlfg::Tensor& head, const mlfg::Tensor* prevFlow, const mlfg::Tensor& out, const mlfg::Tensor& flow, float scale, float fscale) {
        mlfg::RefineConstants k = {{mode}, {int32_t(L), int32_t(L), int32_t(Lh)}, {int32_t(head.w), int32_t(head.w), int32_t(head.h)},
                                   {int32_t(prevFlow ? prevFlow->w : 1)}, {int32_t(out.w), int32_t(out.h), int32_t(out.w), int32_t(out.h)}, {int32_t(flow.w)}, {scale, fscale, 0.5f, 0}};
        const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Buf(1, *netIn.buf), Bind::Buf(2, *head.buf), Bind::Buf(3, prevFlow ? *prevFlow->buf : *head.buf),
                          Bind::Buf(4, *out.buf), Bind::Buf(5, *flow.buf)};
        return refine->Dispatch(b, 6, G(out.w), G(out.h));
    };
    if (!net->FlowBlock(0, inq.v(), f54.v()) || !refineRun(0, f54, nullptr, x54, flow54, 0.25f, 4.0f)) return false;
    if (!net->FlowBlock(1, x54.v(), f83.v()) || !refineRun(1, f83, &flow54, x83, flow83, 0.5f, 2.0f)) return false;
    if (!net->FlowBlock(2, x83.v(), f112.v()) || !refineRun(2, f112, &flow83, x112, flow112, 0.0f, 1.0f)) return false;
    if (!net->FlowBlock(3, x112.v(), f149.v())) return false;
    dev.Mark("mlfg stage 1");
    const bool needsAlpha = in.uiKind >= 2 && in.uiKind <= 5;
    const Texture* ui     = in.uiSource ? dev.Wrap(in.uiSource, false) : nullptr;
    const Texture* alpha  = needsAlpha && in.uiAlpha ? dev.Wrap(in.uiAlpha, false) : nullptr;
    const Texture* noWarp = in.noWarp ? dev.Wrap(in.noWarp, false) : nullptr;
    const uint32_t uiKind = !ui || (needsAlpha && !alpha) ? 0u : in.uiKind;
    const bool     finishing = ui || noWarp || set.swapRB;
    if (finishing && !gen && !(gen = dev.CreateTexture(set.outputFormat, sz.W, sz.H))) return false;
    for (uint32_t i = 0; i < count; ++i)
    {
        const float t = float(i + 1) / float(total + 1);
        {
            const int32_t k[16] = {int32_t(L), int32_t(L), 0, 0, int32_t(L), int32_t(Lh), int32_t(L), int32_t(Lh), int32_t(sz.livew), int32_t(sz.liveh), 0, 0};
            float kf[16];
            memcpy(kf, k, sizeof(k));
            kf[12] = 2.0f * t, kf[13] = 2.0f * (1.0f - t), kf[14] = kf[15] = 0;
            const Bind b[] = {Bind::Data(0, kf, sizeof(kf)), Bind::Buf(1, *f149.buf), Bind::Buf(2, *flow112.buf), Bind::Buf(3, *flowout.buf), Bind::Tex(4, *img0), Bind::Tex(5, *img1)};
            if (!flowOut->Dispatch(b, 6, G(std::max(L, sz.livew)), G(std::max(Lh, sz.liveh)))) return false;
        }
        if (!Splat(*fwd, d0, t, -1.0f, *key0, *word0) || !Splat(mv1, d1, 1.0f - t, 1.0f, *key1, *word1)) return false;
        {
            mlfg::Stage2InputConstants k = {{int32_t(sz.s2w), int32_t(sz.s2h), int32_t(sz.s2w), int32_t(sz.s2h)}, {int32_t(sz.W), int32_t(sz.H), int32_t(sz.livew), int32_t(sz.liveh)},
                                            {int32_t(sz.prew), int32_t(sz.preh), 0, Bits(0.5f * (float(sz.prew) / float(sz.s2w)))}, {0, 0, 1, 1},
                                            {0, 0, 1.0f / float(sz.s2w), 1.0f / float(sz.s2h)}};
            const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Tex(1, prev), Bind::Tex(2, color), Bind::Tex(3, *img0), Bind::Tex(4, *img1), Bind::Tex(5, *key0),
                              Bind::Tex(6, *word0), Bind::Tex(7, *key1), Bind::Tex(8, *word1), Bind::Buf(9, *s2in.buf)};
            if (!s2input->Dispatch(b, 10, G(sz.s2w), G(sz.s2h))) return false;
        }
        if (!net->Stage2(s2in.v(), logits.v())) return false;
        {
            const int32_t k[4] = {int32_t(sz.s2w), int32_t(sz.s2h), int32_t(sz.s2w), int32_t(sz.s2w)};
            const Bind    b[] = {Bind::Data(0, k, sizeof(k)), Bind::Buf(1, *logits.buf), Bind::Buf(2, *weights.buf)};
            if (!softmax->Dispatch(b, 3, G(sz.s2w), G(sz.s2h))) return false;
        }
        const Texture& out = finishing ? *gen : *outs[i];
        {
            // The weights' horizontal upsample once a column per weight row (same bits; the composite does the vertical mix).
            const int32_t hk[4] = {int32_t(sz.W), int32_t(sz.s2w), int32_t(sz.s2w), int32_t(sz.s2h)};
            const Bind    hb[] = {Bind::Data(0, hk, sizeof(hk)), Bind::Buf(1, *weights.buf), Bind::Buf(2, *rows.buf)};
            if (!weightsH->Dispatch(hb, 3, (sz.W + 63) / 64, sz.s2h)) return false;
            mlfg::CompositeConstants k = {{int32_t(sz.W), int32_t(sz.H), int32_t(sz.livew), int32_t(sz.liveh)}, {int32_t(sz.prew), int32_t(sz.preh), int32_t(sz.s2w), 1},
                                          {int32_t(sz.s2w), int32_t(sz.s2h), 0, 0}, {0, 0, 0, Bits(0.5f * (float(sz.prew) / float(sz.W)))}, {0, 0, 1, 1}, {0, 0, t, 1.0f / float(sz.W)}, {0, 0, 0, 1.0f / float(sz.H)}};
            const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Buf(1, *rows.buf), Bind::Tex(2, prev), Bind::Tex(3, color), Bind::Tex(4, *img0), Bind::Tex(5, *img1),
                              Bind::Tex(6, *key0), Bind::Tex(7, *word0), Bind::Tex(8, *key1), Bind::Tex(9, *word1), Bind::Tex(10, color), Bind::Tex(11, out)};
            if (!composite->Dispatch(b, 12, G(sz.W), G(sz.H))) return false;
        }
        if (finishing)
        {
            const int32_t k[4] = {int32_t(sz.W), int32_t(sz.H), int32_t(uiKind), (ui ? 1 : 0) | (noWarp ? 2 : 0) | (set.swapRB ? 4 : 0)};
            const Bind    b[] = {Bind::Data(0, k, sizeof(k)), Bind::Tex(1, *gen), Bind::Tex(2, color), Bind::Tex(3, ui ? *ui : color), Bind::Tex(4, alpha ? *alpha : color),
                                 Bind::Tex(5, noWarp ? *noWarp : color), Bind::Tex(6, *outs[i])};
            if (!finish->Dispatch(b, 7, G(sz.W), G(sz.H))) return false;
        }
    }
    dev.Mark("mlfg generated frames");
    net->FrameDone();
    dev.CopyTexture(color, backup);
    memcpy(vpm1, vp0, 64), memcpy(vp0, in.viewProjection, 64);
    hc ^= 1;
    return true;
}

bool MlfgInterpolator::Dispatch(const FrameInputs& in, uint32_t count, uint32_t total)
{
    return count && Frame(in, std::min(count, kMaxGenerated), total);
}

std::unique_ptr<Interpolator> CreateMlfgInterpolator(Device& device, const Settings& settings, std::string* error)
{
    auto it = std::make_unique<MlfgInterpolator>(device, settings);
    if (!it->Create(error)) return nullptr;
    return it;
}
}  // namespace nfrux
