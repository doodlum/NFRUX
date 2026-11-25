// SPDX-License-Identifier: MIT
#include "frame_generator.h"

#include "../core/log.h"
#include "nfrux_shaders.h"

#include <windows.h>
#include <dxgiformat.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nfrux
{
Format ClassOf(const ExternalImage& e, bool vulkan)
{
    if (vulkan)
        switch (VkFormat(e.format))
        {
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: return Format::RGBA8Unorm;
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: return Format::BGRA8Unorm;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return Format::RGB10A2Unorm;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return Format::RGBA16Float;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return Format::RG11B10Float;
        default: return Format::Unknown;
        }
    switch (DXGI_FORMAT(e.format))
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return Format::RGBA8Unorm;
    case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return Format::BGRA8Unorm;
    case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_TYPELESS: return Format::RGB10A2Unorm;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_TYPELESS: return Format::RGBA16Float;
    case DXGI_FORMAT_R11G11B10_FLOAT: return Format::RG11B10Float;
    default: return Format::Unknown;
    }
}
bool IsDepthFormat(const ExternalImage& e, bool vulkan)
{
    if (vulkan)
        return e.format >= VK_FORMAT_D16_UNORM && e.format <= VK_FORMAT_D32_SFLOAT_S8_UINT;
    switch (DXGI_FORMAT(e.format))
    {
    case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32G8X24_TYPELESS: return true;
    default: return false;
    }
}

namespace
{
// Column-major 4x4, as the interpolator's.
void Mul(const float a[16], const float b[16], float out[16])
{
    float r[16];
    for (int c = 0; c < 4; ++c)
        for (int w = 0; w < 4; ++w)
        {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + w] * b[c * 4 + k];
            r[c * 4 + w] = s;
        }
    memcpy(out, r, sizeof(r));
}
bool Inverse(const float m[16], float out[16])
{
    double a[4][8] = {};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) a[r][c] = m[c * 4 + r], a[r][4 + r] = 1;
    for (int i = 0; i < 4; ++i)
    {
        int p = i;
        for (int r = i + 1; r < 4; ++r) if (fabs(a[r][i]) > fabs(a[p][i])) p = r;
        if (fabs(a[p][i]) < 1e-20) return false;
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
// Streamline's matrices are row-major for row vectors: the same numbers as column-major for column vectors.
void CameraViewProjection(const EvalInputs& in, float out[16])
{
    float v[16] = {};
    for (int k = 0; k < 3; ++k) v[k * 4 + 0] = in.right[k], v[k * 4 + 1] = in.up[k], v[k * 4 + 2] = in.fwd[k];
    for (int k = 0; k < 3; ++k) v[12] -= in.pos[k] * in.right[k], v[13] -= in.pos[k] * in.up[k], v[14] -= in.pos[k] * in.fwd[k];
    v[15] = 1.0f;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
        {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += v[r * 4 + k] * in.viewToClip[k * 4 + c];
            out[r * 4 + c] = s;
        }
}
bool Whole(const Rect& r, const ExternalImage& e) { return !r.w || !r.h || (!r.x && !r.y && r.w == e.width && r.h == e.height); }
bool SameSize(const ExternalImage& a, const ExternalImage& b) { return a && b && a.width == b.width && a.height == b.height; }
}  // namespace

FrameGenerator::~FrameGenerator() { m_dev.WaitIdle(); }

FrameGenerator::Entry* FrameGenerator::Get(const ExternalImage& color, const EvalInputs& in, uint32_t w, uint32_t h, uint32_t rw, uint32_t rh)
{
    for (size_t i = 0; i < m_entries.size(); ++i)
    {
        Entry& e = *m_entries[i];
        if (e.w == w && e.h == h && e.rw == rw && e.rh == rh && e.format == color.format && e.inverted == in.depthInverted)
        {
            if (i) std::rotate(m_entries.begin(), m_entries.begin() + i, m_entries.begin() + i + 1), m_restart = true;
            return m_entries[0].get();
        }
    }
    auto e = std::make_unique<Entry>(Entry{nullptr, w, h, rw, rh, color.format, in.depthInverted});
    Settings s;
    s.displayWidth = w, s.displayHeight = h, s.renderWidth = rw, s.renderHeight = rh, s.depthInverted = in.depthInverted;
    s.colorLike = color, s.outputFormat = ClassOf(color, m_vulkan);
    if (m_vulkan && s.outputFormat == Format::BGRA8Unorm)
        s.outputFormat = Format::RGBA8Unorm, s.swapRB = true;  // the bytes of BGRA8, copied raw
    CameraViewProjection(in, s.initialViewProjection);
    memcpy(m_vp, s.initialViewProjection, sizeof(m_vp));
    std::string error;
    e->ip     = Interpolator::Create(m_dev, s, &error);
    e->failed = !e->ip;
    if (e->failed)
        Log("frame generation %ux%u (render %ux%u, format %u) not possible: %s; showing rendered frames", w, h, rw, rh, color.format, error.c_str());
    m_entries.insert(m_entries.begin(), std::move(e));
    while (m_entries.size() > 2)
        m_retired.push_back({std::move(m_entries.back()), m_evaluations}), m_entries.pop_back();
    m_haveVp = false, m_restart = true;
    return m_entries[0].get();
}

const Texture* FrameGenerator::Crop(const ExternalImage& image, const Rect& r, bool depth)
{
    Cut& c = m_cut[depth ? 0 : 1];
    if (r.x + r.w > image.width || r.y + r.h > image.height)
        return nullptr;
    if (!c.pass) c.pass = m_dev.CreatePass(depth ? shaders::kCropR32f : shaders::kCropRg16f);
    if (!c.tex || c.tex->width != r.w || c.tex->height != r.h)
    {
        if (c.tex) m_oldCuts.push_back({std::move(c.tex), m_evaluations});
        c.tex = m_dev.CreateTexture(depth ? Format::R32Float : Format::RG16Float, r.w, r.h);
        Log("%s cut to the rendered region %ux%u of %ux%u", depth ? "depth" : "motion vectors", r.w, r.h, image.width, image.height);
    }
    const Texture* src = m_dev.Wrap(image, false);
    if (!c.pass || !c.tex || !src)
        return nullptr;
    const int32_t k[4] = {int32_t(r.x), int32_t(r.y), int32_t(r.w), int32_t(r.h)};
    const Bind b[] = {Bind::Data(0, k, sizeof(k)), Bind::Tex(1, *src), Bind::Tex(2, *c.tex)};
    return c.pass->Dispatch(b, 3, (r.w + 7) / 8, (r.h + 7) / 8) ? c.tex.get() : nullptr;
}

// Which parts of which images the generated region is.
struct Layout
{
    Rect back, hudless, ui, noWarp;
    bool hasHudless = false, uiAlpha = false, uiLayer = false, hasNoWarp = false, whole = false;  // whole: every one is a whole image
};
namespace
{
// An image's region: its own subrect, else the back buffer's when it has the back buffer's size, else all of it.
Rect RegionOf(const ExternalImage& im, const Rect& own, const ExternalImage& back, const Rect& backRect)
{
    if (own.w && own.h) return own;
    if (SameSize(im, back)) return backRect;
    return {0, 0, im.width, im.height};
}
bool Fits(const Rect& r, const ExternalImage& e, const Rect& like) { return e && r.w == like.w && r.h == like.h && r.x + r.w <= e.width && r.y + r.h <= e.height; }
// HUD-less in another format is compared and composited as values: only between formats of the same transfer.
bool SameTransfer(Format a, Format b, bool hdr)
{
    auto sdr = [](Format f) { return f == Format::RGBA8Unorm || f == Format::BGRA8Unorm || f == Format::RGB10A2Unorm; };
    return a == b || (!hdr && sdr(a) && sdr(b));
}
}  // namespace

Texture* FrameGenerator::Keep(std::unique_ptr<Texture>& t, Format f, uint32_t w, uint32_t h)
{
    if (!t || t->width != w || t->height != h || t->format != f)
    {
        if (t) m_oldCuts.push_back({std::move(t), m_evaluations});
        t = m_dev.CreateTexture(f, w, h);
    }
    return t.get();
}

bool FrameGenerator::InMenu(const EvalInputs& in)
{
    const bool           alpha = bool(in.uiAlpha);
    const ExternalImage& ui    = alpha ? in.uiAlpha : in.ui;
    if (!in.menuDetection || !ui)
        return m_inMenu = false;
    const Rect own = alpha ? in.uiAlphaRect : in.uiRect, r = RegionOf(ui, own, in.back, Whole(in.backRect, in.back) ? Rect{0, 0, in.back.width, in.back.height} : in.backRect);
    // The latest counts that came back. Over 60% of the screen it is a menu, under 40% not any more.
    while (!m_menuReads.empty() && !m_menuReads[0]->empty())
    {
        uint32_t n[2];
        memcpy(n, m_menuReads[0]->data(), sizeof(n));
        m_menuReads.erase(m_menuReads.begin());
        const float share = n[1] + 1 ? float(n[0] + 1) / float(n[1] + 1) : 0.0f;
        const bool  menu  = share > (m_inMenu ? 0.4f : 0.6f);
        if (menu != m_inMenu) Log(menu ? "full-screen menu (%.0f%%): no interpolation" : "menu gone (%.0f%%): interpolating again", share * 100.0f);
        m_inMenu = menu;
    }
    if (!m_menuPass) m_menuPass = m_dev.CreatePass(shaders::kMenu);
    Texture*       counts = Keep(m_menuCounts, Format::R32Uint, 2, 1);
    const Texture* src    = m_dev.Wrap(ui, false);
    if (!m_menuPass || !counts || !src || r.x + r.w > ui.width || r.y + r.h > ui.height || m_menuReads.size() > 16)
        return m_inMenu;
    m_dev.Clear(*counts, 1.0f);  // every bit set: the shader's adds start from there
    const int32_t k[8] = {int32_t(r.x), int32_t(r.y), int32_t(r.w), int32_t(r.h), 8, alpha};
    const Bind    b[]  = {Bind::Data(0, k, sizeof(k)), Bind::Tex(1, *src), Bind::Tex(2, *counts)};
    if (m_menuPass->Dispatch(b, 3, std::max(1u, (r.w / 8 + 7) / 8), std::max(1u, (r.h / 8 + 7) / 8)))
        m_menuReads.push_back(std::make_shared<std::vector<uint8_t>>()), m_dev.ReadBack(*counts, m_menuReads.back());
    return m_inMenu;
}

// The interpolated frames (fp16, the region's size) into images of the output's format.
bool FrameGenerator::Staged(const EvalInputs& in, const Layout& l)
{
    Format f = ClassOf(in.back, m_vulkan);
    const bool swap = m_vulkan && f == Format::BGRA8Unorm;
    if (swap) f = Format::RGBA8Unorm;
    if (!m_compose || m_composeFormat != f)
    {
        if (m_compose) m_oldPasses.push_back(std::move(m_compose));  // maybe in flight; a format change is rare: kept
        m_compose = m_dev.CreatePass(f == Format::RGB10A2Unorm ? shaders::kComposeRgb10a2 : f == Format::RGBA16Float ? shaders::kComposeRgba16f
                                     : f == Format::RG11B10Float ? shaders::kComposeR11 : shaders::kComposeRgba8);
        m_composeFormat = f;
    }
    const Texture *frame = m_dev.Wrap(in.back, false), *base = l.hasHudless ? m_dev.Wrap(in.hudless, false) : frame;
    const Texture *ui = l.uiAlpha ? m_dev.Wrap(in.uiAlpha, false) : l.uiLayer ? m_dev.Wrap(in.ui, false) : frame, *noWarp = l.hasNoWarp ? m_dev.Wrap(in.noWarp, false) : frame;
    const Texture* field = in.distortion ? m_dev.Wrap(in.distortion, false) : frame;
    if (!m_compose || !frame || !base || !ui || !noWarp || !field)
        return false;
    const Rect fr = Whole(in.distortionRect, in.distortion) ? Rect{0, 0, in.distortion.width, in.distortion.height} : in.distortionRect;
    const Rect baseRect = l.hasHudless ? l.hudless : l.back;
    struct { int32_t outSize[2], origin[2], size[2], base[2], ui[2], noWarp[2], fieldBase[2], fieldSize[2]; float scale[2], bias[2]; int32_t distort, uiKind, hasNoWarp, swapRB, scrgb, pad[3]; } k = {};
    k.outSize[0] = int32_t(in.back.width), k.outSize[1] = int32_t(in.back.height);
    k.origin[0] = int32_t(l.back.x), k.origin[1] = int32_t(l.back.y), k.size[0] = int32_t(l.back.w), k.size[1] = int32_t(l.back.h);
    k.base[0] = int32_t(baseRect.x), k.base[1] = int32_t(baseRect.y), k.ui[0] = int32_t(l.ui.x), k.ui[1] = int32_t(l.ui.y), k.noWarp[0] = int32_t(l.noWarp.x), k.noWarp[1] = int32_t(l.noWarp.y);
    k.fieldBase[0] = int32_t(fr.x), k.fieldBase[1] = int32_t(fr.y), k.fieldSize[0] = int32_t(fr.w), k.fieldSize[1] = int32_t(fr.h);
    for (int i = 0; i < 2; ++i)
    {
        // Field values are in the motion vectors' units: x their scale motion-vector pixels, / the motion vectors' size UV.
        const uint32_t size = i ? (in.motionRect.h ? in.motionRect.h : in.motion.height) : (in.motionRect.w ? in.motionRect.w : in.motion.width);
        const float    m    = in.mvScale[i] / float(std::max(1u, size));
        k.scale[i] = in.distortionScale * m, k.bias[i] = in.distortionBias * m;
    }
    k.distort = in.distortion ? 1 : 0, k.hasNoWarp = l.hasNoWarp, k.swapRB = swap, k.scrgb = in.hdr && f == Format::RGBA16Float;
    k.uiKind = !l.hasHudless ? (l.uiAlpha ? 4 : l.uiLayer ? 5 : -1) : l.uiAlpha ? 2 : l.uiLayer ? 3 : 0;
    for (uint32_t i = 0; i < m_ready; ++i)
    {
        Texture* out = Keep(m_composed[i], f, in.back.width, in.back.height);
        if (!out) return false;
        const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Tex(1, m_current->ip->Output(i)), Bind::Tex(2, *field), Bind::Tex(3, *base), Bind::Tex(4, *frame),
                          Bind::Tex(5, *ui), Bind::Tex(6, *noWarp), Bind::Tex(7, *out)};
        if (!m_compose->Dispatch(b, 8, (in.back.width + 7) / 8, (in.back.height + 7) / 8)) return false;
    }
    return true;
}

bool FrameGenerator::Generate(const EvalInputs& in)
{
    Layout l;
    l.back = Whole(in.backRect, in.back) ? Rect{0, 0, in.back.width, in.back.height} : in.backRect;
    l.hudless = RegionOf(in.hudless, in.hudlessRect, in.back, l.back);
    const Format bf = ClassOf(in.back, m_vulkan), hf = in.hudless ? ClassOf(in.hudless, m_vulkan) : Format::Unknown;
    if (!in.depth || !in.motion || bf == Format::Unknown || !Fits(l.back, in.back, l.back))
        return false;
    // The frame interpolated: HUD-less when it fits the back buffer.
    l.hasHudless = Fits(l.hudless, in.hudless, l.back);
    if (l.hasHudless && !SameTransfer(hf, bf, in.hdr))
    {
        if (!m_hudlessLogged) Log("HUD-less format %u does not fit the back buffer's %u: not used", in.hudless.format, in.back.format), m_hudlessLogged = true;
        l.hasHudless = false;
    }
    const Rect alphaRect = RegionOf(in.uiAlpha, in.uiAlphaRect, in.back, l.back), layerRect = RegionOf(in.ui, in.uiRect, in.back, l.back);
    l.uiAlpha = Fits(alphaRect, in.uiAlpha, l.back), l.uiLayer = !l.uiAlpha && Fits(layerRect, in.ui, l.back);
    l.ui = l.uiAlpha ? alphaRect : layerRect;
    l.noWarp = RegionOf(in.noWarp, in.noWarpRect, in.back, l.back), l.hasNoWarp = Fits(l.noWarp, in.noWarp, l.back);
    l.whole = Whole(l.back, in.back) && (!l.hasHudless || (Whole(l.hudless, in.hudless) && in.hudless.format == in.back.format)) &&
              (!l.uiAlpha || Whole(l.ui, in.uiAlpha)) && (!l.uiLayer || Whole(l.ui, in.ui)) && (!l.hasNoWarp || Whole(l.noWarp, in.noWarp));
    const Rect fieldRect = Whole(in.distortionRect, in.distortion) ? Rect{0, 0, in.distortion.width, in.distortion.height} : in.distortionRect;
    if (in.distortion && !Fits(fieldRect, in.distortion, fieldRect))
        return false;
    const bool hudless = l.hasHudless, staged = !l.whole || in.distortion;
    ExternalImage color = hudless ? in.hudless : in.back;
    if (staged) color = ExternalImage{}, color.format = m_vulkan ? uint32_t(VK_FORMAT_R16G16B16A16_SFLOAT) : uint32_t(DXGI_FORMAT_R16G16B16A16_FLOAT);
    ExternalImage depth = in.depth, motion = in.motion;
    const uint32_t rw = in.depthRect.w ? in.depthRect.w : depth.width, rh = in.depthRect.h ? in.depthRect.h : depth.height;
    const bool cutDepth = !Whole(in.depthRect, depth), cutMotion = !Whole(in.motionRect, motion);
    const uint32_t mw = cutMotion ? in.motionRect.w : motion.width, mh = cutMotion ? in.motionRect.h : motion.height;
    Entry* e = Get(color, in, l.back.w, l.back.h, rw, rh);
    m_current = e;
    if (e->failed)
        return false;

    FrameInputs f;
    f.color = color, f.depth = depth, f.motion = motion;
    if (cutDepth && !(f.depthTexture = Crop(depth, in.depthRect, true))) return false;
    if (cutMotion && !(f.motionTexture = Crop(motion, in.motionRect, false))) return false;
    if (staged)
    {
        if (!m_prepare) m_prepare = m_dev.CreatePass(shaders::kPrepare);
        Texture*       t   = Keep(m_prepared, Format::RGBA16Float, l.back.w, l.back.h);
        const Texture *src = m_dev.Wrap(hudless ? in.hudless : in.back, false), *field = in.distortion ? m_dev.Wrap(in.distortion, false) : src;
        if (!m_prepare || !t || !src || !field) return false;
        const Rect     r = hudless ? l.hudless : l.back;
        struct { int32_t base[2], size[2], fieldBase[2], fieldSize[2]; float scale[2], bias[2]; int32_t distort[4]; } k = {
            {int32_t(r.x), int32_t(r.y)}, {int32_t(r.w), int32_t(r.h)}, {int32_t(fieldRect.x), int32_t(fieldRect.y)}, {int32_t(fieldRect.w), int32_t(fieldRect.h)}};
        k.scale[0] = in.distortionScale * in.mvScale[0] / float(std::max(1u, mw)), k.scale[1] = in.distortionScale * in.mvScale[1] / float(std::max(1u, mh));
        k.bias[0] = in.distortionBias * in.mvScale[0] / float(std::max(1u, mw)), k.bias[1] = in.distortionBias * in.mvScale[1] / float(std::max(1u, mh));
        k.distort[0] = in.distortion ? 1 : 0;
        const Bind b[] = {Bind::Data(0, &k, sizeof(k)), Bind::Tex(1, *src), Bind::Tex(2, *field), Bind::Tex(3, *t)};
        if (!m_prepare->Dispatch(b, 4, (r.w + 7) / 8, (r.h + 7) / 8)) return false;
        f.colorTexture = t;
    }
    const bool uiAlpha = l.uiAlpha, uiLayer = l.uiLayer;
    f.scrgb = in.hdr && bf == Format::RGBA16Float;
    if (!staged && (hudless || uiAlpha || uiLayer))
    {
        f.uiSource = in.back;
        f.uiKind   = !hudless ? (uiAlpha ? 4u : 5u) : uiAlpha ? 2u : uiLayer ? 3u : 0u;
        if (uiAlpha || uiLayer) f.uiAlpha = uiAlpha ? in.uiAlpha : in.ui;
    }
    if (!staged && l.hasNoWarp) f.noWarp = in.noWarp;
    f.reset = in.reset || m_restart;
    f.jitter[0] = in.jitter[0], f.jitter[1] = in.jitter[1];
    f.mvScale[0] = in.mvScale[0] * float(l.back.w) / float(std::max(1u, mw));
    f.mvScale[1] = in.mvScale[1] * float(l.back.h) / float(std::max(1u, mh));
    f.cameraNear = in.cameraNear, f.cameraFar = in.cameraFar, f.fovY = in.fovY, f.orthographic = in.ortho;
    float inv[16];
    if (!in.hasClipToPrevClip || !m_haveVp)
    {
        if (!in.hasClipToPrevClip) CameraViewProjection(in, m_vp);
        m_haveVp = true;
    }
    else if (Inverse(in.clipToPrevClip, inv))
        Mul(inv, m_vp, m_vp);
    // The chain only ever enters as one frame's motion: start it again before its numbers grow.
    float big = 0;
    for (float x : m_vp) big = std::max(big, fabsf(x));
    if (!(big < 1e6f))
        CameraViewProjection(in, m_vp), f.reset = true;
    memcpy(f.viewProjection, m_vp, sizeof(m_vp));
    const uint32_t count = std::min(in.count, kMaxGenerated);
    if (!e->ip->Dispatch(f, count, in.count))
    {
        e->failed = true, m_restart = true;  // its histories are half made: not used again
        Log("recording a frame failed: showing rendered frames at this size");
        return false;
    }
    const bool wasReset = f.reset;
    m_restart = false;
    m_ready   = wasReset ? 0 : count;  // on a reset every output is the frame itself
    m_staged  = staged;
    if (staged && !Staged(in, l)) m_ready = 0;
    return true;
}

bool FrameGenerator::Evaluate(void* commands, const EvalInputs& in)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!in.back || !in.output)
        return false;
    ++m_evaluations;
    static const int dumpAt = Env("NFRUX_DUMP") ? atoi(Env("NFRUX_DUMP")) : -1;  // diagnosis: three rendered frames and what was made
    static int rendered = 0;
    static std::vector<std::pair<std::string, std::shared_ptr<std::vector<uint8_t>>>> dumps;
    if (in.index <= 1)
    {
        char v[16];
        rendered = dumpAt >= 0 && GetEnvironmentVariableA("NFRUX_FRAME", v, 16) ? std::max(rendered + 1, atoi(v)) : rendered + 1;  // the sample's frame number
    }
    if (dumpAt >= 0 && rendered >= dumpAt + 60 && !dumps.empty())
    {
        m_dev.WaitIdle();
        for (auto& [n, v] : dumps)
            if (FILE* f = fopen((std::string(Env("NFRUX_DUMP_DIR") ? Env("NFRUX_DUMP_DIR") : ".") + "\\" + n).c_str(), "wb")) fwrite(v->data(), 1, v->size(), f), fclose(f);
        dumps.clear();
    }    m_retired.erase(std::remove_if(m_retired.begin(), m_retired.end(), [&](auto& r) { return m_evaluations - r.second > 256; }), m_retired.end());
    m_oldCuts.erase(std::remove_if(m_oldCuts.begin(), m_oldCuts.end(), [&](auto& r) { return m_evaluations - r.second > 256; }), m_oldCuts.end());
    if (!m_dev.BeginFrame(commands))
        return false;
    if (in.index <= 1)
    {
        m_ready = 0;
        if (in.notRendering != m_menu)
            Log("%s", in.notRendering ? "not rendering game frames: no interpolation" : "game frames again"), m_menu = in.notRendering;
        const bool menu = InMenu(in);
        if (in.notRendering || menu) m_restart = true;
        else if (!Generate(in) && m_generated == 0 && m_evaluations == 1) Log("first frame not generated");
        static uint32_t logged = 0;
        if (in.count != logged) Log("%u generated frame(s) per rendered frame", in.count), logged = in.count;
    }
    Texture *pBack = m_dev.Wrap(in.back, false), *pOut = m_dev.Wrap(in.output, true);
    if (!pBack || !pOut)
    {
        m_dev.EndFrame();
        return false;
    }
    Texture &back = *pBack, &out = *pOut;
    const uint32_t k = std::max(1u, in.index) - 1;
    if (k < m_ready && m_current && m_current->ip)
        m_dev.CopyTexture(m_staged ? *m_composed[k] : m_current->ip->Output(k), out), ++m_generated;
    else
        m_dev.CopyTexture(back, out);
    if (in.outputReal && in.outputReal.handle != in.back.handle)
        if (Texture* real = m_dev.Wrap(in.outputReal, true))
            m_dev.CopyTexture(back, *real);
    if (dumpAt >= 0 && rendered >= dumpAt && rendered < dumpAt + 8)
    {
        auto add = [&](const std::string& what, const Texture& t) {
            dumps.push_back({what + "_" + std::to_string(in.back.width) + "x" + std::to_string(in.back.height) + "_f" + std::to_string(rendered - dumpAt) + "_k" + std::to_string(in.index) + ".bin",
                             std::make_shared<std::vector<uint8_t>>()});
            m_dev.ReadBack(t, dumps.back().second);
        };
        if (in.index <= 1)
        {
            add("back", back);
            if (in.depth && in.motion)  // in known formats, through the cut-out pass
            {
                const Rect dr = in.depthRect.w ? in.depthRect : Rect{0, 0, in.depth.width, in.depth.height}, mr = in.motionRect.w ? in.motionRect : Rect{0, 0, in.motion.width, in.motion.height};
                if (const Texture* t = Crop(in.depth, dr, true)) add("depth", *t);
                if (const Texture* t = Crop(in.motion, mr, false)) add("motion", *t);
                if (FILE* pf = fopen((std::string(Env("NFRUX_DUMP_DIR") ? Env("NFRUX_DUMP_DIR") : ".") + "\\params_f" + std::to_string(rendered - dumpAt) + ".txt").c_str(), "w"))
                {
                    fprintf(pf, "%u %u %u %u %u %u %u\n", in.back.width, in.back.height, dr.w, dr.h, mr.w, mr.h, in.back.format);
                    fprintf(pf, "%.9g %.9g %.9g %.9g %.9g %.9g %.9g %d %d %d %u\n", in.mvScale[0], in.mvScale[1], in.jitter[0], in.jitter[1], in.cameraNear, in.cameraFar, in.fovY, in.depthInverted, in.reset, in.ortho, in.count);
                    for (float x : in.clipToPrevClip) fprintf(pf, "%.9g ", x);
                    fclose(pf);
                }
            }
            if (in.hudless) add("hudless", *m_dev.Wrap(in.hudless, false));
        }
        add("gen", out);
    }
    m_dev.EndFrame();
    if (m_evaluations % 3000 == 0)
        Log("%llu evaluations, %llu generated", (unsigned long long)m_evaluations, (unsigned long long)m_generated);
    return true;
}
}  // namespace nfrux
