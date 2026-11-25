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

bool FrameGenerator::Generate(const EvalInputs& in)
{
    // The frame interpolated: HUD-less when it is the back buffer's size and format.
    const bool hudless = SameSize(in.hudless, in.back) && in.hudless.format == in.back.format && Whole(in.hudlessRect, in.hudless);
    const ExternalImage& color = hudless ? in.hudless : in.back;
    if (!in.depth || !in.motion || ClassOf(color, m_vulkan) == Format::Unknown || !Whole(in.backRect, in.back))
        return false;
    ExternalImage depth = in.depth, motion = in.motion;
    const uint32_t rw = in.depthRect.w ? in.depthRect.w : depth.width, rh = in.depthRect.h ? in.depthRect.h : depth.height;
    const bool cutDepth = !Whole(in.depthRect, depth), cutMotion = !Whole(in.motionRect, motion);
    const uint32_t mw = cutMotion ? in.motionRect.w : motion.width, mh = cutMotion ? in.motionRect.h : motion.height;
    Entry* e = Get(color, in, in.back.width, in.back.height, rw, rh);
    m_current = e;
    if (e->failed)
        return false;

    FrameInputs f;
    f.color = color, f.depth = depth, f.motion = motion;
    if (cutDepth && !(f.depthTexture = Crop(depth, in.depthRect, true))) return false;
    if (cutMotion && !(f.motionTexture = Crop(motion, in.motionRect, false))) return false;
    const bool uiAlpha = SameSize(in.uiAlpha, in.back) && Whole(in.uiAlphaRect, in.uiAlpha);
    const bool uiLayer = !uiAlpha && SameSize(in.ui, in.back) && Whole(in.uiRect, in.ui);
    if (hudless || uiAlpha || uiLayer)
    {
        f.uiSource = in.back;
        f.uiKind   = !hudless ? (uiAlpha ? 4u : 5u) : uiAlpha ? 2u : uiLayer ? 3u : 0u;
        if (uiAlpha || uiLayer) f.uiAlpha = uiAlpha ? in.uiAlpha : in.ui;
    }
    if (SameSize(in.noWarp, in.back) && Whole(in.noWarpRect, in.noWarp)) f.noWarp = in.noWarp;
    f.reset = in.reset || m_restart;
    f.jitter[0] = in.jitter[0], f.jitter[1] = in.jitter[1];
    f.mvScale[0] = in.mvScale[0] * float(in.back.width) / float(std::max(1u, mw));
    f.mvScale[1] = in.mvScale[1] * float(in.back.height) / float(std::max(1u, mh));
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
        if (in.notRendering) m_restart = true;
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
        m_dev.CopyTexture(m_current->ip->Output(k), out), ++m_generated;
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
