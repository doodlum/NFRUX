// One NGX frame generation feature: reads an Evaluate's parameters and records its frame (docs/DESIGN.md section 4).
// SPDX-License-Identifier: MIT
#pragma once
#include "../core/interpolator.h"

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nfrux
{
struct Rect { uint32_t x = 0, y = 0, w = 0, h = 0; };
struct EvalInputs
{
    ExternalImage back, output, outputReal, hudless, ui, uiAlpha, noWarp, depth, motion, distortion;
    Rect  backRect, hudlessRect, uiRect, uiAlphaRect, noWarpRect, depthRect, motionRect, distortionRect;
    float distortionScale = 1, distortionBias = 0;
    float mvScale[2] = {1, 1}, jitter[2] = {}, cameraNear = 0.1f, cameraFar = 1000.0f, fovY = 1.0f;
    float viewToClip[16] = {}, clipToPrevClip[16] = {}, pos[3] = {}, up[3] = {}, right[3] = {}, fwd[3] = {};
    bool  hasClipToPrevClip = false, depthInverted = false, hdr = false, reset = false, notRendering = false, ortho = false, menuDetection = false, motionJittered = false;
    uint32_t count = 1, index = 1;  // DLSSG.MultiFrameCount / MultiFrameIndex
};

class FrameGenerator
{
public:
    // formatOf: an external image's format class for the composite (Unknown: not supported).
    FrameGenerator(std::unique_ptr<Device> device, bool vulkan) : m_device(std::move(device)), m_dev(*m_device), m_vulkan(vulkan) {}
    ~FrameGenerator();
    // Records one Evaluate. False: nothing usable was passed (the caller reports an invalid parameter).
    bool Evaluate(void* commands, const EvalInputs& in);
    bool vulkan() const { return m_vulkan; }
    void Restart() { m_restart = true; }
    bool Idle() { return m_dev.WaitIdle(); }

private:
    std::unique_ptr<Device> m_device;  // its own: recording state is never shared between features. First: destroyed last
    Device&    m_dev;
    struct Entry
    {
        std::unique_ptr<Interpolator> ip;
        uint32_t w, h, rw, rh, format;
        bool     inverted;
        bool     failed = false;
        std::future<std::unique_ptr<Interpolator>> making;  // on a worker thread: the game is not held up
    };
    Entry* Get(const ExternalImage& color, const EvalInputs& in, uint32_t w, uint32_t h, uint32_t rw, uint32_t rh);
    bool   Generate(const EvalInputs& in);
    // UI rules, colour regions and the distortion field around the interpolator (section 4, D8 of docs/DESIGN.md).
    bool   Staged(const EvalInputs& in, const struct Layout& l);
    bool   InMenu(const EvalInputs& in);
    // D7: the network gives the CPU reference's bytes on this device before any frame of its is shown.
    bool   Checked();
    std::unique_ptr<Network> m_checkNet;
    std::unique_ptr<Pass>    m_checkPass;
    std::shared_ptr<std::vector<uint8_t>> m_checkRead[2];
    std::vector<uint8_t>     m_checkWant[2];  // four bytes a pixel as read; m_checkChannels of them count
    uint32_t m_checkChannels[2] = {}, m_checks = 0;
    uint32_t m_checkWaits = 0, m_kind = 0;
    std::future<bool> m_checkMake;
    enum { kCheckNone, kCheckMaking, kCheckWaiting, kCheckOk, kCheckFailed } m_check = kCheckNone;
    Texture* Keep(std::unique_ptr<Texture>& t, Format f, uint32_t w, uint32_t h);
    std::vector<std::unique_ptr<Network>> m_oldNets;
    std::vector<std::unique_ptr<Pass>> m_oldPasses;
    std::unique_ptr<Pass>    m_prepare, m_compose, m_menuPass;
    std::unique_ptr<Texture> m_prepared, m_composed[kMaxGenerated], m_menuCounts;
    Format   m_composeFormat = Format::Unknown;
    bool     m_staged = false, m_inMenu = false, m_hudlessLogged = false, m_formatLogged = false;
    uint32_t m_menuStep = 0;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> m_menuReads;
    const Texture* Crop(const ExternalImage& image, const Rect& r, bool depth);

    struct Cut { std::unique_ptr<Pass> pass; std::unique_ptr<Texture> tex; } m_cut[2];  // depth, motion
    std::vector<std::pair<std::unique_ptr<Texture>, uint64_t>> m_oldCuts;
    bool       m_vulkan;
    std::mutex m_mutex;
    std::vector<std::unique_ptr<Entry>> m_entries;  // most recent first, two kept
    std::vector<std::pair<std::unique_ptr<Entry>, uint64_t>> m_retired;
    Entry*     m_current = nullptr;
    float      m_vp[16] = {};
    bool       m_haveVp = false, m_restart = true, m_menu = false;
    uint32_t   m_ready = 0;       // generated frames of the current batch
    uint64_t   m_evaluations = 0, m_generated = 0;
};

Format   ClassOf(const ExternalImage& image, bool vulkan);
bool     IsDepthFormat(const ExternalImage& image, bool vulkan);
}  // namespace nfrux
