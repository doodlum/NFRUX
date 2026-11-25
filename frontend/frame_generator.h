// One NGX frame generation feature: reads an Evaluate's parameters and records its frame (docs/DESIGN.md section 4).
// SPDX-License-Identifier: MIT
#pragma once
#include "../core/interpolator.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nfrux
{
struct Rect { uint32_t x = 0, y = 0, w = 0, h = 0; };
struct EvalInputs
{
    ExternalImage back, output, outputReal, hudless, ui, uiAlpha, noWarp, depth, motion;
    Rect  backRect, hudlessRect, uiRect, uiAlphaRect, noWarpRect, depthRect, motionRect;
    float mvScale[2] = {1, 1}, jitter[2] = {}, cameraNear = 0.1f, cameraFar = 1000.0f, fovY = 1.0f;
    float viewToClip[16] = {}, clipToPrevClip[16] = {}, pos[3] = {}, up[3] = {}, right[3] = {}, fwd[3] = {};
    bool  hasClipToPrevClip = false, depthInverted = false, hdr = false, reset = false, notRendering = false, ortho = false;
    uint32_t count = 1, index = 1;  // DLSSG.MultiFrameCount / MultiFrameIndex
};

class FrameGenerator
{
public:
    // formatOf: an external image's format class for the composite (Unknown: not supported).
    FrameGenerator(Device& device, bool vulkan) : m_dev(device), m_vulkan(vulkan) {}
    ~FrameGenerator();
    // Records one Evaluate. False: nothing usable was passed (the caller reports an invalid parameter).
    bool Evaluate(void* commands, const EvalInputs& in);
    void Restart() { m_restart = true; }

private:
    struct Entry
    {
        std::unique_ptr<Interpolator> ip;
        uint32_t w, h, rw, rh, format;
        bool     inverted;
        bool     failed = false;
    };
    Entry* Get(const ExternalImage& color, const EvalInputs& in, uint32_t w, uint32_t h, uint32_t rw, uint32_t rh);
    bool   Generate(const EvalInputs& in);
    ExternalImage Crop(const ExternalImage& image, const Rect& r, std::unique_ptr<Texture>& cache);

    Device&    m_dev;
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
