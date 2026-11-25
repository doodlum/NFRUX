// Frame interpolation, ported from Arm's Neural Graphics SDK 1.1.1 (NFRU v2) through DeckyUpscaler, on NFRUX's rhi with the
// 20-channel multi-frame network. SPDX-License-Identifier: MIT
#pragma once
#include "rhi.h"

namespace nfrux
{
constexpr uint32_t kMaxGenerated = 5;

struct FrameInputs
{
    ExternalImage color;     // the frame interpolated (HUD-less when the game has one)
    ExternalImage uiSource;  // the frame with its UI, or none
    ExternalImage uiAlpha, noWarp, depth, motion;
    const Texture *depthTexture = nullptr, *motionTexture = nullptr;  // instead of depth / motion: already cut to the rendered region
    const Texture* colorTexture = nullptr;  // instead of color, of Settings::colorLike's format
    bool          scrgb = false;  // linear HDR: the UI difference test compresses its values
    uint32_t      uiKind = 0;  // 0 difference test, 2/3 UI alpha red / alpha over HUD-less, 4/5 without HUD-less
    bool          reset  = false;
    float         jitter[2] = {}, mvScale[2] = {1, 1};  // mvScale: to display pixels
    float         viewProjection[16] = {};
    float         cameraNear = 0.1f, cameraFar = 1000.0f, fovY = 1.0f;
    bool          orthographic = false;
};
struct Settings
{
    uint32_t displayWidth = 0, displayHeight = 0, renderWidth = 0, renderHeight = 0;
    ExternalImage colorLike;  // an image of the colour's format
    Format   outputFormat = Format::RGBA8Unorm;  // the composite's UAV format
    bool     swapRB = false;   // stored blue first into RGBA8 scratch (Vulkan BGRA8 outputs)
    bool     depthInverted = false;
    float    initialViewProjection[16] = {};
};

class Interpolator
{
public:
    static std::unique_ptr<Interpolator> Create(Device& device, const Settings& settings, std::string* error);
    ~Interpolator();
    // Output i of count at t = (i + 1) / (total + 1). Textures, not the host's images: the front end copies them out.
    bool Dispatch(const FrameInputs& in, uint32_t count, uint32_t total);  // between Device::BeginFrame and EndFrame
    const Texture& Output(uint32_t i) const;
    const Settings& settings() const;
    // For the parity harness: every named intermediate of the last dispatch.
    std::vector<std::pair<std::string, const Texture*>> Intermediates() const;

private:
    Interpolator() = default;
    struct Impl;
    std::unique_ptr<Impl> m;
};
}  // namespace nfrux