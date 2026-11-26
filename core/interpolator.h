// Frame interpolation, ported from Arm's Neural Graphics SDK 1.1.1 (NFRU v2), on NFRUX's rhi with the
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
    uint32_t      uiKind = 0;  // 0 difference test, 2/3 UI alpha red / alpha over HUD-less, 4/5 without HUD-less
    bool          reset  = false, motionJittered = false;
    float         jitter[2] = {}, mvScale[2] = {1, 1};  // mvScale: to display pixels
    float         viewProjection[16] = {};
    float         cameraNear = 0.1f, cameraFar = 1000.0f, fovY = 1.0f;
    bool          orthographic = false;
};
// Which network interpolates: NFRUX's NFRU v2 model, or AMD FSR Frame Generation 4's (docs/FFX_MLFG.md).
enum class Model { Nfru, Amd };
Model SelectedModel();  // nfrux.ini [General] Model = nfru | amd, NFRUX_MODEL; default nfru

struct Settings
{
    Model    model = Model::Nfru;
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
    // The model settings.model names: NFRU (core/interpolator.cpp) or AMD's (core/mlfg/mlfg_interpolator.cpp).
    static std::unique_ptr<Interpolator> Create(Device& device, const Settings& settings, std::string* error);
    virtual ~Interpolator() = default;
    // Output i of count at t = (i + 1) / (total + 1). Textures, not the host's images: the front end copies them out.
    virtual bool Dispatch(const FrameInputs& in, uint32_t count, uint32_t total) = 0;  // between Device::BeginFrame and EndFrame
    virtual const Texture& Output(uint32_t i) const = 0;
    virtual const Settings& settings() const = 0;
    // For the parity harness: every named intermediate of the last dispatch.
    virtual std::vector<std::pair<std::string, const Texture*>> Intermediates() const = 0;
    // Tests: the NHWC fp16 tensors of the last dispatch (FSR FG4 only).
    struct TensorInfo { std::string name; const Buffer* buffer; uint32_t w, h, pitch; };
    virtual std::vector<TensorInfo> Tensors() const { return {}; }
};
std::unique_ptr<Interpolator> CreateNfruInterpolator(Device& device, const Settings& settings, std::string* error);
std::unique_ptr<Interpolator> CreateMlfgInterpolator(Device& device, const Settings& settings, std::string* error);
}  // namespace nfrux