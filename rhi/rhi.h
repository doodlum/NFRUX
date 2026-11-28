// What NFRUX's passes need of a graphics API: compute passes, textures, buffers and the network, recorded into the host's
// command buffer (VkCommandBuffer) or command list (ID3D12GraphicsCommandList) on the host's device.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nfrux
{
enum class Format : uint32_t
{
    Unknown, R8Unorm, RG16Float, R32Float, R32Uint, RGBA8Unorm, RGBA8Snorm, BGRA8Unorm, RGBA16Float, RGB10A2Unorm, RG11B10Float, Native
};

enum class BindKind : uint32_t { Cbv, SrvTexture, UavTexture, SrvBuffer, UavBuffer, Sampler };
struct ShaderBinding
{
    uint32_t binding;
    BindKind kind;
};
struct Shader
{
    const char*          name;
    const uint32_t*      spirv;
    size_t               spirvBytes;
    const uint32_t*      dxil;
    size_t               dxilBytes;
    const ShaderBinding* bindings;  // D3D12's (Vulkan reads them from the SPIR-V)
    uint32_t             bindingCount;
    uint32_t             shaderModel;  // D3D12's, 60 / 62 / 66
};

// A texture NFRUX owns, or one of the host's for one frame (Device::Wrap). uintView: an R32_UINT view was made.
class Texture
{
public:
    virtual ~Texture() = default;
    uint32_t width = 0, height = 0;
    Format   format   = Format::Unknown;
    bool     uintView = false;
    Format   storeFormat = Format::Unknown;  // a host output (Device::Wrap) a pass can write directly: the format it is written as
};
class Buffer
{
public:
    virtual ~Buffer() = default;
    uint64_t size = 0;
};

// One of the host's images. Vulkan: handle = VkImage, format = VkFormat, state = VkImageLayout. D3D12: handle =
// ID3D12Resource*, format = the DXGI_FORMAT of the view (0: the resource's), state = D3D12_RESOURCE_STATES.
struct ExternalImage
{
    void*    handle = nullptr;
    uint32_t format = 0;
    uint32_t width = 0, height = 0;
    uint32_t state = 0;
    uint32_t mip = 0, layer = 0;
    explicit operator bool() const { return handle != nullptr; }
};

struct Bind
{
    uint32_t       binding = 0;
    const Texture* texture = nullptr;
    bool           asUint  = false;  // the texture's R32_UINT view
    const Buffer*  buffer  = nullptr;
    const void*    data    = nullptr;
    size_t         size    = 0;
    static Bind Tex(uint32_t b, const Texture& t) { Bind x; x.binding = b, x.texture = &t; return x; }
    static Bind Uint(uint32_t b, const Texture& t) { Bind x; x.binding = b, x.texture = &t, x.asUint = true; return x; }
    static Bind Buf(uint32_t b, const Buffer& f) { Bind x; x.binding = b, x.buffer = &f; return x; }
    static Bind Data(uint32_t b, const void* d, size_t s) { Bind x; x.binding = b, x.data = d, x.size = s; return x; }
};

// What the GPU has (and the host enabled), as far as NFRUX picks its code by it.
struct Caps
{
    std::string name;
    uint32_t    vendorId = 0, deviceId = 0;
    uint32_t    driverId = 0;       // Vulkan: VkDriverId; D3D12: 0
    uint64_t    driverVersion = 0;  // Vulkan: driverVersion; D3D12: the user-mode driver's version
    bool        native16Bit = false, dot4 = false, cooperativeMatrix = false, cuda = false, opticalFlow = false;
    uint32_t    shaderModel = 0;  // D3D12: 60, 62, 64, 66, ...
    std::string Describe() const;
};

class Pass
{
public:
    virtual ~Pass() = default;
    // groupsZ: a batch's timesteps, one a z (the passes read it as their timestep)
    virtual bool Dispatch(const Bind* binds, uint32_t count, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ = 1) = 0;
};

// The network at one size. Input: the 20-channel int8 tensor; outputs: the blend parameters (4 int8) and the gate (its int8
// in the first of 4), as textures the composite samples (RGBA8Snorm with R32_UINT views).
class Network
{
public:
    virtual ~Network() = default;
    const Buffer* input = nullptr;
    const Texture *params = nullptr, *gate = nullptr;
};

class Device
{
public:
    virtual ~Device() = default;
    virtual const Caps& caps() const = 0;
    virtual std::unique_ptr<Texture> CreateTexture(Format format, uint32_t width, uint32_t height, bool uintView = false) = 0;
    // A texture of an external image's own format (the colour history). storage: also writable by passes where the format allows (storeFormat).
    virtual std::unique_ptr<Texture> CreateTextureLike(const ExternalImage& image, uint32_t width, uint32_t height, bool storage = false) = 0;
    virtual std::unique_ptr<Buffer>  CreateBuffer(uint64_t size) = 0;
    virtual std::unique_ptr<Pass>    CreatePass(const Shader& shader) = 0;
    virtual std::unique_ptr<Network> CreateNetwork(uint32_t width, uint32_t height, std::string* error) = 0;

    virtual bool DropRung() = 0;  // the next network is made one rung lower; false: there is none
    // A frame's recording into the host's command buffer / list; every wrapped image is back in its own state at EndFrame.
    virtual bool BeginFrame(void* commands) = 0;
    virtual void EndFrame() = 0;
    virtual Texture* Wrap(const ExternalImage& image, bool output) = 0;  // valid until EndFrame
    virtual void CopyTexture(const Texture& src, const Texture& dst, uint32_t dstX = 0, uint32_t dstY = 0) = 0;  // all of src
    virtual void CopyRegion(const Texture& src, uint32_t x, uint32_t y, const Texture& dst) = 0;                  // dst's size out of src
    virtual void CopyRect(const Texture& src, uint32_t sx, uint32_t sy, const Texture& dst, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h) = 0;
    virtual void WriteWord(void* hostBuffer, uint32_t value) = 0;  // 4 bytes at its start
    virtual void Clear(const Texture& t, float value) = 0;
    virtual void Fill(const Buffer& b, uint32_t value) = 0;
    virtual void RunNetworks(Network* const* nets, uint32_t count) = 0;
    // A texture's texels into host memory, tightly packed: filled at a BeginFrame or WaitIdle after the frame has completed.
    virtual void ReadBack(const Texture& t, std::shared_ptr<std::vector<uint8_t>> out) = 0;
    virtual void Mark(const char* label) = 0;  // profiling
    // GPU time (microseconds) per label of the last profiled frame, when profiling is on.
    virtual std::vector<std::pair<std::string, float>> Profile() { return {}; }
    virtual bool WaitIdle() = 0;  // before destroying what the GPU may still use; false: still busy after 2 s
    // Rung timing (core/rungs.h): GPU timestamps around what is recorded between Begin and End. A ticket (-1: none left);
    // its milliseconds once that recording has finished: -1 not yet, -2 unavailable.
    virtual int    TimerBegin() { return -1; }
    virtual void   TimerEnd(int ticket) {}
    virtual double TimerMs(int ticket) { return -2; }
};

uint32_t TexelSize(Format f);
}  // namespace nfrux
