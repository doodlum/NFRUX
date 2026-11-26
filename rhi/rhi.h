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
    bool        native16Bit = false, dot4 = false, cooperativeMatrix = false, cuda = false, opticalFlow = false;
    uint32_t    shaderModel = 0;  // D3D12: 60, 62, 64, 66, ...
    bool        vulkan = false, nvidia = false;
    // Vulkan: fp16 x fp16 -> fp32 subgroup cooperative matrices usable (16-bit storage and the memory model enabled): every
    // M x N x K the device reports (coopShapes), coopMatF16 = any of them; the subgroup size compute shaders get by default,
    // and the range a pass may require (Vulkan 1.3 subgroup size control; min = max = subgroupSize without it).
    struct MatShape { uint32_t m, n, k; };
    std::vector<MatShape> coopShapes;
    bool        coopMatF16 = false;
    uint32_t    subgroupSize = 0, subgroupMin = 0, subgroupMax = 0;
    std::string Describe() const;
};

class Pass
{
public:
    virtual ~Pass() = default;
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
    // A texture of an external image's own format (the colour history).
    virtual std::unique_ptr<Texture> CreateTextureLike(const ExternalImage& image, uint32_t width, uint32_t height) = 0;
    // cuda: CUDA kernels will use it too (Vulkan: allocated with a device address when CudaAvailable()).
    virtual std::unique_ptr<Buffer>  CreateBuffer(uint64_t size, bool cuda = false) = 0;
    // Host data into a buffer, recorded into the current frame (between BeginFrame and EndFrame); the staging copy is freed later.
    virtual bool Upload(const Buffer& dst, const void* data, uint64_t size) = 0;
    // waveLanes (Vulkan): the subgroup size the pass requires (0: the driver's choice); must be in [subgroupMin, subgroupMax].
    virtual std::unique_ptr<Pass>    CreatePass(const Shader& shader, uint32_t waveLanes = 0) = 0;
    virtual std::unique_ptr<Network> CreateNetwork(uint32_t width, uint32_t height, std::string* error) = 0;

    virtual bool DropRung() = 0;  // the next network is made one rung lower; false: there is none
    // CUDA kernels on NFRUX's buffers (core/mlfg; NVIDIA): D3D12 through NVAPI, Vulkan through VK_NVX_binary_import with
    // buffer device addresses. The adapter's LUID (to link PTX for it), modules and functions from a cubin, and a launch
    // into the current recording: the buffers it touches go to UNORDERED_ACCESS / storage first, with barriers.
    virtual bool     CudaAvailable() { return false; }
    virtual bool     CudaLuid(uint8_t luid[8]) { return false; }
    virtual void*    CudaModule(const std::vector<uint8_t>& cubin) { return nullptr; }
    virtual void*    CudaFunction(void* module, const char* name) { return nullptr; }
    virtual void     CudaDestroy(void* module, const std::vector<void*>& functions) {}
    virtual bool     CudaLaunch(void* function, const uint32_t grid[3], uint32_t block, const void* params, uint32_t size, const Buffer* const* buffers, uint32_t count) { return false; }
    virtual uint64_t NativeAddress(const Buffer& buffer) { return 0; }  // its address for the kernels
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
    virtual void ReadBack(const Buffer& b, std::shared_ptr<std::vector<uint8_t>> out) = 0;  // all of it, the same way
    virtual void Mark(const char* label) = 0;  // profiling
    virtual void Label(const char*) {}         // profiling: the next pass's name in the times
    virtual bool Timing() const { return false; }
    // GPU time (microseconds) per label of the last profiled frame, when profiling is on.
    virtual std::vector<std::pair<std::string, float>> Profile() { return {}; }
    virtual bool WaitIdle() = 0;  // before destroying what the GPU may still use; false: still busy after 2 s
};

uint32_t TexelSize(Format f);
}  // namespace nfrux
