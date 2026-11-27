// Which way to run a network on which GPU (DESIGN D3, rung choice): per-architecture priors by PCI device ID, a quirk list,
// and the timed choice cached in %LOCALAPPDATA%\NFRUX\rungs.txt.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nfrux
{
class Device;
struct Caps;
// CUDA kernels, cooperative matrices, dot4, 32-bit integers (the floor).
enum class Rung : uint8_t { Cuda, CoopMat, Dot4, Scalar };
const char* RungName(Rung r);  // cuda, coopmat, dot4, scalar
bool        RungFromName(const char* s, Rung* r);

struct Gpu
{
    uint32_t    vendorId = 0, deviceId = 0, driverId = 0;  // driverId: VkDriverId (0 on D3D12)
    std::string name;
    bool        vulkan = false;
};
struct RungPlan
{
    const char*       arch = "other";
    std::vector<Rung> order;           // fastest first, before the device's capabilities are checked
    const char*       quirk = nullptr;  // the quirk applied, if any
    bool              coopmatQuirk = false;  // cooperative matrices dropped or ranked below dot4 by a quirk
};
Gpu      GpuOf(const Caps& caps, bool vulkan);
RungPlan PlanFor(const Gpu& gpu);
bool     Has(const RungPlan& p, Rung r);
bool     Before(const RungPlan& p, Rung a, Rung b);  // a is listed and comes before b (or b is not listed)

// Timing: tickets of Device::TimerBegin per rung; Result: false while one is not back. best = the lowest median (each rung's
// first sample skipped: pipeline warm-up); false in *usable when timers are unavailable. text: "cuda 0.41, dot4 0.60 ms".
struct RungTimes
{
    std::vector<std::pair<Rung, std::vector<int>>> tickets;
    void Add(Rung r, int ticket);
    bool Result(Device& dev, Rung* best, bool* usable, std::string* text) const;
};

// The cache. prefix: what, API, GPU, driver version, NFRUX build; cls: ResolutionClass, or -1 for the latest entry of the prefix.
uint32_t    ResolutionClass(uint32_t w, uint32_t h);  // 0 up to 0.25 MP, 1 up to 1 MP, 2 up to 4 MP, 3 above
std::string CachePrefix(const char* what, const Gpu& gpu, uint64_t driverVersion);
bool        CachedRung(const std::string& prefix, int cls, Rung* r);
void        CacheRung(const std::string& prefix, uint32_t cls, Rung r);
}  // namespace nfrux
