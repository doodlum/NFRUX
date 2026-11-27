// SPDX-License-Identifier: MIT
#include "rungs.h"

#include "cuda_network.h"
#include "log.h"
#include "marker.h"
#include "rhi.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace nfrux
{
namespace
{
constexpr Rung C = Rung::Cuda, M = Rung::CoopMat, D = Rung::Dot4, S = Rung::Scalar;
// tensor: NVIDIA rows only, 1 tensor cores (device ID 0x2200 and up, Ampere on, or a Turing RTX by name), 0 none, -1 any.
struct Prior { uint32_t vendor, lo, hi; int tensor; const char* arch; Rung order[4]; uint32_t n; };
// The default order per architecture family when nothing is cached. First match wins.
const Prior kPriors[] = {
    {0x10DE, 0x0000, 0xFFFF, 1, "nvidia-rtx", {C, M, D, S}, 4},
    {0x10DE, 0x0000, 0xFFFF, 0, "nvidia-gtx", {D, M, S}, 3},
    {0x1002, 0x7440, 0x749F, -1, "amd-rdna3", {M, D, S}, 3},  // Navi 31 / 32 / 33
    {0x1002, 0x15BF, 0x15C8, -1, "amd-rdna3", {M, D, S}, 3},  // Phoenix, Hawk Point
    {0x1002, 0x150E, 0x150E, -1, "amd-rdna3", {M, D, S}, 3},  // Strix Point
    {0x1002, 0x1586, 0x1586, -1, "amd-rdna3", {M, D, S}, 3},  // Strix Halo
    {0x1002, 0x7550, 0x75FF, -1, "amd-rdna4", {M, D, S}, 3},  // Navi 44 / 48
    {0x1002, 0x0000, 0xFFFF, -1, "amd-rdna2-rdna1-gcn", {D, M, S}, 3},  // no matrix units: Steam Deck, RX 6000 / 5000, Vega
    {0x8086, 0x5690, 0x56BF, -1, "intel-arc-alchemist", {M, D, S}, 3},
    {0x8086, 0xE200, 0xE2FF, -1, "intel-arc-battlemage", {M, D, S}, 3},
    {0x8086, 0x0000, 0xFFFF, -1, "intel-igpu", {D, M, S}, 3},
};
// driverId 0: any. drop: cooperative matrices are not used; else they rank below dot4 (still timed).
struct Quirk { uint32_t vendor, lo, hi, driverId; int tensor; bool drop; const char* why; };
const Quirk kQuirks[] = {
    {0x10DE, 0x0000, 0xFFFF, 0, 0, true, "NVIDIA without tensor cores (GTX 16 and older): cooperative matrices are emulated by the driver, dot4 is faster"},
    {0x1002, 0x0000, 0xFFFF, 2, -1, false, "AMDVLK (VkDriverId 2): cooperative matrices slower than dot4"},
    {0x8086, 0x5690, 0x56BF, 0, -1, false, "Intel Arc A-series (Alchemist): cooperative matrices slower than dot4"},
};

// The hardware's vendor: Proton reports AMD GPUs as NVIDIA, so the driver and the name decide first.
uint32_t RealVendor(const Gpu& g)
{
    switch (g.driverId)
    {
    case 1: case 2: case 3: return 0x1002;  // AMD proprietary, AMDVLK, RADV
    case 5: case 6: return 0x8086;          // Intel Windows, Mesa
    case 4: case 24: return 0x10DE;         // NVIDIA, NVK
    default: break;
    }
    if (g.name.find("AMD") != std::string::npos || g.name.find("Radeon") != std::string::npos) return 0x1002;
    return g.vendorId == 0x1022 ? 0x1002 : g.vendorId;
}
bool Matches(uint32_t vendor, uint32_t lo, uint32_t hi, int tensor, const Gpu& g, uint32_t real)
{
    const bool tc = g.deviceId >= 0x2200 || HasTensorCores(g.name.c_str());
    return vendor == real && g.deviceId >= lo && g.deviceId <= hi && (tensor < 0 || tc == (tensor == 1));
}
uint32_t Build()
{
    HMODULE m = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&Build), &m);
    if (!m) return 0;
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(m);
    return reinterpret_cast<const IMAGE_NT_HEADERS*>(reinterpret_cast<const uint8_t*>(m) + dos->e_lfanew)->FileHeader.TimeDateStamp;
}
std::mutex g_cache;
std::vector<std::pair<std::string, std::string>> ReadCache(const std::wstring& path)
{
    std::vector<std::pair<std::string, std::string>> v;
    if (FILE* f = _wfopen(path.c_str(), L"r"))
    {
        char k[160], r[16];
        while (fscanf(f, "%159s %15s", k, r) == 2) v.push_back({k, r});
        fclose(f);
    }
    return v;
}
std::wstring CachePath() { const std::wstring d = NfruxFolder(); return d.empty() ? d : d + L"\\rungs.txt"; }
}  // namespace

const char* RungName(Rung r)
{
    static const char* const names[] = {"cuda", "coopmat", "dot4", "scalar"};
    return names[uint32_t(r)];
}
bool RungFromName(const char* s, Rung* r)
{
    for (uint32_t i = 0; i < 4; ++i) if (s && !strcmp(s, RungName(Rung(i)))) return *r = Rung(i), true;
    return false;
}

Gpu GpuOf(const Caps& c, bool vulkan)
{
    Gpu g;
    static const uint32_t pci[] = {0, 0x10DE, 0x1002, 0x8086};  // Caps.vendorId is NRI's Vendor: unknown, NVIDIA, AMD, Intel
    g.vendorId = c.vendorId < 4 ? pci[c.vendorId] : c.vendorId, g.deviceId = c.deviceId, g.driverId = c.driverId, g.name = c.name, g.vulkan = vulkan;
    return g;
}
RungPlan PlanFor(const Gpu& g)
{
    RungPlan p;
    const uint32_t real = RealVendor(g);
    p.order = {M, D, S};
    for (const Prior& r : kPriors)
        if (Matches(r.vendor, r.lo, r.hi, r.tensor, g, real)) { p.arch = r.arch, p.order.assign(r.order, r.order + r.n); break; }
    for (const Quirk& q : kQuirks)
    {
        if (!Matches(q.vendor, q.lo, q.hi, q.tensor, g, real) || (q.driverId && q.driverId != g.driverId)) continue;
        auto m = std::find(p.order.begin(), p.order.end(), M);
        if (m == p.order.end()) continue;
        p.order.erase(m), p.quirk = q.why, p.coopmatQuirk = true;
        if (!q.drop) p.order.insert(std::find(p.order.begin(), p.order.end(), D) + 1, M);  // dot4 is in every row
        break;
    }
    return p;
}
bool Has(const RungPlan& p, Rung r) { return std::find(p.order.begin(), p.order.end(), r) != p.order.end(); }
bool Before(const RungPlan& p, Rung a, Rung b)
{
    const auto ia = std::find(p.order.begin(), p.order.end(), a), ib = std::find(p.order.begin(), p.order.end(), b);
    return ia != p.order.end() && ia < ib;
}

void RungTimes::Add(Rung r, int ticket)
{
    for (auto& t : tickets) if (t.first == r) return t.second.push_back(ticket);
    tickets.push_back({r, {ticket}});
}
bool RungTimes::Result(Device& dev, Rung* best, bool* usable, std::string* text) const
{
    *usable = !tickets.empty();
    double bestMs = 1e30;
    std::string s;
    for (const auto& [r, v] : tickets)
    {
        std::vector<double> ms;
        for (size_t i = 0; i < v.size(); ++i)
        {
            const double t = v[i] < 0 ? -2 : dev.TimerMs(v[i]);
            if (t == -1) return false;  // not back yet
            if (t < 0) *usable = false;
            else if (i || v.size() == 1) ms.push_back(t);
        }
        if (ms.empty()) { *usable = false; continue; }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        if (med < bestMs) bestMs = med, *best = r;
        char b[48];
        snprintf(b, sizeof(b), "%s%s %.3f", s.empty() ? "" : ", ", RungName(r), med);
        s += b;
    }
    if (text) *text = s + " ms";
    return true;
}

uint32_t ResolutionClass(uint32_t w, uint32_t h)
{
    const uint64_t px = uint64_t(w) * h;
    return px <= 250000 ? 0 : px <= 1000000 ? 1 : px <= 4000000 ? 2 : 3;
}
std::string CachePrefix(const char* what, const Gpu& g, uint64_t driverVersion)
{
    char b[128];
    snprintf(b, sizeof(b), "%s_%s_%04x_%04x_%llx_%08x", what, g.vulkan ? "vulkan" : "d3d12", g.vendorId, g.deviceId, (unsigned long long)driverVersion, Build());
    return b;
}
bool CachedRung(const std::string& prefix, int cls, Rung* r)
{
    std::lock_guard<std::mutex> lock(g_cache);
    const std::wstring path = CachePath();
    if (path.empty()) return false;
    const std::string want = prefix + "_r" + (cls >= 0 ? std::to_string(cls) : "");
    bool found = false;
    for (const auto& [k, v] : ReadCache(path))
        if ((cls >= 0 ? k == want : k.compare(0, want.size(), want) == 0) && RungFromName(v.c_str(), r)) found = true;  // the last: the latest
    return found;
}
void CacheRung(const std::string& prefix, uint32_t cls, Rung r)
{
    std::lock_guard<std::mutex> lock(g_cache);
    const std::wstring path = CachePath();
    if (path.empty()) return;
    const std::string key = prefix + "_r" + std::to_string(cls);
    auto v = ReadCache(path);
    v.erase(std::remove_if(v.begin(), v.end(), [&](const auto& e) { return e.first == key; }), v.end());
    if (v.size() > 63) v.erase(v.begin());  // old drivers and builds go
    v.push_back({key, RungName(r)});
    if (FILE* f = _wfopen(path.c_str(), L"w"))
    {
        for (const auto& [k, x] : v) fprintf(f, "%s %s\n", k.c_str(), x.c_str());
        fclose(f);
    }
}
}  // namespace nfrux
