// The rung plan (core/rungs.h) for GPUs this machine does not have: expected order and quirk per architecture. Exit 0: all pass.
// SPDX-License-Identifier: MIT
#include "rungs.h"

#include <windows.h>

#include <cstdio>
#include <string>

using namespace nfrux;

int main()
{
    struct Case { const char* name; uint32_t vendor, device, driver; bool vulkan; const char* arch; const char* order; bool quirk; };
    const Case cases[] = {
        {"NVIDIA GeForce RTX 4070", 0x10DE, 0x2786, 4, true, "nvidia-rtx", "cuda coopmat dot4 scalar", false},
        {"NVIDIA GeForce RTX 2060", 0x10DE, 0x1F08, 4, true, "nvidia-rtx", "cuda coopmat dot4 scalar", false},
        {"NVIDIA GeForce GTX 1660 SUPER", 0x10DE, 0x21C4, 4, true, "nvidia-gtx", "dot4 scalar", true},
        {"NVIDIA GeForce GTX 1060 6GB", 0x10DE, 0x1C03, 0, false, "nvidia-gtx", "dot4 scalar", true},
        {"AMD Radeon RX 7900 XTX", 0x1002, 0x744C, 3, true, "amd-rdna3", "coopmat dot4 scalar", false},             // RADV
        {"AMD Radeon RX 7900 XTX", 0x1002, 0x744C, 2, true, "amd-rdna3", "dot4 coopmat scalar", true},              // AMDVLK
        {"AMD Radeon RX 9070 XT", 0x1002, 0x7550, 1, true, "amd-rdna4", "coopmat dot4 scalar", false},
        {"AMD Radeon RX 6800", 0x1002, 0x73BF, 1, true, "amd-rdna2-rdna1-gcn", "dot4 coopmat scalar", false},
        {"AMD Custom GPU 0405 (Steam Deck)", 0x1002, 0x1435, 3, true, "amd-rdna2-rdna1-gcn", "dot4 coopmat scalar", false},
        {"AMD Radeon 890M", 0x1002, 0x150E, 1, true, "amd-rdna3", "coopmat dot4 scalar", false},
        {"Intel(R) Arc(TM) A770 Graphics", 0x8086, 0x56A0, 5, true, "intel-arc-alchemist", "dot4 coopmat scalar", true},
        {"Intel(R) Arc(TM) B580 Graphics", 0x8086, 0xE20B, 5, true, "intel-arc-battlemage", "coopmat dot4 scalar", false},
        {"Intel(R) UHD Graphics 770", 0x8086, 0x4680, 5, true, "intel-igpu", "dot4 coopmat scalar", false},
        {"AMD Radeon RX 7800 XT (reported as NVIDIA)", 0x10DE, 0x2487, 3, true, "amd-rdna2-rdna1-gcn", "dot4 coopmat scalar", false},
    };
    int failed = 0;
    for (const Case& k : cases)
    {
        Gpu g;
        g.vendorId = k.vendor, g.deviceId = k.device, g.driverId = k.driver, g.name = k.name, g.vulkan = k.vulkan;
        const RungPlan p = PlanFor(g);
        std::string o;
        for (Rung r : p.order) o += std::string(o.empty() ? "" : " ") + RungName(r);
        const bool ok = o == k.order && std::string(p.arch) == k.arch && (p.quirk != nullptr) == k.quirk;
        failed += !ok;
        printf("%s %-44s %-22s %-24s%s%s\n", ok ? "ok  " : "FAIL", k.name, p.arch, o.c_str(), p.quirk ? " quirk: " : "", p.quirk ? p.quirk : "");
    }
    // The cache (in a folder of its own, not the user's): a written entry is read back for its class, and the latest of a prefix for -1.
    char tmp[MAX_PATH] = {};
    GetTempPathA(MAX_PATH, tmp);
    SetEnvironmentVariableA("LOCALAPPDATA", (std::string(tmp) + "nfrux_rungs_test").c_str());
    CreateDirectoryA((std::string(tmp) + "nfrux_rungs_test").c_str(), nullptr);
    Gpu g;
    g.vendorId = 0xFFFF, g.deviceId = 0xFFFF, g.name = "test";
    const std::string prefix = CachePrefix("test", g, 1);
    CacheRung(prefix, 1, Rung::Dot4), CacheRung(prefix, 2, Rung::Cuda);
    Rung r1 = Rung::Scalar, r2 = Rung::Scalar, rl = Rung::Scalar, r3 = Rung::Scalar;
    const bool c1 = CachedRung(prefix, 1, &r1), c2 = CachedRung(prefix, 2, &r2), cl = CachedRung(prefix, -1, &rl), c3 = CachedRung(prefix, 3, &r3);
    const bool cacheOk = c1 && r1 == Rung::Dot4 && c2 && r2 == Rung::Cuda && cl && rl == Rung::Cuda && !c3;
    failed += !cacheOk;
    printf("%s cache read back (class 1 dot4, class 2 cuda, latest cuda, class 3 none)\n", cacheOk ? "ok  " : "FAIL");
    printf("%s\n", failed ? "FAILED" : "ALL PASSED");
    return failed ? 1 : 0;
}
