// SPDX-License-Identifier: MIT
#include "marker.h"

#include "log.h"

#include <windows.h>

#include <cstdio>

namespace nfrux
{
static std::string Path(const std::string& rung)
{
    char p[MAX_PATH];
    const DWORD n = GetEnvironmentVariableA("LOCALAPPDATA", p, MAX_PATH);
    if (!n || n > MAX_PATH - 100) return {};
    std::string s = std::string(p) + "\\NFRUX";
    CreateDirectoryA(s.c_str(), nullptr);
    return s + "\\" + rung + ".attempts";
}
bool MarkerBegin(const std::string& rung)
{
    if (EnvFlag("NFRUX_IGNORE_MARKERS", false)) return true;
    const std::string p = Path(rung);
    if (p.empty()) return false;
    int count = 0;
    if (FILE* f = fopen(p.c_str(), "r")) { if (fscanf(f, "%d", &count) != 1) count = 0; fclose(f); }
    if (count >= 2) return Log("%s: the last two starts did not finish their first frames: not used (delete %s to try again)", rung.c_str(), p.c_str()), false;
    FILE* f = fopen(p.c_str(), "w");
    if (!f) return false;
    fprintf(f, "%d\n", count + 1);
    fclose(f);
    return true;
}
void MarkerOk(const std::string& rung)
{
    if (const std::string p = Path(rung); !p.empty()) DeleteFileA(p.c_str());
}
}  // namespace nfrux