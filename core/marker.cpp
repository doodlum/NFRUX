// SPDX-License-Identifier: MIT
#include "marker.h"

#include "log.h"

#include <windows.h>

#include <cstdio>
#include <map>
#include <mutex>

namespace nfrux
{
static std::string Path(const std::string& rung)
{
    char p[MAX_PATH];
    const DWORD n = GetEnvironmentVariableA("LOCALAPPDATA", p, MAX_PATH);
    if (!n || n > MAX_PATH - 100) return {};
    std::string s = std::string(p) + "\\NFRUX";
    CreateDirectoryA(s.c_str(), nullptr);
    // A new driver is a new chance: its file's time is in the name.
    wchar_t dll[MAX_PATH] = {};
    GetSystemDirectoryW(dll, MAX_PATH - 20), wcscat_s(dll, L"\\nvcuda.dll");
    WIN32_FILE_ATTRIBUTE_DATA a = {};
    GetFileAttributesExW(dll, GetFileExInfoStandard, &a);
    char key[24];
    snprintf(key, sizeof(key), "_%08lx", a.ftLastWriteTime.dwLowDateTime ^ a.ftLastWriteTime.dwHighDateTime);
    return s + "\\" + rung + key + ".attempts";
}
bool MarkerBegin(const std::string& rung)
{
    if (EnvFlag("NFRUX_IGNORE_MARKERS", false)) return true;
    static std::mutex mutex;
    static std::map<std::string, bool> begun;  // one attempt a process, however many features
    std::lock_guard<std::mutex> lock(mutex);
    if (auto it = begun.find(rung); it != begun.end()) return it->second;
    bool& ok = begun[rung];
    const std::string p = Path(rung);
    if (p.empty()) return false;
    int count = 0;
    if (FILE* f = fopen(p.c_str(), "r")) { if (fscanf(f, "%d", &count) != 1) count = 0; fclose(f); }
    if (count >= 2) return Log("%s: the last two starts did not finish their first frames: not used (delete %s to try again)", rung.c_str(), p.c_str()), false;
    FILE* f = fopen(p.c_str(), "w");
    if (!f) return false;
    fprintf(f, "%d\n", count + 1);
    fclose(f);
    return ok = true;
}
void MarkerOk(const std::string& rung)
{
    if (const std::string p = Path(rung); !p.empty()) DeleteFileA(p.c_str());
}
}  // namespace nfrux