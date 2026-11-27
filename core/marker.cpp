// SPDX-License-Identifier: MIT
#include "marker.h"

#include "log.h"

#include <windows.h>

#include <cstdio>
#include <map>
#include <mutex>

namespace nfrux
{
// Wide paths: a user name outside the system code page must not lose CUDA.
std::wstring NfruxFolder()
{
    std::wstring p(32768, 0);
    const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", p.data(), DWORD(p.size()));
    if (!n || n >= p.size()) return {};
    p.resize(n);
    std::wstring s = p + L"\\NFRUX";
    CreateDirectoryW(s.c_str(), nullptr);
    return s;
}
static std::wstring Path(const std::string& rung)
{
    const std::wstring s = NfruxFolder();
    if (s.empty()) return {};
    // A new driver is a new chance: its file's time is in the name.
    wchar_t dll[MAX_PATH] = {};
    GetSystemDirectoryW(dll, MAX_PATH - 20), wcscat_s(dll, L"\\nvcuda.dll");
    WIN32_FILE_ATTRIBUTE_DATA a = {};
    GetFileAttributesExW(dll, GetFileExInfoStandard, &a);
    wchar_t key[24];
    swprintf(key, 24, L"_%08lx", a.ftLastWriteTime.dwLowDateTime ^ a.ftLastWriteTime.dwHighDateTime);
    return s + L"\\" + std::wstring(rung.begin(), rung.end()) + key + L".attempts";
}
static std::string Utf8(const std::wstring& w)
{
    std::string s(w.size() * 3 + 1, 0);
    s.resize(WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), int(s.size()), nullptr, nullptr) - 1);
    return s;
}
bool MarkerBegin(const std::string& rung)
{
    if (EnvFlag("NFRUX_IGNORE_MARKERS", false)) return true;
    static std::mutex mutex;
    static std::map<std::string, bool> begun;  // one attempt a process, however many features
    std::lock_guard<std::mutex> lock(mutex);
    if (auto it = begun.find(rung); it != begun.end()) return it->second;
    bool& ok = begun[rung];
    const std::wstring p = Path(rung);
    if (p.empty()) return Log("%s: no %%LOCALAPPDATA%% for its crash marker: not used (IgnoreCrashMarkers=1 to use it anyway)", rung.c_str()), false;
    int count = 0;
    if (FILE* f = _wfopen(p.c_str(), L"r")) { if (fscanf(f, "%d", &count) != 1) count = 0; fclose(f); }
    if (count >= 2) return Log("%s: the last two starts did not finish their first frames: not used (delete %s to try again)", rung.c_str(), Utf8(p).c_str()), false;
    FILE* f = _wfopen(p.c_str(), L"w");
    if (!f) return Log("%s: cannot write its crash marker %s: not used (IgnoreCrashMarkers=1 to use it anyway)", rung.c_str(), Utf8(p).c_str()), false;
    fprintf(f, "%d\n", count + 1);
    Log("%s: crash marker %s (start %d)", rung.c_str(), Utf8(p).c_str(), count + 1);
    fclose(f);
    return ok = true;
}
void MarkerOk(const std::string& rung)
{
    if (const std::wstring p = Path(rung); !p.empty()) DeleteFileW(p.c_str()), RemoveDirectoryW(p.substr(0, p.rfind(L'\\')).c_str());  // (only when empty)
}}  // namespace nfrux