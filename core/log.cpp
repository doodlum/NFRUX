// SPDX-License-Identifier: MIT
#include "log.h"

#include <windows.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace nfrux
{
const char* Env(const char* name) { const char* v = getenv(name); return v && *v ? v : nullptr; }
bool        EnvFlag(const char* name, bool fallback) { const char* v = Env(name); return v ? strcmp(v, "0") != 0 : fallback; }

static std::string LogPath()
{
    const char* v = Env("NFRUX_LOG");
    if (!v || !strcmp(v, "0"))
        return {};
    if (strcmp(v, "1"))
        return v;
    HMODULE module = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCSTR>(&LogPath), &module);
    char path[MAX_PATH] = {};
    GetModuleFileNameA(module, path, MAX_PATH);
    const std::string s = path;
    return s.substr(0, s.find_last_of("\\/") + 1) + "nfrux.log";
}

void Log(const char* fmt, ...)
{
    static std::mutex m;
    static FILE*      file = [] { const std::string p = LogPath(); return p.empty() ? nullptr : fopen(p.c_str(), "w"); }();
    char    line[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);
    std::lock_guard<std::mutex> lock(m);
    OutputDebugStringA((std::string("[NFRUX] ") + line + "\n").c_str());
    if (file)
        fprintf(file, "%s\n", line), fflush(file);
}
}  // namespace nfrux