// SPDX-License-Identifier: MIT
#include "log.h"

#include <windows.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace nfrux
{
// nfrux.ini beside the DLL. An environment variable wins over it. Read once; never logs itself (Log reads it): LogOptions does.
namespace
{
struct Known { const char *section, *key, *env; bool flag; };
const Known kKnown[] = {
    {"general", "maxmultiplier", nullptr, false},
    {"general", "ignorecrashmarkers", "NFRUX_IGNORE_MARKERS", true},
    {"general", "log", "NFRUX_LOG", false},
    {"d3d12", "network", nullptr, false},
    {"vulkan", "network", nullptr, false},
    {"vulkan", "dot4tiled", "FFX_ML_DOT4_TILED", true},
    {"vulkan", "dot4subgroup", "FFX_ML_DOT4_SUBGROUP", false},
    {"vulkan", "fullbarriers", "NFRUX_FULL_BARRIERS", true},
};
struct Ini
{
    std::map<std::string, std::string> values;  // "section.key"
    std::vector<std::string> notes;
    Ini()
    {
        const std::wstring path = ModuleFolder() + L"nfrux.ini";
        FILE* f = _wfopen(path.c_str(), L"r");
        if (!f) return;
        auto trim = [](std::string s) {
            const size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
            return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
        };
        auto lower = [](std::string s) { for (char& c : s) if (c >= 'A' && c <= 'Z') c += 32; return s; };
        std::string section;
        char raw[1024];
        for (bool first = true; fgets(raw, sizeof(raw), f); first = false)
        {
            std::string s = raw;
            if (first && !s.compare(0, 3, "\xEF\xBB\xBF")) s.erase(0, 3);
            if (const size_t c = s.find(" ;"); c != std::string::npos) s.erase(c);
            s = trim(s);
            if (s.empty() || s[0] == ';' || s[0] == '#') continue;
            if (s[0] == '[') { section = lower(trim(s.substr(1, s.find(']') - 1))); continue; }
            const size_t eq = s.find('=');
            const std::string key = lower(trim(s.substr(0, eq)));
            std::string value = eq == std::string::npos ? "" : trim(s.substr(eq + 1));
            const Known* known = nullptr;
            for (const Known& k : kKnown) if (section == k.section && key == k.key) known = &k;
            if (!known) { notes.push_back("nfrux.ini: unknown [" + section + "] " + key); continue; }
            const std::string v = lower(value);
            if (v.empty() || v == "auto") continue;
            if (known->flag)
            {
                if (v == "1" || v == "true" || v == "on" || v == "yes") value = "1";
                else if (v == "0" || v == "false" || v == "off" || v == "no") value = "0";
                else { notes.push_back("nfrux.ini: " + key + "=" + value + " is not auto, 0 or 1: ignored"); continue; }
            }
            else if (key != "log") value = v;
            else if (v == "true" || v == "on" || v == "yes") value = "1";  // Log: a flag or a file name
            else if (v == "false" || v == "off" || v == "no") value = "0";
            values[section + "." + key] = value;
            notes.push_back("nfrux.ini: [" + section + "] " + key + "=" + value);
        }
        fclose(f);
    }
    static std::wstring ModuleFolder()
    {
        HMODULE module = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&kKnown), &module);
        std::wstring p(32768, 0);
        p.resize(GetModuleFileNameW(module, p.data(), DWORD(p.size())));
        return p.substr(0, p.find_last_of(L"\\/") + 1);
    }
};
}
std::wstring DllFolder() { return Ini::ModuleFolder(); }
namespace
{
const Ini& TheIni() { static const Ini& ini = *new Ini; return ini; }
}
const char* Option(const char* section, const char* key)
{
    const auto it = TheIni().values.find(std::string(section) + "." + key);
    return it == TheIni().values.end() ? nullptr : it->second.c_str();
}
const char* Env(const char* name)
{
    if (const char* v = getenv(name); v && *v) return v;
    for (const Known& k : kKnown) if (k.env && !strcmp(k.env, name)) return Option(k.section, k.key);
    return nullptr;
}
void LogOptions()
{
    static const bool once = [] {
        Log("NFRUX %s", NFRUX_VERSION_STRING);
        for (const auto& s : TheIni().notes) Log("%s", s.c_str());
        return true;
    }();
    (void)once;
}
bool        EnvFlag(const char* name, bool fallback) { const char* v = Env(name); return v ? strcmp(v, "0") != 0 : fallback; }

static std::wstring LogPath()
{
    const char* v = Env("NFRUX_LOG");
    if (!v || !strcmp(v, "0"))
        return {};
    if (!strcmp(v, "1"))
        return DllFolder() + L"nfrux.log";
    std::wstring w(strlen(v) + 1, 0);  // a path the user gave, in the system code page
    w.resize(MultiByteToWideChar(CP_ACP, 0, v, -1, w.data(), int(w.size())) - 1);
    return w;
}
void Log(const char* fmt, ...)
{
    static std::mutex m;
    static FILE*      file = [] { const std::wstring p = LogPath(); return p.empty() ? nullptr : _wfopen(p.c_str(), L"w"); }();
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