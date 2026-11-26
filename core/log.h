// Logging: OutputDebugString and nfrux.log (NFRUX_LOG=path, or =1: next to the DLL). SPDX-License-Identifier: MIT
#pragma once
#include <string>
namespace nfrux
{
void        Log(const char* fmt, ...);
std::wstring DllFolder();  // NFRUX's DLL's folder, with the final backslash
const char* Env(const char* name);  // null when unset; nfrux.ini's value where it has that option
const char* Option(const char* section, const char* key);  // nfrux.ini's, lower case names; null when absent or auto
void        LogOptions();  // nfrux.ini's lines, once
bool        EnvFlag(const char* name, bool fallback);
}