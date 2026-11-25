// Logging: OutputDebugString and nfrux.log (NFRUX_LOG=path, or =1: next to the DLL). SPDX-License-Identifier: MIT
#pragma once
namespace nfrux
{
void        Log(const char* fmt, ...);
const char* Env(const char* name);  // null when unset
bool        EnvFlag(const char* name, bool fallback);
}