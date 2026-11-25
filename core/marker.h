// Crash markers for optional rungs (DESIGN D3): two starts in a row that did not get through their first frames disable the rung.
// SPDX-License-Identifier: MIT
#pragma once
#include <string>
namespace nfrux
{
// Counts an attempt; false: the rung is not to be used (two attempts unfinished, or no folder to note it in).
bool MarkerBegin(const std::string& rung);
// The rung got through its first frames (or an orderly end): the count goes back to zero.
void MarkerOk(const std::string& rung);
}