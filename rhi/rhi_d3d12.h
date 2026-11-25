// SPDX-License-Identifier: MIT
#pragma once
#include "rhi.h"
struct ID3D12Device;
namespace nfrux { std::unique_ptr<Device> CreateD3D12Device(ID3D12Device* device); }
