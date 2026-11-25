// The NGX core ABI as far as NFRUX uses it (declarations after DeckyUpscaler's ngx_abi.h, MIT). SPDX-License-Identifier: MIT
#pragma once
#include <d3d12.h>
#include <vulkan/vulkan.h>
#include <cstdint>
struct ID3D11Resource;
using NgxResult = uint32_t;
constexpr NgxResult kNgxSuccess = 1, kNgxFail = 0xBAD00000, kNgxInvalidParameter = 0xBAD00005, kNgxNotSupported = 0xBAD00001,
                    kNgxPlatformError = 0xBAD00002, kNgxUnsupportedParameter = 0xBAD00010, kNgxFeatureNotFound = 0xBAD00004;
constexpr int kFeatureFrameGeneration = 11;
struct NgxHandle { unsigned int Id; };
// The abstract class of nvsdk_ngx_params.h: MSVC lays out this overload set the same way.
struct NgxParameter
{
    virtual void Set(const char*, unsigned long long) = 0;
    virtual void Set(const char*, float) = 0;
    virtual void Set(const char*, double) = 0;
    virtual void Set(const char*, unsigned int) = 0;
    virtual void Set(const char*, int) = 0;
    virtual void Set(const char*, ID3D11Resource*) = 0;
    virtual void Set(const char*, ID3D12Resource*) = 0;
    virtual void Set(const char*, void*) = 0;
    virtual NgxResult Get(const char*, unsigned long long*) const = 0;
    virtual NgxResult Get(const char*, float*) const = 0;
    virtual NgxResult Get(const char*, double*) const = 0;
    virtual NgxResult Get(const char*, unsigned int*) const = 0;
    virtual NgxResult Get(const char*, int*) const = 0;
    virtual NgxResult Get(const char*, ID3D11Resource**) const = 0;
    virtual NgxResult Get(const char*, ID3D12Resource**) const = 0;
    virtual NgxResult Get(const char*, void**) const = 0;
    virtual void Reset() = 0;
};
struct NgxImageViewVk { VkImageView ImageView; VkImage Image; VkImageSubresourceRange Range; VkFormat Format; unsigned int Width, Height; };
struct NgxResourceVk { union { NgxImageViewVk Image; struct { VkBuffer Buffer; unsigned int Size; } Buffer; } R; int Type; bool ReadWrite; };
struct NgxFeatureDiscoveryInfo { int SDKVersion; int FeatureID; /* ... */ };
struct NgxFeatureRequirement { int FeatureSupported; unsigned int MinHWArchitecture; char MinOSVersion[255]; };