# Fetches NRI at the pinned commit and builds the static libraries NFRUX links. Needs the network once (NRI fetches VMA, D3D12MA and
# Vulkan-Headers at its own pins), CMake and MSVC.   tools\build_nri.ps1 [DIR]   (default third_party\NRI)
param([string]$Dir = "$PSScriptRoot\..\third_party\NRI")
$commit = '993db1a29f0c1e0d3e0c784d1666ef06f6403622'
if (-not (Test-Path $Dir)) { git init -q $Dir; git -C $Dir remote add origin https://github.com/NVIDIA-RTX/NRI }
git -C $Dir fetch -q --depth 1 origin $commit; git -C $Dir checkout -q -f $commit
# One change: SHADER_RESOURCE is NON_PIXEL_SHADER_RESOURCE on DIRECT lists too (NRI adds PIXEL there). NFRUX has only compute passes, its host's images
# arrive in NON_PIXEL_SHADER_RESOURCE, and textures go from a COMPUTE list to a DIRECT one.
$p = "$Dir\Source\D3D12\CommandBufferD3D12.hpp"
$t = [IO.File]::ReadAllText($p)
$n = [regex]::Replace($t, '(resourceStates \|= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;)\s+if \(commandListType == D3D12_COMMAND_LIST_TYPE_DIRECT\)\s+resourceStates \|= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;', '$1 (void)commandListType;')
if ($n -ne $t) { [IO.File]::WriteAllText($p, $n) }
elseif ($t -match 'commandListType == D3D12_COMMAND_LIST_TYPE_DIRECT\)\s+resourceStates') { throw 'NRI: the state patch does not apply' }
# Serviced Windows SDKs (10.0.26100.0 on current hosted runners) declare the opacity-micromap structs but not the alignment constant NRI tests
# for, so NRI's stand-ins would redefine them. Where the SDK CMake picks (the newest) declares them, the constant is defined first.
$sdk = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\Include" -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path "$($_.FullName)\um\d3d12.h" } |
    Sort-Object { [version]$_.Name } | Select-Object -Last 1
$p = "$Dir\Source\D3D12\SharedD3D12.h"
$t = [IO.File]::ReadAllText($p)
$guard = '#ifndef D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_BYTE_ALIGNMENT'
if ($sdk -and (Select-String "$($sdk.FullName)\um\d3d12.h" -Pattern 'D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC' -SimpleMatch -Quiet) -and
    -not (Select-String "$($sdk.FullName)\um\d3d12.h" -Pattern 'define D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_BYTE_ALIGNMENT' -SimpleMatch -Quiet))
{
    if (-not $t.Contains($guard)) { throw 'NRI: the micromap patch does not apply' }
    Write-Host "NRI: Windows SDK $($sdk.Name) declares the micromap structs: NRI's stand-ins are skipped"
    [IO.File]::WriteAllText($p, $t.Replace($guard, "#define D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_BYTE_ALIGNMENT 128  // NFRUX build_nri.ps1: the SDK declares the structs`n$guard"))
}
cmake -S $Dir -B "$Dir\_build" -A x64 -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DNRI_STATIC_LIBRARY=ON -DNRI_ENABLE_D3D11_SUPPORT=OFF -DNRI_ENABLE_NONE_SUPPORT=OFF -DNRI_ENABLE_AGILITY_SDK_SUPPORT=OFF -DNRI_ENABLE_NVAPI=OFF -DNRI_ENABLE_AMDAGS=OFF -DNRI_ENABLE_NVTX_SUPPORT=OFF
cmake --build "$Dir\_build" --config Release --parallel
if ($LASTEXITCODE) { throw 'NRI did not build' }