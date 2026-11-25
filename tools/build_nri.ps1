# Fetches NRI at the pinned commit and builds the static libraries NFRUX links. Needs the network once (NRI fetches VMA, D3D12MA and
# Vulkan-Headers at its own pins), CMake and MSVC.   tools\build_nri.ps1 [DIR]   (default third_party\NRI)
param([string]$Dir = "$PSScriptRoot\..\third_party\NRI")
$commit = '993db1a29f0c1e0d3e0c784d1666ef06f6403622'
if (-not (Test-Path $Dir)) { git init -q $Dir; git -C $Dir remote add origin https://github.com/NVIDIA-RTX/NRI }
git -C $Dir fetch -q --depth 1 origin $commit; git -C $Dir checkout -q $commit
cmake -S $Dir -B "$Dir\_build" -A x64 -DNRI_STATIC_LIBRARY=ON -DNRI_ENABLE_D3D11_SUPPORT=OFF -DNRI_ENABLE_NONE_SUPPORT=OFF -DNRI_ENABLE_AGILITY_SDK_SUPPORT=OFF -DNRI_ENABLE_NVAPI=OFF -DNRI_ENABLE_AMDAGS=OFF -DNRI_ENABLE_NVTX_SUPPORT=OFF
cmake --build "$Dir\_build" --config Release --parallel