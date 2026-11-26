# Downloads the FidelityFX SDK signed frame-generation DLLs (not kept in this repo) and runs the extraction.
#   powershell -File tools\ffx_extract\fetch.ps1 [-Out C:\dev\modelgym\ref\ffx_signedbin_v2.4] [-Python python]
# The SDK's Kits/FidelityFX/docs/license.md (v2.4.0) lists Kits\FidelityFX\signedbin\amd_fidelityfx_framegeneration_*.dll
# in its MIT section, which permits modification; see docs/FFX_MLFG.md.
param([string]$Out = 'C:\dev\modelgym\ref\ffx_signedbin_v2.4', [string]$Python = 'python', [string]$Ref = 'v2.4.0')
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $Out | Out-Null
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
foreach ($a in 'dx12', 'vk') {
    $n = "amd_fidelityfx_framegeneration_$a.dll"
    if (-not (Test-Path "$Out\$n")) {
        Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/$Ref/Kits/FidelityFX/signedbin/$n" -OutFile "$Out\$n"
    }
    & $Python -I "$here\scan.py" "$Out\$n" "$Out\scan\$a"
}
$vk = "$Out\scan\vk"
New-Item -ItemType Directory -Force "$vk\dis" | Out-Null
$bin = if ($env:VULKAN_SDK) { "$env:VULKAN_SDK\Bin" } else { 'C:\VulkanSDK\1.4.363.0\Bin' }
Get-ChildItem "$vk\spirv\*.spv" | ForEach-Object { & "$bin\spirv-dis.exe" $_.FullName -o "$vk\dis\$($_.BaseName).spvasm" }
& $Python -I "$here\catalogue.py" "$Out\scan" "$Out\scan\catalogue.txt"
