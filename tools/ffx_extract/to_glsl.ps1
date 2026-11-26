# Converts every captured ML pass module to readable GLSL: drops the VariablePointers capability (the passes only use
# it for ByteAddressBuffer aliases that spirv-cross can express without it), reassembles, cross-compiles.
#   powershell -File tools\ffx_extract\to_glsl.ps1 CAPTURE_DIR
param([Parameter(Mandatory)][string]$Capture)
$bin = if ($env:VULKAN_SDK) { "$env:VULKAN_SDK\Bin" } else { 'C:\VulkanSDK\1.4.363.0\Bin' }
$out = "$Capture\glsl"; New-Item -ItemType Directory -Force $out | Out-Null
$tmp = "$Capture\glsl\tmp"; New-Item -ItemType Directory -Force $tmp | Out-Null
Get-Content "$Capture\pipelines.txt" | ForEach-Object {
    $a = $_ -split ' '; $mod = $a[1]; $entry = $a[2]
    if (Test-Path "$out\$entry.glsl") { return }
    & "$bin\spirv-dis.exe" "$Capture\modules\$mod.spv" -o "$tmp\$entry.spvasm"
    Get-Content "$tmp\$entry.spvasm" | Where-Object { $_ -notmatch 'OpCapability VariablePointers|SPV_KHR_variable_pointers' } | Set-Content -Encoding ascii "$tmp\$entry.novp.spvasm"
    & "$bin\spirv-as.exe" "$tmp\$entry.novp.spvasm" -o "$tmp\$entry.spv" --target-env spv1.5
    & "$bin\spirv-cross.exe" "$tmp\$entry.spv" --vulkan-semantics --output "$out\$entry.glsl"
    if (-not (Test-Path "$out\$entry.glsl")) { Write-Warning "no GLSL for $entry" }
}
Remove-Item -Recurse -Force $tmp
