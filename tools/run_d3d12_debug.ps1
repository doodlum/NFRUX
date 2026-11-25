# Installs Windows' Graphics Tools (asks for elevation) when they are missing, then runs the harness on the D3D12 debug layer.
$root = Split-Path $PSScriptRoot
if (-not (Test-Path "$env:WINDIR\System32\d3d12SDKLayers.dll")) {
    Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile -Command "Add-WindowsCapability -Online -Name Tools.Graphics.DirectX~~~~0.0.1.0"'
}
if (-not (Test-Path "$env:WINDIR\System32\d3d12SDKLayers.dll")) { Write-Error 'Graphics Tools are still not installed.'; exit 1 }
foreach ($level in 1, 2) {  # 2: with GPU-based validation
    $env:NFRUX_D3D12_DEBUG = $level
    "== debug layer, level $level"
    & "$root\build-ship\Release\ngx_harness.exe" "$root\build-ship\NFRUX\Release\nvngx_dlssg.dll" d3d12 | Tee-Object "$root\build\d3d12_debug_$level.txt" | Select-String 'debug layer|GPU validation|D3D12:|FAIL|messages|checks,'
}
Remove-Item Env:NFRUX_D3D12_DEBUG
