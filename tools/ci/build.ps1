# The whole release build from a clean checkout: Vulkan SDK, NRI, the shipped and the strict build, the tests that need no GPU,
# then dist\NFRUX_<version>.zip, dist\NFRUX_bench_kit_<version>.zip and dist\SHA256SUMS.txt. Runs on a GitHub windows-2022 runner
# (.github/workflows/ci.yml) and on a developer machine, where it uses the installed Vulkan SDK and Visual Studio 2022.
#
#   tools\ci\build.ps1 [-BuildDir DIR] [-OutDir DIR] [-NriDir DIR] [-Tag vX.Y.Z] [-InstallVulkanSdk] [-NoWarp]
#
# -Tag: fail unless it is v<the built version> (the release workflow passes the pushed tag).
# -InstallVulkanSdk: download and install the pinned SDK when none is found (CI); otherwise a missing SDK is an error.
# -NoWarp: skip ngx_harness on WARP (the software D3D12 adapter; about a minute a build on a fast CPU, more on a runner).
param(
    [string]$BuildDir = "$PSScriptRoot\..\..\build\ci",
    [string]$OutDir = "$PSScriptRoot\..\..\dist",
    [string]$NriDir = "$PSScriptRoot\..\..\third_party\NRI",
    [string]$Tag = '',
    [switch]$InstallVulkanSdk,
    [switch]$NoWarp
)
$ErrorActionPreference = 'Stop'
$VulkanSdkVersion = '1.4.363.0'  # pinned; the workflow's cache key names it too
$started = Get-Date
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
function Step([string]$m) { Write-Host "`n== $m  (+$([int]((Get-Date) - $started).TotalSeconds)s)" }
function Invoke-Native([string]$what) { if ($LASTEXITCODE) { throw "$what failed (exit $LASTEXITCODE)" } }
foreach ($v in 'BuildDir', 'OutDir', 'NriDir') { Set-Variable $v $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath((Get-Variable $v).Value) }
New-Item -ItemType Directory -Force $BuildDir, $OutDir | Out-Null  # not NriDir: tools\build_nri.ps1 makes it a git repository when absent

# The version: CMakeLists.txt project(); the DLL's FileVersion is checked against it after the build.
$m = Select-String -Path "$Root\CMakeLists.txt" -Pattern '^\s*project\(\s*NFRUX\s+VERSION\s+(\d+\.\d+\.\d+)' | Select-Object -First 1
if (-not $m) { throw 'no project(NFRUX VERSION x.y.z) in CMakeLists.txt' }
$Version = $m.Matches[0].Groups[1].Value
if ($Tag -and $Tag -ne "v$Version") { throw "tag $Tag does not match the version in CMakeLists.txt ($Version)" }
Write-Host "NFRUX $Version"

Step 'Visual Studio 2022 and its CMake'
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw 'vswhere.exe not found: install Visual Studio 2022 with the C++ workload' }
$vs = & $vswhere -version '[17.0,18.0)' -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -latest -property installationPath
if (-not $vs) { throw 'Visual Studio 2022 with the x64 C++ tools not found' }
$cmakeBin = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
if (-not (Test-Path "$cmakeBin\cmake.exe")) { throw "no CMake in $vs (the 'C++ CMake tools for Windows' component)" }
# First on PATH, so tools\build_nri.ps1's plain `cmake` is this one too; the generator by environment for the same reason.
$env:PATH = "$cmakeBin;$env:PATH"
$env:CMAKE_GENERATOR = 'Visual Studio 17 2022'
$env:CMAKE_GENERATOR_INSTANCE = $vs
$cmake = "$cmakeBin\cmake.exe"
Write-Host "$vs`n$(& $cmake --version | Select-Object -First 1)"

Step 'Python 3'
$py = Get-Command python, py -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $py) { throw 'Python 3 not found on PATH' }
& $py.Source --version  # CMake's find_package(Python3) picks the interpreter itself

Step "Vulkan SDK (pinned $VulkanSdkVersion)"
function Test-VulkanSdk([string]$d) {
    $d -and (Test-Path "$d\Include\vulkan\vulkan.h") -and (Test-Path "$d\Lib\vulkan-1.lib") -and
    (Test-Path "$d\Bin\glslangValidator.exe") -and (Test-Path "$d\Bin\spirv-cross.exe") -and (Test-Path "$d\Bin\dxc.exe")
}
$sdk = $null
foreach ($c in @($env:VULKAN_SDK, "C:\VulkanSDK\$VulkanSdkVersion")) { if (Test-VulkanSdk $c) { $sdk = $c; break } }
if (-not $sdk -and $InstallVulkanSdk) {
    $sdk = "C:\VulkanSDK\$VulkanSdkVersion"
    $exe = "vulkansdk-windows-X64-$VulkanSdkVersion.exe"
    $dl = Join-Path ([IO.Path]::GetTempPath()) "nfrux_vksdk\$exe"
    New-Item -ItemType Directory -Force (Split-Path $dl) | Out-Null
    Write-Host "downloading $exe"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -UseBasicParsing -Uri "https://sdk.lunarg.com/sdk/download/$VulkanSdkVersion/windows/$exe" -OutFile $dl
    Write-Host "installing to $sdk"
    $p = Start-Process -FilePath $dl -ArgumentList '--root', $sdk, '--accept-licenses', '--default-answer', '--confirm-command', 'install' -Wait -PassThru
    if ($p.ExitCode) { throw "Vulkan SDK installer failed (exit $($p.ExitCode))" }
    Remove-Item $dl -Force
}
if (-not (Test-VulkanSdk $sdk)) { throw "no Vulkan SDK with glslangValidator, spirv-cross and dxc (VULKAN_SDK or C:\VulkanSDK\$VulkanSdkVersion); -InstallVulkanSdk installs it" }
$env:VULKAN_SDK = $sdk
Write-Host $sdk

Step "NRI ($NriDir)"
& "$Root\tools\build_nri.ps1" $NriDir
if (-not (Test-Path "$NriDir\_Bin\Release\NRI.lib")) { throw 'NRI did not build' }

$builds = [ordered]@{ ship = 'OFF'; strict = 'ON' }
foreach ($b in $builds.Keys) {
    $dir = "$BuildDir\$b"
    Step "configure $b (NFRUX_STRICT_MATH=$($builds[$b]))"
    & $cmake -S $Root -B $dir -A x64 "-DNRI_DIR=$NriDir" "-DNFRUX_STRICT_MATH=$($builds[$b])"; Invoke-Native "configure $b"
    Step "build $b"
    & $cmake --build $dir --config Release --parallel -- /nologo /verbosity:minimal; Invoke-Native "build $b"
    Step "rungs_test ($b)"
    & "$dir\Release\rungs_test.exe"; Invoke-Native "rungs_test ($b)"
    if (-not $NoWarp) {
        # The NGX harness on the software D3D12 adapter: no GPU needed. Exit 2 is no device (WARP missing or too old): a warning, not a failure.
        Step "ngx_harness on WARP ($b)"
        $env:NFRUX_WARP = '1'
        $out = & "$dir\Release\ngx_harness.exe" "$dir\NFRUX\Release\nvngx_dlssg.dll" d3d12
        $rc = $LASTEXITCODE
        Remove-Item Env:NFRUX_WARP
        $out | Where-Object { $_ -cmatch 'FAIL |device failed|no WARP' } | Write-Host
        $out | Select-Object -Last 1 | Write-Host
        if ($rc -eq 2) { Write-Warning "ngx_harness ($b): no WARP device, skipped" }
        elseif ($rc) { throw "ngx_harness on WARP ($b) failed (exit $rc)" }
    }
}

$ship = "$BuildDir\ship"
Step 'package and bench_kit (shipped build)'
& $cmake --build $ship --config Release --target bench_kit -- /nologo /verbosity:minimal; Invoke-Native 'bench_kit'  # depends on package

$dll = "$ship\package\nvngx_dlssg.dll"
$fv = [Diagnostics.FileVersionInfo]::GetVersionInfo($dll).FileVersion
if ($fv -ne $Version) { throw "the DLL's FileVersion ($fv) is not the CMakeLists.txt version ($Version)" }

Step "dist ($OutDir)"
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
function New-Zip([string]$src, [string]$zip) {
    # Entries relative to the folder with forward slashes, files sorted: the same listing on every run.
    if (Test-Path $zip) { Remove-Item $zip -Force }
    $z = [IO.Compression.ZipFile]::Open($zip, 'Create')
    try {
        Get-ChildItem $src -Recurse -File | Sort-Object FullName | ForEach-Object {
            $rel = $_.FullName.Substring($src.Length + 1).Replace('\', '/')
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($z, $_.FullName, $rel, 'Optimal')
        }
    } finally { $z.Dispose() }
}
Get-ChildItem $OutDir -File | Remove-Item -Force
$zips = @("NFRUX_$Version.zip", "NFRUX_bench_kit_$Version.zip")
New-Zip "$ship\package" "$OutDir\$($zips[0])"
New-Zip "$ship\bench_kit" "$OutDir\$($zips[1])"
$sums = foreach ($z in $zips) { "$((Get-FileHash "$OutDir\$z" -Algorithm SHA256).Hash.ToLower())  $z" }
[IO.File]::WriteAllText("$OutDir\SHA256SUMS.txt", ($sums -join "`n") + "`n")
[IO.File]::WriteAllText("$OutDir\version.txt", $Version)
Get-ChildItem $OutDir | Format-Table Name, Length -AutoSize | Out-String | Write-Host
$sums | Write-Host
Step "done in $([int]((Get-Date) - $started).TotalMinutes) min"
