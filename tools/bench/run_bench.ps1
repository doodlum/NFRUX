<#
NFRUX benchmark kit: checks that every network path gives the right bytes on this GPU, times every variant, and writes one report
(docs/OPTIMISING.md, "Running the kit"). Headless: no window opens. Close games first; the numbers are only as clean as the GPU is idle.

  powershell -ExecutionPolicy Bypass -File run_bench.ps1            the full run (about 10 minutes)
  powershell -ExecutionPolicy Bypass -File run_bench.ps1 -Quick     1080p only (about 3 minutes)

Output: report\nfrux_bench_<gpu>_<date>.md (send this file) and report\raw\ (the logs, if asked for).
SPDX-License-Identifier: MIT
#>
param([switch]$Quick)
$ErrorActionPreference = 'Continue'
$kit = $PSScriptRoot
$dll = Join-Path $kit 'nvngx_dlssg.dll'
foreach ($f in 'nvngx_dlssg.dll', 'ngx_harness.exe', 'ml_conformance_d3d12.exe', 'ml_conformance_vulkan.exe', 'parity.exe') {
    if (-not (Test-Path (Join-Path $kit $f))) { Write-Host "missing $f beside this script"; exit 2 }
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmm'
$raw = Join-Path $kit "report\raw\$stamp"
New-Item -ItemType Directory -Force $raw | Out-Null
# NFRUX's crash markers and timed-path cache go to a folder of this run, not the user's.
$env:LOCALAPPDATA = Join-Path $raw 'localappdata'
New-Item -ItemType Directory -Force $env:LOCALAPPDATA | Out-Null
foreach ($v in 'NFRUX_ML', 'NFRUX_TEST_CUDA_VARIANT', 'FFX_ML_DOT4_TILED', 'FFX_ML_DOT4_SUBGROUP', 'NFRUX_TEST_TIMES') { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }

function Invoke-Tool([string]$name, [string[]]$toolArgs, [string]$log) {
    $out = & (Join-Path $kit $name) @toolArgs 2>&1 | Out-String
    Set-Content -Path (Join-Path $raw $log) -Value $out -Encoding UTF8
    return @{ out = $out; code = $LASTEXITCODE }
}

# --- The GPU
$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.PNPDeviceID -match 'VEN_(10DE|1002|8086)' } | Select-Object -First 1
if (-not $gpu) { $gpu = Get-CimInstance Win32_VideoController | Select-Object -First 1 }
$ven = if ($gpu.PNPDeviceID -match 'VEN_([0-9A-F]{4})') { $matches[1] } else { '?' }
$dev = if ($gpu.PNPDeviceID -match 'DEV_([0-9A-F]{4})') { $matches[1] } else { '?' }
$os = (Get-CimInstance Win32_OperatingSystem).Version
Write-Host "GPU: $($gpu.Name) (VEN_$ven DEV_$dev), driver $($gpu.DriverVersion)"

# --- Correctness: every path this GPU has must give the reference bytes
Write-Host 'correctness...'
$checks = @()
foreach ($api in 'd3d12', 'vulkan') {
    $env:NFRUX_LOG = Join-Path $raw "nfrux_$api.log"
    $r = Invoke-Tool 'ngx_harness.exe' @($dll, $api) "harness_$api.txt"
    $checks += [pscustomobject]@{ test = "ngx_harness $api"; result = (($r.out -split "`n" | Where-Object { $_ -match 'checks' } | Select-Object -Last 1) -as [string]).Trim(); ok = $r.code -eq 0 }
}
Remove-Item Env:NFRUX_LOG
foreach ($t in 'ml_conformance_d3d12', 'ml_conformance_vulkan', 'parity') {
    $r = Invoke-Tool "$t.exe" @() "$t.txt"
    $last = ($r.out.Trim() -split "`n" | Select-Object -Last 1).Trim()
    $checks += [pscustomobject]@{ test = $t; result = $last; ok = $r.code -eq 0 }
}
$plan = @()
foreach ($api in 'd3d12', 'vulkan') {
    $l = Join-Path $raw "nfrux_$api.log"
    if (Test-Path $l) { $plan += Get-Content $l | Where-Object { $_ -match '^(NFRUX |D3D12 \(NRI\)|Vulkan \(NRI\)|rungs:|rung timing|network \d+x\d+:)' } }
}

# --- Timing: every variant, through ngx_harness --time (each evaluate submitted and waited for; ms a rendered frame, medians)
$sizes = if ($Quick) { @('1920x1080') } else { @('1280x720', '1920x1080', '2560x1440', '3840x2160') }
$gens = @(1, 3, 5)  # generated frames: 2x, 4x, 6x
$variants = @(
    @{ api = 'd3d12';  name = 'auto';           env = @{} },
    @{ api = 'd3d12';  name = 'cuda/0 tiled';   env = @{ NFRUX_ML = 'cuda'; NFRUX_TEST_CUDA_VARIANT = '0' } },
    @{ api = 'd3d12';  name = 'cuda/1 rows';    env = @{ NFRUX_ML = 'cuda'; NFRUX_TEST_CUDA_VARIANT = '1' } },
    @{ api = 'd3d12';  name = 'dot4';           env = @{ NFRUX_ML = 'dot4' } },
    @{ api = 'd3d12';  name = 'scalar';         env = @{ NFRUX_ML = 'scalar' }; small = $true },
    @{ api = 'vulkan'; name = 'auto';           env = @{} },
    @{ api = 'vulkan'; name = 'cuda/0 tiled';   env = @{ NFRUX_ML = 'cuda'; NFRUX_TEST_CUDA_VARIANT = '0' } },
    @{ api = 'vulkan'; name = 'cuda/1 rows';    env = @{ NFRUX_ML = 'cuda'; NFRUX_TEST_CUDA_VARIANT = '1' } },
    @{ api = 'vulkan'; name = 'coopmat';        env = @{ NFRUX_ML = 'coopmat' } },
    @{ api = 'vulkan'; name = 'coopmat_int8';   env = @{ NFRUX_ML = 'coopmat_int8' } },
    @{ api = 'vulkan'; name = 'coopmat_fp16';   env = @{ NFRUX_ML = 'coopmat_fp16' } },
    @{ api = 'vulkan'; name = 'coopmat_bf16';   env = @{ NFRUX_ML = 'coopmat_bf16' } },
    @{ api = 'vulkan'; name = 'dot4 tiled';     env = @{ NFRUX_ML = 'dot4'; FFX_ML_DOT4_TILED = '1' } },
    @{ api = 'vulkan'; name = 'dot4 untiled';   env = @{ NFRUX_ML = 'dot4'; FFX_ML_DOT4_TILED = '0' } },
    @{ api = 'vulkan'; name = 'scalar';         env = @{ NFRUX_ML = 'scalar' }; small = $true }
)
$rows = @()
$i = 0
foreach ($v in $variants) {
    $i++
    Write-Host ("timing {0}/{1}: {2} {3}" -f $i, $variants.Count, $v.api, $v.name)
    foreach ($k in $v.env.Keys) { Set-Item "Env:$k" $v.env[$k] }
    $vs = if ($v.small) { @('1920x1080') } else { $sizes }
    $vg = if ($v.small) { @(1) } else { $gens }
    foreach ($s in $vs) {
        foreach ($g in $vg) {
            $log = "time_$($v.api)_$($v.name -replace '[ /]', '_')_${s}_$g.txt"
            $r = Invoke-Tool 'ngx_harness.exe' @($dll, '--time', "$s@$g", $v.api) $log
            $line = $r.out -split "`n" | Where-Object { $_ -match 'ms \(each' } | Select-Object -First 1
            $total = $null; $cpu = $null; $first = $null
            if ($line -match ':\s+([\d.]+)') { $first = [double]$matches[1] }
            if ($line -match '([\d.]+) ms \(each') { $total = [double]$matches[1] }
            if ($line -match 'calls ([\d.]+) ms CPU') { $cpu = [double]$matches[1] }
            $rows += [pscustomobject]@{ api = $v.api; variant = $v.name; size = $s; mult = "$($g + 1)x"; ms = $total; first = $first; cpu = $cpu;
                                        note = if (-not $line) { 'not available' } elseif ($line -match 'OUTPUT IS THE BACK BUFFER') { 'NO OUTPUT' } else { '' } }
        }
    }
    foreach ($k in $v.env.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
}

# --- The report
$safe = ($gpu.Name -replace '[^A-Za-z0-9]+', '-').Trim('-')
$report = Join-Path $kit "report\nfrux_bench_${safe}_$stamp.md"
$md = @()
$md += "# NFRUX bench: $($gpu.Name)"
$md += ''
$md += "| | |"
$md += "|---|---|"
$md += "| GPU | $($gpu.Name), VEN_$ven DEV_$dev |"
$md += "| Driver | $($gpu.DriverVersion) |"
$md += "| Windows | $os |"
$md += "| NFRUX | $((Get-Item $dll).VersionInfo.ProductVersion) |"
$md += "| Run | $stamp$(if ($Quick) { ', quick' }) |"
$md += ''
$md += '## Correctness'
$md += ''
$md += '| Test | Result | |'
$md += '|---|---|---|'
foreach ($c in $checks) { $md += "| $($c.test) | $($c.result) | $(if ($c.ok) { 'ok' } else { '**FAILED**' }) |" }
$md += ''
$md += '## What NFRUX chose'
$md += ''
$md += '```'
$md += $plan | Select-Object -Unique
$md += '```'
$md += ''
$md += '## Timing (ms a rendered frame; first = the evaluate that runs the network; cpu = NFRUX''s calls)'
$md += ''
$md += '```csv'
$md += 'api,variant,size,mult,ms,first,cpu,note'
foreach ($r in $rows) { $md += "$($r.api),$($r.variant),$($r.size),$($r.mult),$($r.ms),$($r.first),$($r.cpu),$($r.note)" }
$md += '```'
Set-Content -Path $report -Value $md -Encoding UTF8
$failed = @($checks | Where-Object { -not $_.ok }).Count
Write-Host ''
Write-Host "report: $report"
if ($failed) { Write-Host "$failed correctness check(s) FAILED: please send the report and the raw folder $raw" }
