# The release body for a version: its section of docs/RELEASE_NOTES.md (a heading naming the version, such as "## 1.0.0" or
# "## v1.0.0 - 2026-12-01", up to the next heading of the same or a higher level), else a short default.
#   tools\ci\release_notes.ps1 -Version 1.0.0 [-Notes docs\RELEASE_NOTES.md] [-Out notes.md]
param(
    [Parameter(Mandatory)][string]$Version,
    [string]$Notes = "$PSScriptRoot\..\..\docs\RELEASE_NOTES.md",
    [string]$Out = ''
)
$ErrorActionPreference = 'Stop'
$body = @()
if (Test-Path $Notes) {
    $lines = [IO.File]::ReadAllLines((Resolve-Path $Notes).Path)
    $v = [regex]::Escape($Version)
    $level = 0
    foreach ($l in $lines) {
        if ($l -match '^(#+)\s') {
            $n = $Matches[1].Length
            if ($level -and $n -le $level) { break }
            if (-not $level -and $l -match "(^|[^\d.])v?$v([^\d.]|\.?$)") { $level = $n; continue }
        }
        if ($level) { $body += $l }
    }
}
$text = ($body -join "`n").Trim()
if (-not $text) {
    $text = "NFRUX $Version.`n`nInstall: see INSTALL.txt in NFRUX_$Version.zip. Checksums: SHA256SUMS.txt."
    Write-Warning "no section for $Version in $Notes; using the default body"
}
if ($Out) { [IO.File]::WriteAllText($Out, $text + "`n") } else { $text }
