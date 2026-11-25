# Writes loader\proxy_exports.h: the exports of the Windows DLLs NFRUX's file can stand in for, forwarded to Windows' own. Needs dumpbin on PATH.
$seen = @{}
$out = "// Generated (tools/make_proxy_exports.ps1): the exports of the DLLs NFRUX's file can stand in for, forwarded to Windows' own.`n"
foreach ($n in 'dbghelp', 'version') {
    dumpbin /exports "C:\Windows\System32\$n.dll" | Select-String '^\s+\d+\s+[0-9A-F]+\s+([0-9A-F]{8}\s+)?(\S+)' | ForEach-Object {
        $x = $_.Matches[0].Groups[2].Value
        if ($x -match '^[A-Za-z_]\w*$' -and -not $seen[$x]) { $seen[$x] = 1; $out += "#pragma comment(linker, `"/export:$x=C:\\Windows\\System32\\$n.$x`")`n" }
    }
}
[IO.File]::WriteAllText("$PSScriptRoot\..\loader\proxy_exports.h", $out)
