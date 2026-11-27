NFRUX benchmark kit

Checks that NFRUX gives correct results on your GPU and measures how fast each of its ways of running is. Nothing is installed, no
window opens, and nothing outside this folder is written.

1. Close games and anything else using the GPU.
2. In this folder: right-click run_bench.ps1 > Run with PowerShell
   (or in a terminal: powershell -ExecutionPolicy Bypass -File run_bench.ps1, add -Quick for a 3-minute run)
3. Send report\nfrux_bench_<your GPU>_<date>.md (an issue on the NFRUX repository, or to whoever asked you).
   If it says a check FAILED, send the report\raw folder too.

The report holds your GPU's name and ID, its driver version and Windows version, and the measurements. Nothing else.
Licences: licenses\.
