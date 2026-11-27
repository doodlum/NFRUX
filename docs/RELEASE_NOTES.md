# Release notes

## 1.0.0

The first release. Tuned for NVIDIA RTX 40; every other GPU runs through NFRUX's general paths (README, "Status").

- **Frame generation in place of DLSS-G**: 2x to 6x on D3D12 and Vulkan, through OptiScaler (`FGNvngxReplacement=NFRUX`, or the 0.9.5
  slot) or directly in DLSS-G games as `dbghelp.dll` / `version.dll`.
- **Network**: NFRU v15, an int8 network fine-tuned from Arm's NFRU v2.
- **Fastest path per GPU**: on its first start NFRUX times every way it can run on the GPU (CUDA kernels, Vulkan cooperative matrices,
  integer dot products, 32-bit integers) and keeps the fastest. Every path gives the same image.
- **RTX 40**: tiled CUDA kernels, tuned on an RTX 4070, on D3D12 and Vulkan. RTX 20/30/50 run the same kernels with RTX 40 settings,
  timed against the previous kernels.
- **Vulkan**: NFRUX uses only features that are enabled on the game's device. On OptiScaler's routes that means CUDA kernels on RTX, and
  the 32-bit integer network on other GPUs.
- **Cost on an RTX 4070**, ms per rendered frame, D3D12: 0.93 at 1080p 2x, 2.67 at 1080p 6x, 1.47 at 4K 2x, 4.38 at 4K 6x.
- **Help wanted**: only the RTX 4070 has been measured. The benchmark kit (`NFRUX_bench_kit_1.0.0.zip`, docs/OPTIMISING.md) checks
  correctness and times every path on your GPU in about ten minutes. Please send the report.

Known limits: no anti-cheat games; 8-bit and RGB10A2 back buffers only.
