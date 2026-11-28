# Release notes

## Unreleased

- **Faster frames on both APIs**, the same image (every path still gives the same bytes). On an RTX 4070, ms per rendered frame, before / after:
  D3D12 1080p 2x 0.93 / 0.77, 6x 2.67 / 2.40, 4K 2x 1.49 / 1.16, 6x 4.37 / 3.86; Vulkan 1080p 2x 1.03 / 0.92, 6x 2.98 / 2.72,
  4K 2x 1.72 / 1.47, 6x 4.94 / 4.47.
- **What changed**: a faster optical-flow block match and one pass fewer a flow level; independent passes recorded without barriers between
  them (a batch's timesteps run side by side); on D3D12 the first generated frame is written straight into the game's output; the history of
  the colour is written by the composite instead of copied; fewer barriers around the CUDA network on D3D12; cheaper loads in the warp,
  downsample and composite passes.
- **For testers and optimisers**: `NFRUX_TEST_TIMES` now works on D3D12 too, and `NFRUX_TEST_TIMES=2` lists every pass by its place in the
  frame (docs/OPTIMISING.md).

## 0.0.1

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
- **Help wanted**: only the RTX 4070 has been measured. The benchmark kit (`NFRUX_bench_kit_0.0.1.zip`, docs/OPTIMISING.md) checks
  correctness and times every path on your GPU in about ten minutes. Please send the report.

Known limits: no anti-cheat games; 8-bit and RGB10A2 back buffers only.
