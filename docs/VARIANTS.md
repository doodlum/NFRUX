# Variant tracker

Every way NFRUX can run its network, which GPUs use it, and how much we actually know about its speed there. How to run the
benchmark kit, read a report and optimise a variant: [OPTIMISING.md](OPTIMISING.md). Keep this file current: a change that tunes or
measures a variant updates its row in the same commit.

## Status words

| Status | Meaning |
|---|---|
| **tuned** | Its parameters were chosen by measuring on that hardware (evidence linked). |
| **measured** | A kit report from that hardware exists; nobody has tuned for it. |
| **inherited** | Runs with another family's tuning; never measured there. |
| **untested** | Never run on that hardware, not even for correctness. |
| **gap** | Measured, and slower than it should be: see [Open gaps](#open-gaps). |

Correctness is separate from speed: every variant must give the same bytes (DESIGN D2), and a kit report's correctness section is
the evidence that it does on that GPU.

## The variants

What can differ between two runs of the same frame. The ini and the environment can force each; the rung timing (DESIGN D3,
`core/rungs.cpp`) otherwise picks the fastest valid one per GPU, driver, build and resolution class on the first start.

| Axis | Values | Where | Force with |
|---|---|---|---|
| API | D3D12, Vulkan | the game's | (the game) |
| Rung | cuda, coopmat, dot4, scalar | `rhi/rhi_nri.cpp` MakeNetwork | `NFRUX_ML` / ini `Network` |
| CUDA kernels | 0 tiled, 1 rows | `core/ml/cuda_network.cpp` (`kCudaVariants`) | `NFRUX_ML=cuda` + `NFRUX_TEST_CUDA_VARIANT` |
| CUDA instruction | m8n8k16 (sm_75), m16n8k32 (sm_80 up) | MakeCudaPlan, by compute capability | (the GPU) |
| Coopmat type | int8, fp16, bf16 (order per vendor) | `core/ml/vulkan_network.cpp` TuningFor `matOrder` | `NFRUX_ML=coopmat_int8` / `_fp16` / `_bf16` |
| Coopmat tile | M x N x K from the driver; subgroups, subgroup size, preferred M per vendor | TuningFor | (the driver) |
| Vulkan dot4 | tiled (acc, ocb, workgroup, k-split per vendor), untiled | TuningFor `dot4*` | `FFX_ML_DOT4_TILED`, `FFX_ML_DOT4_SUBGROUP` |
| D3D12 dot4 / scalar | per-layer HLSL kernels | `core/ml/d3d12_network.cpp`, `core/ml/*.hlsl` | `NFRUX_ML=dot4` / `scalar` |
| Passes | flow, warp, preprocess, inpaint, post-process (one build for all GPUs) | `core/interpolator.cpp`, `core/shaders`, `core/gpu` | - |

## By GPU family

The family is decided by PCI device ID (`kPriors` in `core/rungs.cpp`); "order" is the default before timing. Evidence: kit
reports in [bench/](bench/).

| Family | API | Order | Rung / variant | Status | Evidence, notes |
|---|---|---|---|---|---|
| NVIDIA RTX 40 (Ada, sm_89) | both | cuda, coopmat, dot4 | cuda/0 tiled | **tuned** | RTX 4070: tile shapes, the 100k-pixel threshold ([bench](bench/nfrux_bench_NVIDIA-GeForce-RTX-4070_20261128-0220.md), RESULTS "CUDA tiled kernels") |
| | | | cuda/1 rows | **tuned** | RTX 4070, earlier rounds ([bench](bench/nfrux_bench_NVIDIA-GeForce-RTX-4070_20261128-0220.md)) |
| | Vulkan | | coopmat fp16 | **tuned** | RTX 4070 (fp16 20% over int8), 4080 per TuningFor's comment ([bench](bench/nfrux_bench_NVIDIA-GeForce-RTX-4070_20261128-0220.md)) |
| | Vulkan | | dot4 tiled | **measured** | RTX 4070 with AMD's tuning (1080p 4x 5.0 -> 3.0 ms; 2.58 on 2026-11-28, [bench](bench/nfrux_bench_NVIDIA-GeForce-RTX-4070_20261128-0220.md)) |
| | D3D12 | | dot4, scalar | **measured** | RTX 4070 ([bench](bench/nfrux_bench_NVIDIA-GeForce-RTX-4070_20261128-0220.md)): since 2026-11-28 a batch's networks run layer by layer, side by side (1080p 6x 5.38 -> 3.57 ms, 4K 6x 7.24 -> 4.98 between the 2026-11-27 and 2026-11-28 reports) |
| | both | | passes | **gap** | Vulkan at 1080p and below: see gap 1. Measured and changed on the RTX 4070 for every GPU (no per-family setting): dispatch groups, the flow match, preprocess in 16 x 16 workgroups, from 3x a batch's timesteps in one dispatch a pass (RESULTS, "Passes, barriers and copies" and "Timesteps in one dispatch", 2026-11-28) |
| NVIDIA RTX 50 (Blackwell, sm_120) | both | cuda, coopmat, dot4 | cuda/0, cuda/1 | **inherited** | sm_80 PTX linked by the driver; tile shapes from sm_89 |
| NVIDIA RTX 30 (Ampere, sm_86) | both | cuda, coopmat, dot4 | cuda/0, cuda/1 | **inherited** | same instructions as Ada |
| NVIDIA RTX 20 (Turing, sm_75) | both | cuda, coopmat, dot4 | cuda/0 (m8n8k16) | **untested** | the only sm_75 instruction path; the timing falls back to cuda/1 if slower |
| NVIDIA GTX 16 / older | Vulkan | dot4, scalar | dot4 tiled | **inherited** | coopmat dropped (driver-emulated, quirk) |
| | D3D12 | dot4, scalar | dot4 | **untested** | |
| AMD RDNA4 (RX 9000) | Vulkan | coopmat, dot4 | coopmat int8 | **inherited** | RDNA3 settings, untuned |
| AMD RDNA3 (RX 7000, 780M/890M) | Vulkan | coopmat, dot4 | coopmat int8 (WMMA wave32 16x16x16) | **untested** | TuningFor says "untuned"; AMDVLK: coopmat ranked below dot4 (quirk) |
| | Vulkan | | dot4 tiled | **inherited** | RDNA2 APU tuning |
| | D3D12 | dot4 | dot4 | **untested** | no matrix path on D3D12 (gap 4) |
| AMD RDNA2 / RDNA1 / GCN (RX 6000/5000, Vega, Steam Deck) | Vulkan | dot4, coopmat | dot4 tiled | **tuned** | RDNA2 APUs (acc 32, ocb 40, 8x8 workgroups, k-split 4); report not yet in bench/ |
| | D3D12 | dot4 | dot4 | **untested** | |
| Intel Arc B (Battlemage) | Vulkan | coopmat, dot4 | coopmat int8 (8xNxK, SIMD16) | **untested** | TuningFor says "untuned" |
| Intel Arc A (Alchemist) | Vulkan | dot4, coopmat | dot4 untiled | **untested** | coopmat below dot4 (quirk); Intel has no tiled-dot4 tuning (gap 3) |
| Intel integrated | Vulkan | dot4, coopmat | dot4 untiled | **untested** | gap 3 |
| any | D3D12 | dot4, scalar | dot4 | see family | the only D3D12 path off NVIDIA |

## Open gaps

Numbered so commits and reports can refer to them. Close one by linking the change and the report that shows it.

1. **Vulkan at 1080p and below (RTX 4070).** Narrowed (RESULTS, "Passes, barriers and copies" and "Timesteps in one dispatch", 2026-11-28),
   still slower than the fastest competing frame generator we measure: 1080p 2x / 6x 0.92 / 2.59 ms (was 1.11 / 2.99) against 0.88 / 2.55;
   720p-class 2x 0.73 against 0.51. GPU time a frame at 1080p 2x is 0.63 ms (was 0.73): network 0.18, optical flow 0.155 (12 dispatches, the level-2 match 0.045),
   composite with the history store 0.086, motion vectors 0.10, preprocess 0.05, output copy 0.027. The 0.15 ms over D3D12 (0.77) at the same
   size is now accounted for: the CUDA kernels and their copies 0.04 more through VK_NVX_binary_import and vkCmdCopyBufferToImage, the output
   copy (D3D12 writes frame 1 into the host's output; a Vulkan output's usage is not known), slower downsample passes, about 0.03 a recording
   in submission, and descriptor-set binding (1.4 us a dispatch of CPU against 0.1 on D3D12).
2. **Per-timestep work at 4x-6x.** Narrowed (RESULTS, "Timesteps in one dispatch", 2026-11-28): from 3x each pass after the flow is one
   dispatch for all timesteps (z = timestep; 60 dispatches -> 12 at 6x), and the D3D12 dot4 / integer networks of a batch run layer by layer
   side by side. 1080p 6x: D3D12 2.40 -> 2.35 ms, Vulkan 2.71 -> 2.59; D3D12 dot4 5.05 -> 3.56. Each timestep still costs its own work in
   every pass (they read the timestep; only a network that shares work across timesteps, i.e. retraining, removes that), the copies of each
   network's two outputs into the textures the composite filters, the composite's UI test (the same for every timestep), and the copy out of
   frames 2-n. Cheaper passes help every multiplier.
3. **Intel dot4.** No tiled-dot4 tuning for Intel (TuningFor uses the untiled kernel); Arc A and integrated GPUs run it first.
4. **No matrix path on D3D12 off NVIDIA.** AMD and Intel D3D12 games run dot4. D3D12 Cooperative Vectors were skipped while preview-only.
5. **CUDA tile shapes tuned on sm_89 only.** Turing, Ampere and Blackwell need reports; the timing protects against a loss, not
   against leaving speed unused.
6. **No reports outside the RTX 4070.** Every **untested** and **inherited** row above.
7. **Vulkan without the loader (OptiScaler routes).** Only Streamline's guarantees are known there (DESIGN, "Vulkan features"): CUDA
   kernels on NVIDIA RTX, and only the 32-bit integer network everywhere else, since cooperative matrices and dot4 need features nobody
   is known to have enabled. Installing the DLL as `dbghelp.dll` / `version.dll` as well would enable them; or OptiScaler could enable them.
