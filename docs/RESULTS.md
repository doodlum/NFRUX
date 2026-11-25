# Results

## A1, D3D12 (2026-11-25, RTX 4070, driver 610.88)

`build\Release\ml_conformance_d3d12.exe`: v14 network, D3D12 compute, against the CPU reference (graph outputs). 64x64: 23 cases (all -128,
all 127, alternating, 20 random); 744x464 and 960x544: 5 cases each; packed dot product (SM 6.4) and integer (SM 6.0): 0 differing bytes.

Not yet covered of A1: Vulkan rungs, CUDA, per-layer dumps, recorded tensors, batches, 20 seeds at the larger sizes. Whether the D3D12 debug
layer was active was not checked. Timings printed while another process trained on the GPU are not recorded as cost.

## D1 (2026-11-25)

`tools/build_shaders.py`: 70 variants build to SPIR-V (spirv-val, NoContraction on float arithmetic, no 8/16-bit types or group operations) and
DXIL. Float literals rewritten from `HF`: values listed and checked against the originals' set.
## A1, Vulkan (2026-11-25, same machine)

`ml_conformance_vulkan --iterations 30`, 744x464: cooperative matrix int8 (default), fp16, bf16, dot4, scalar: all 0 of 690432 bytes differ on
its random input. Times, GPU shared with training: coopmat int8 0.37 ms, fp16 0.30, dot4 0.70, scalar 1.6 (D3D12 dot4 0.98, int32 1.76).
Worst-case inputs not yet run on Vulkan; fp16 stays off by default until the sum |w| bound of D2 is checked.

## M0a loader probe (2026-11-25, Streamline 2.14.16, driver 610.88, Streamline_Sample built from source + scripting patch)

`loader/probe.cpp` as `dbghelp.dll` beside the sample (sl.interposer imports it): IAT patch of GetProcAddress / LoadLibraryExW in the exe, sl.interposer and every non-Windows module
loaded later (LdrRegisterDllNotification); modules already loaded otherwise were not patched. NGX core loaded from the DriverStore; exports of interest wrapped; no system setting changed.
Runner: `C:\dev\nfrux-ref\run_sample.ps1` (gives the window focus; DLSS-G does not interpolate unfocused).

| API | asked | `numFramesActuallyPresented` | stub |
|---|---|---|---|
| D3D12 | 1 / 3 / 5 generated | 2 / 4 / 6 | copy of the back buffer, COMMON in and out, accepted |
| Vulkan | 1 / 3 / 5 generated | 2 / 4 / 6 | copy, GENERAL |

Runs were 500-900 frames (a few seconds each), not the 60 s the gate names, and without debug layer / validation: still to do.
Found:
- Streamline loads its over-the-air plugins from C:\ProgramData\NVIDIA\NGX\models (not the sample's folder) when newer or equal.
- The multi-frame limit is the NGX value `DLSSG.MultiFrameCountMax` (GetCapabilityParameters): 1 on RTX 40, 5 once the loader sets it
  (Streamline's own limit is 5).
- Evaluate: command list type COMPUTE (2); MultiFrameIndex 1..n in order; no DLSSG.FrameID; back buffer and outputs BGRA8 (87 / VK 44), UAV
  allowed; depth VK_FORMAT_D24_UNORM_S8_UINT; UIAlpha R16F; MvecScale 1; CmdQueue, CmdAlloc, IsRecording = 1 on **both** APIs.
- Vulkan init arrives as VULKAN_Init_Ext2 with null GetInstance/DeviceProcAddr (take vulkan-1.dll's). vkCreateDevice is asked through GetProcAddress.
- Streamline creates Vulkan devices with native optical flow ("Native VK OFA feature supported"); with a queue and pool in Evaluate, D4's
  "infeasible on Vulkan" has to be re-examined.
- D3D12: inbox runtime tops out at SM 6.8 (6.9 needs the host's Agility SDK); DirectML 1.15.6 inbox; driver metacommands include Convolution,
  GEMM and QuantizedGEMM. D3D12 debug layer is not installed (so A1 on D3D12 ran without it).
## A2, first results (2026-11-25, RTX 4070, synthetic pan + moving square, RGBA8, `tests/parity.cpp`)

D3D12 (rhi_d3d12, SM 6.4 dot4 network) against Vulkan (rhi_nri on NRI v192, int8 cooperative matrix): **every generated frame, and 13 intermediates of the first
generated frame of the last rendered one, compared byte for byte, 0 differ**, at 1280x720 (2x, 3x, 5x, 6x), and 4x at 1920x1080, 1920x1080 from 1280x720, 1280x800 from
742x464, 2560x1440 from 1707x960, 1366x767. Mean PSNR against the true in-between frames 43-45 dB (the same on both, by construction).

Faults the comparison found, all fixed in the shared source or backend:
- `findMSB(0)`: -1 in GLSL, unsigned ~0 after SPIRV-Cross (zero-flow texels got exponent 15 on D3D12);
- SPIRV-Cross emits the 32-bit `InterlockedMin` for 64-bit raw buffer atomics (D3D12's 64-bit scatter was wrong; `InterlockedMin64` now);
- Vulkan: an image's first layout transition needs earlier work as its source, else the first interpolated frame was corrupted at 5x and
  more (not reported by synchronization validation, whose activation was not proven; mechanism not established);
- `log` / `exp` written as `log2` / `exp2`.

Not covered by this run (some later, below): NRI on D3D12, HDR formats, BGRA8, inverted depth, the input tensor, later generated frames' intermediates, UI inputs, real captures, 4K, other GPUs, the
32-bit scatter (checked at 1280x720 before the last fixes), validation layers on D3D12.
## A4 / A5, first runs with the real DLL (2026-11-25, Streamline_Sample 1280x701, RTX 4070)

`loader/loader.cpp` (dbghelp.dll) + `NFRUX\nvngx_dlssg.dll`; scripted camera and fixed timestep (NFRUX_SCRIPT, patch in the sample's Animate).

| API | network | asked | presented |
|---|---|---|---|
| D3D12 (rhi_d3d12) | SM 6.4 dot4 | 2x, 4x, 6x | 2x (392 fps), 4x (567-574), 6x (642; 2990 of 3000 evaluations generated) |
| Vulkan (rhi_nri) | int8 cooperative matrix, features added by the loader's vkCreateDevice hook | 4x | 4x (557-562 fps) |
| Vulkan, floor (no hook yet) | 32-bit integer | 2x, 4x | 2x (296), 4x (318) |

Runs of 500-700 frames; no crash. Not the 5-minute, validation or 6x-Vulkan-on-tensor-cores runs.

Same script on both APIs, frames paired by the rendered image (`tests/compare_sample.py`): the rendered (HUD-less) inputs of the two
APIs match at 31.4 dB (the sample's renderers differ: ray-traced shadows, foliage, reflections), the generated frames outside the UI at
34.6-34.8 dB, so the interpolation adds no difference of its own; visually matching. A5's bar of 35 dB is missed by 0.2-0.4 dB, with the
"no more than 1 dB below the inputs" part met. The cross-replay part of A5: see the end of this file.
## A2 formats and 4K (2026-11-25, after the NRI teardown fix)

NRI built without Agility SDK, NVAPI, AGS, NVTX (its wrapped-device teardown crashed with them on). D3D12 (NRI) against Vulkan (NRI), 4x, 5 frames: RGB10A2, BGRA8, inverted depth, 4K from 1440p, 4K from 1080p RGB10A2: 0 bytes differ. RGBA16F scRGB (+ inverted, 1080p from 720p): PSNR >= 116 dB over the raw bytes, no byte off by more than 1.
The native D3D12 backend was bit-identical to NRI's and no faster in the sample (564 against 572 fps at 4x): removed, NRI is the only backend.
4K flow still runs at display size (the 01 reduction pass of the design is not in).

## Corrections after the revision 4 review (2026-11-25)

The two wordings above were narrowed (they claimed more than was compared or patched). Fixed in code: NRI took features from the GPU instead of the host (now unknownFeatures, API 1.2); completion token ordering (D3D12 MARKER_OUT, Vulkan full barrier + fill + host barrier); host image access masks on Vulkan; D3D12 buffer decay; slot reuse; shared device state; use after free on Release; int8 forced for Auto; unknown NFRUX_ML values refused. Parity after these: 0 bytes differ (6x 720p, BGRA8 3x), FP16 4x 120 dB.
The validation claim earlier (no synchronization errors) is withdrawn: the layer's activation was never proven.

## A4, long runs (2026-11-25; DLL at commit e07fecf, before the revision 5 fixes; release Streamline DLLs, scripted camera, 1280x701)

| API | multiplier | seconds | evaluations | generated | presented multiplier in every log line after the first |
|---|---|---|---|---|---|
| D3D12 | 2x | 275 | 51,000 | 50,998 | 2 (433 of 433) |
| D3D12 | 6x | 290 | 147,000 | 146,990 | 6 (250 of 250) |
| Vulkan | 2x | 271 | 51,000 | 50,998 | 2 (433 of 433) |
| Vulkan | 6x | 292 | 147,000 | 146,990 | 6 (250 of 250) |

No crash. They are 271-292 s, not the 300 s the row asks for; other processes used the machine meanwhile. Not done: validation runs, the
between-neighbours check, the current DLL for the full time.

Current build (after the fixes), 900 frames each: D3D12 2x 384 fps, 6x 623; Vulkan (cooperative matrix) 2x 390, 6x 629.
A loader regression was caught by these (the loader patched its own imports: stack overflow at start), fixed.
## M0b CUDA: go (2026-11-25)

`tests/cuda_spike.cpp`: PTX linked by nvcuda.dll (no toolkit), launched by NVAPI in a D3D12 compute list; chained kernels with UAV barriers 0 of 4096
wrong, `mma.sync` m16n8k16 s8 product 0 of 128 wrong. `core/ml/cuda_network.cpp`: the whole network, one generated kernel per layer, exact against
the CPU reference (64x64: 23 cases, 744x464 and 960x544: 5 each, 0 bytes). sm_80 and newer (RTX 30+); older NVIDIA GPUs fall to dot4. Creation
about 0.7 s the first time at a size, under 0.05 s from the cubin cache (LOCALAPPDATA\NFRUX) - still inside Evaluate (worker thread owed).

Time a network, 40 back to back, idle RTX 4070: 744x464 CUDA 0.155 ms, dot4 0.42, int32 0.85; 960x544 CUDA 0.194, dot4 0.58-0.63, int32 1.02
(Vulkan int8 cooperative matrix 0.37 at 744x464). Earlier single-shot figures here were affected by idle clocks.

In the sample at 1280x701 (900 frames, fps presented): 2x 384 -> 462, 4x 572 -> 736, 6x 623 -> 889. Parity with Vulkan unchanged (0 bytes; 6x 720p and
4x 1080p from 720p RGB10A2). Crash markers (core/marker.cpp) guard the rung; their failure path has not been exercised.
## A6 OptiScaler (2026-11-25, Streamline_Sample copy, no NFRUX loader)

| OptiScaler | route | API | result |
|---|---|---|---|
| 0.9.5 release | `dlssg_to_fsr3_amd_is_better.dll` slot (FGInput / FGOutput=nukems) | D3D12 | 700 frames, 2x presented, CUDA network, 456 fps |
| 0.9.5 release | same | Vulkan | 700 frames, 2x presented: 304 fps on the integer network (features unknown), 420 fps with NFRUX_VK_ASSUME_FEATURES=1 (cooperative matrix) |
| a build with FGNvngxReplacement | `OptiScaler\custom\NFRUX\dlssg\nvngx_dlssg.dll`, FGNvngxReplacement=NFRUX | Vulkan | 900 frames, 2x and 4x presented |
| same | same | D3D12 | loads NFRUX (5 generated frames at most), then the sample crashes in ntdll; it crashes the same way with FGInput=nofg, so without NFRUX in use: OptiScaler's build and this sample |

Found: OptiScaler hands back its own handle object with NFRUX's id: features are now looked up by id.
Not done: 1000 evaluates as A6 asks, the custom route on D3D12, multipliers above 2x on 0.9.5, an OptiScaler master build.

## A5 cross-replay: the sample's real inputs through both APIs (2026-11-25, RTX 4070)

Recorded on each API (`NFRUX_DUMP=300`, scripted camera): HUD-less colour, depth, motion vectors, matrices, 1280x701,
frames 2-7 of each recording (frame 1 was skipped by the sample's frame number, frame 0 unused). Each recording is then
played through D3D12 and Vulkan (`parity --first 2 --replay DIR --frames 6 --gen N`).

- First run: **not** identical (78 dB, 0.06% of bytes off, at most 0.00004% by more than 1). The synthetic scenes hid it:
  their camera matrices are too simple. First differing pass: the motion-vector hints. The cause is in the class "opaque
  products" (matrix * vector, dot, mix, length): neither `precise` nor NoContraction fixes how a driver fuses inside them, and
  the two APIs' compilers differ. (The hints pass was fixed and everything else in the class at the same time, so which
  single operation caused each difference was not isolated.)
- Fix: all of them written out, and `build_shaders.py` now refuses a pass that contains one.
- After: **0 bytes different** in every intermediate and every generated frame, both recordings, 2x, 4x and 6x (5, 15, 25
  generated frames each). The synthetic cases still are too (1080p, 4K, inverted depth, RGB10A2; 2x and 6x).

Not covered: HDR recordings, render size below display size from a real recording, other GPUs.

## Strict and relaxed math (decision, 2026-11-25)

"Slight differences are ok if the compiler is trying to optimise the code"; "prove exactness, then drop for when actually using the mod".
CMake option `NFRUX_STRICT_MATH` (OFF, the shipped build; ON, the proof build: NoContraction on every float result, DXC `-Gis`).

| RTX 4070, one generated frame | strict | relaxed |
|---|---|---|
| D3D12 1080p, ms a rendered frame | 0.80 / 0.80 | 0.80 / 0.80 |
| D3D12 4K from 1080p | 1.63 / 1.62 | 1.62 / 1.57 |
| Vulkan 1080p | 1.09 / 1.07 | 1.08 / 1.04 |
| Vulkan 4K from 1080p | 1.98 / 1.98 | 1.99 / 1.93 |
| D3D12 against Vulkan, sample recording (2x, 6x) | 0 bytes | 76.8 dB, 0.00009% of bytes off by more than 1, none by more than 3 |

(16 frames a submission, recording time taken off, median of 30, two runs.) Strict costs 0 to about 2-3%, which is near the runs' own spread.

## A9, A10: NGX harness (`tests/ngx_harness.cpp`, the real DLL, its NGX exports, 2026-11-25)

53 checks an API, all pass on D3D12 and Vulkan, strict and relaxed: HUD-less difference test, UI alpha, UI layer, both without HUD-less, no-warp mask,
colour subrects, HUD-less in RGB10A2 / BGRA8, outputs BGRA8 / RGB10A2 / RGBA16F, the scRGB difference test, distortion fields (zero, shifting; 2x and
5x), menu detection, reset with HUD-less, NotRendering, missing inputs, indices out of order and past five, changes of count / format / size, two
features, release, release twice, evaluate after release, create again. Between the APIs: strict 49 of 49 outputs byte-identical; relaxed 40 of 49,
the rest 103 dB or better. Depth and motion subrects: only checked to evaluate. PQ UI values are not checked separately.

- Soak, 2000 evaluates each API: video memory and handles flat.
- Minimal Vulkan 1.2 host (only shaderStorageImageExtendedFormats): all checks, byte-identical to D3D12, no validation message.
- Found and fixed by it: members outliving their device (crash when a second feature was released); the network kernels were SPIR-V 1.6 (not valid on
  a 1.2 device).

## Validation (A4's part)

- Harness, Khronos validation with synchronisation validation: 0 messages (host on 1.3, and the minimal 1.2 one). The layer is known active: it
  reported the SPIR-V 1.6 kernels before their fix.
- Streamline_Sample on Vulkan, 4x 3000 and 6x 5000 rendered frames, with a control run without frame generation: one message concerned NFRUX (an image
  view destroyed while Streamline's command buffer formally counted as pending). Fixed (kept 16 recordings longer), gone in the second run. All
  others are the sample's or Streamline's; none names a NFRUX object or a dispatch.
- D3D12 debug layer: **not run.** Graphics Tools is not installed (elevated, decision). Agility SDK packages 1.614-1.620 were tried
  app-local, also by device factory: the core loads (1.618 on) but the runtime refuses its layers (DXGI_ERROR_SDK_COMPONENT_MISSING). NRI's own
  validation on D3D12: 0 messages. (On Vulkan NRI's validation objects to wrapped host images and ends the run; not used there.)

## A4, long runs with the current code (2026-11-25, 1280x701, release Streamline DLLs, worker-thread creation not yet in)

| API | multiplier | seconds | generated |
|---|---|---|---|
| D3D12 | 2x | 339 | 66,000 of 66,000 logged |
| D3D12 | 6x | 323 | 221,990 of 222,000 |
| Vulkan | 2x | 287 | 56,998 of 57,000 |
| Vulkan | 6x | 266 | 155,990 of 156,000 |

Exit code 0 each. The Vulkan runs are short of 300 s; builds and tests used the machine meanwhile.

## A7, D7: rungs, self-check, crash marker

- Every rung forced (D3D12 CUDA, dot4, int32; Vulkan cooperative matrix, dot4, scalar), each with 64-bit and 32-bit scatter: 12 of 12 harness runs pass.
- Self-check: passes; with `NFRUX_TEST_BAD_CHECK=1 / 2 / 3` it goes to dot4, to int32, to no frame generation.
- Crash marker: the process killed twice while the marker was set; the third start used dot4 and said so; deleting the file brought CUDA back.

## A6, 0.9.5 again with the current DLL

D3D12 2x: 3500 frames, 2990 of 3000 generated (439 fps). Vulkan 2x: 2874 of 3000 (315 fps, integer network: the host's features are unknown there;
the first frames are the worker thread's). Above 2x: that slot offers 2x only (Streamline refuses 4x); the sample then ends with an access violation,
on Vulkan without NFRUX having been loaded. No newer OptiScaler release exists.

## M0d

- SM 6.9 cooperative vectors: dropped. The inbox runtime reports SM 6.8; the retail Agility SDK (1.620.2) has SM 6.9 and 6.10 but no
  cooperative-vector interface in d3d12.h, and NFRUX cannot choose the host's D3D12 core. On NVIDIA GPUs of those generations CUDA is the rung.
- DirectML: see the table at the end (measured on an idle GPU).
