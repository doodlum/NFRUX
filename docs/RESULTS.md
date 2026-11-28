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
Worst-case inputs and the 2^24 bound: see "Revision 8".

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

`experiments/cuda_launch.cpp`: PTX linked by nvcuda.dll (no toolkit), launched by NVAPI in a D3D12 compute list; chained kernels with UAV barriers 0 of 4096
wrong, `mma.sync` m16n8k16 s8 product 0 of 128 wrong. `core/ml/cuda_network.cpp`: the whole network, one generated kernel per layer, exact against
the CPU reference (64x64: 23 cases, 744x464 and 960x544: 5 each, 0 bytes). sm_80 and newer (RTX 30+) at the time; superseded: m8n8k16 runs on sm_75 (RTX 20) as well (DESIGN D3). Creation
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

- Soak, 2000 evaluates each API: video memory and handles flat (video memory on D3D12 only here; on Vulkan it was not measured: see the final evidence).
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
- DirectML integer convolution (idle RTX 4070, 372x232x16, 3x3, 16 out): exact (0 of 1,380,864 sums wrong) in four layouts; 198-204 us, 45 us in
  the best (NHWC, weights owned by DirectML), for one layer's sums without our rescale. The whole 22-layer network is 445 us on dot4 and 173 us
  on CUDA. Dropped.

## The reference is the nfru-mfg branch (2026-11-25): audit and what it found

Every shared header and shader was diffed against `ngsdk-steamdeck` `nfru-mfg` (38cfb23). Found and fixed:

| Divergence | Effect |
|---|---|
| The composite lacked the timestep prior the network is trained with | up to 1.4 dB lost at t = 1/6 and 5/6 |
| Above 1080 rows the flow ran at full size (pass 01 missing) | 4K 4x: 3.2 -> 2.65 ms on D3D12 |
| The flow hint had the game's vectors scattered over the camera motion | no gain (+-0.03 dB); removed with its 64-bit atomics |
| Reduced lanes RGBA8 for RGB10A2, RGBA16F for R11G11B10F | now the colour's own format |
| Jittered motion vectors not handled | `DLSSG.MvecJittered` through the SDK's jitter cancellation |

Earlier parity results in this file (D3D12 against Vulkan) were unaffected by these: both APIs ran the same wrong composite. That is what A3 is for.

## A3, A14: quality against true frames (final code, D3D12, shipped build; `tests/quality_reference.py`, 2026-11-25)

Held-out sets test_new + test_bal (35 sequences, 4 pairs each where they exist), PSNR in dB, mean over generated frames; 8-bit colour and
half-float vectors for every method, outputs rounded to 8 bits.

| | frames | NFRUX | PyTorch, same network | Arm NFRU v2 weights in the PyTorch pipeline, with the same timestep prior | linear blend |
|---|---|---|---|---|---|
| 2x | 68 | 35.72 | 35.71 | 35.24 | 26.46 |
| 3x | 144 | 37.35 | 37.39 | 36.92 | 30.40 |
| 4x | 96 | 39.16 | 39.17 | 38.80 | 29.94 |
| 5x | 144 | 34.13 | 34.11 | 33.71 | 25.85 |
| 6x | 180 | 34.13 | 34.10 | 33.56 | 25.57 |

A3 met (within 0.04 dB; limit 0.2). A14 met (above Arm's v2 weights, given the same prior, at every multiplier). Before the prior fix NFRUX was 33.61 at 6x.

## A12: optical flow hardware (offline, VK_NV_optical_flow, RTX 4070)

70 held-out pairs, same network and pipeline (PyTorch), only the flow differs, dB: block match / hardware (slow level) / no flow:
3x 40.26 / 39.61 / 39.07, 4x 40.17 / 40.22 / 39.44, 5x 34.93 / 34.82 / 33.71, 6x 34.23 / 33.89 / 33.04. Medium level within 0.06 of slow.
(Here the block match is the gym's, without hints; no hints were given to the hardware either.)
Cost, idle: hardware 0.85 / 0.51 / 0.44 ms at 720p and 1.77 / 0.98 / 0.68 ms at 1080p (slow / medium / fast; wall clock, 8 back to back, its own
engine); the block match 0.12-0.16 ms at 720p, 0.19-0.20 at 1080p, 0.19-0.21 at 4K of shader time (with and without it). Not delivered.

## A8: cost (idle RTX 4070, shipped build, RGBA8; D3D12 CUDA / Vulkan cooperative matrix, ms a rendered frame for all generated frames)

| display (render) | 2x | 4x | 6x |
|---|---|---|---|
| 1080p | 0.78 / 1.06 | 1.76 / 2.31 | 2.69 / 3.59 |
| 1440p (1706x960) | 0.87 / 1.13 | 1.96 / 2.57 | 3.04 / 4.02 |
| 4K (1080p) | 1.16 / 1.48 | 2.65 / 3.29 | 4.15 / 5.14 |

Default rung at 1080p 2x: under the 2.5 ms limit. Method: 16 frames a submission, recording time removed, median of 30, not timestamps. VRAM and
the other rungs are not in this table.

## Sample with the final code (4000 rendered frames each, 1280x701, worker-thread creation)

Presented fps near the end: D3D12 2x 429, 4x 675, 6x 804; Vulkan 2x 385, 4x 544 and 553, 6x 651. One earlier Vulkan 4x run had dropped to
multiplier 1 at its end; the cause is not shown (DLSS-G stops without window focus); two reruns stayed at 4x in all 33 log lines. The 5-minute runs
above predate the audit's changes.

Sizes beyond 4K and odd ones, strict, D3D12 against Vulkan: 5120x1440, 3440x1440, 7680x4320, 800x600 from 400x300, 1281x701 from 641x351: 0 bytes
different.

## A11

nvngx_dlssg.dll links no graphics DLL directly (d3d12 and dxgi are delay-loaded; vulkan-1 is loaded by name); no shared-handle call in the sources or
imports. It needs the VC++ 2015-2022 runtime.

# Final evidence (revision 6)

Commit 80c9418 unless said. Strict: `build-strict` (NFRUX_STRICT_MATH=ON). Shipped: the default build. RTX 4070, driver 610.88, 2026-11-25.

| Row | Result |
|---|---|
| A1 | D3D12 int32, dot4 and CUDA exact at 64x64 (23 cases), 744x464 and 960x544 (5 each); Vulkan cooperative matrix (int8; fp16 and bf16 too), dot4 and scalar exact at its sizes (SPIR-V 1.5 kernels) |
| A2a | Strict, 30 cases (10 sizes / formats x 2x, 4x, 6x; 8 frames; RGBA16F included): 0 bytes in every output and intermediate, exit 0 |
| A2b | Shipped, the same 30: worst 86.3 dB, exit 0. The limits caught a real one: above 1080 rows pass 01 rounds from an exact half and fused math picked another texel (63 dB, 0.013% over 3 codes); the reduction passes are strict in every build since |
| A3, A14 | Met: table "A3, A14 final" at the end |
| A5 | 64 recorded frames an API (66 dumped, frame 1 skipped by the sample's counter), HUD-less 1280x701, DLSS off: strict 0 bytes at 2x, 4x, 6x on both recordings; shipped at least 75.5 dB, at most 0.0002% over 1 code; no drift (the worst frame is not late: the limit holds in every frame) |
| A7 | Six rungs forced: harness 61 of 61 each. Self-check faults and crash marker as earlier |
| A9, A10 | Harness 124 checks (61 an API and 2 between): strict 55 of 55 outputs byte-identical; shipped all within 96 dB. New: sRGB images (RGBA8 and BGRA8; stored values equal to UNORM's), jittered vectors, release while another feature evaluates, device destroyed and made again (D3D12: 0 references left), both APIs at once on two threads. Khronos validation + synchronisation: 0 messages (1.3 host and minimal 1.2 host), through device destruction. Soak 2000 an API: video memory (DXGI, the process's, read on both) and handles flat |
| A11 | as above; loaded modules at run time in each mode not recorded |
| Not covered | A4 as written (300 s at 1920x1080 with focus kept); D3D12 debug layer; the reduction passes against an outside reference; PQ UI values; nvapi / nvcuda absent; other GPUs |

A4 on the final DLL (293ca58): D3D12 2x 239 s, x2 in 400 of 400 log lines, 44,975 of 45,000 logged evaluations generated; D3D12 6x 284 s, x6 in 300 of 300, 176,865
of 177,000. Vulkan 2x 225 s and 6x 165 s, exit 0, but the window lost focus part-way (Streamline's log: "DLSS-G disabled: window not focused"; another
user of the machine) and it presented 1x from there: 20,983 of 21,000 and 8,890 of 9,000 generated before. The earlier Vulkan drop to 1x is the same effect.
The runs are under 300 s, and the window was clamped to 1284x701 by the desktop although 1920x1080 was asked.
With the new loader NGX Shutdown reaches NFRUX on both APIs (loader log).

## Low frame rates under a Reflex cap (request; eedcb52, sample in real time, 2026-11-25)

The sample's Reflex cap limits presented frames. Scripted camera at real speed (it was one step a frame before, which tied the simulation to the frame
rate: a fixed step is now used only while recording with NFRUX_DUMP).

| | 2x cap 40 (base 20) | 2x cap 60 (30) | 2x cap 120 (60) | 4x cap 120 (30) | 6x cap 180 (30) |
|---|---|---|---|---|---|
| D3D12 | 40.0 | 60.0 | 120.0 | 120.0 | 180.0 |
| Vulkan | 40.0 | 60.0 | 120.0 | 120.0 | **cap not kept (612-649)** |

The multiplier held in every log line. D3D12 keeps the cap at every multiplier with the slowest network too (scalar, 4x and 6x). Vulkan loses it at 5x, at
6x, at 4x with the scalar network and at 4x with a 240 cap, and keeps it at 6x when NFRUX only copies: it depends on the time generation and presenting
take. NFRUX calls no Reflex or present function. The validation layer reports (both directions, earlier log) that Streamline's pacer presents the swapchain on one
thread while vkLatencySleepNV uses it on another; that race is the likely cause, not a shown one. Open.

A7 again after the marker change: two kills, third start on dot4 (file name with the driver's time), cleared, CUDA back; a failed self-check keeps the marker.
Harness at eedcb52 (strict and shipped): 124 checks, 0 failed; strict 55 of 55 identical.

## A3, A14 final

Shipped build, D3D12, `tests/quality_reference.py` run on the binary of 293ca58 (later commits change no shader this path runs: 720p from 540p uses no reduction
pass, and no pass, constant or kernel else was edited). 35 sequences of test_new + test_bal, up to 4 pairs each, PSNR dB, 8-bit colour and half-float
vectors for every method; network outputs rounded to 8 bits (the blend is not).

| | frames | NFRUX | PyTorch, same network | Arm v2 as released | Arm v2 with the timestep prior | blend |
|---|---|---|---|---|---|---|
| 2x | 68 | 35.72 | 35.71 | 35.24 | 35.24 | 26.46 |
| 3x | 144 | 37.35 | 37.39 | 36.81 | 36.92 | 30.40 |
| 4x | 96 | 39.16 | 39.17 | 38.71 | 38.80 | 29.94 |
| 5x | 144 | 34.13 | 34.11 | 33.53 | 33.71 | 25.85 |
| 6x | 180 | 34.13 | 34.10 | 33.33 | 33.56 | 25.57 |

A3: within 0.04 dB at every multiplier (limit 0.2); the worst single frame against PyTorch is -0.50 / -1.93 / -0.32 / -0.29 / -0.50 dB (2x-6x), which frame the
3x one is was not looked at (frames under 45 dB in PyTorch average +0.02 there). A14: above Arm's, both ways, at every multiplier. The Vulkan repeat (test_new only, one pair a
sequence, same binary; not comparable with the table's sets):

```n2x {'nfrux': 38.41, 'torch': 38.45, 'arm+prior': 37.78, 'arm': 37.78, 'blend': 27.13} n 6
3x {'nfrux': 37.3, 'torch': 37.51, 'arm+prior': 36.97, 'arm': 36.78, 'blend': 30.52} n 12
4x {'nfrux': 41.06, 'torch': 41.14, 'arm+prior': 40.75, 'arm': 40.67, 'blend': 27.84} n 9
5x {'nfrux': 32.01, 'torch': 32.0, 'arm+prior': 32.41, 'arm': 32.06, 'blend': 21.31} n 12
6x {'nfrux': 37.07, 'torch': 37.03, 'arm+prior': 36.39, 'arm': 36.02, 'blend': 29.2} n 15
```

# After revision 6 (ba87aca; 2026-11-25)

- One DLL, 4.1 MB; links KERNEL32 and USER32 (d3d12 and dxgi delay-loaded). Streamline_Sample with only `dbghelp.dll` added: D3D12 4x and Vulkan 4x presented. OptiScaler 0.9.5 slot:
  D3D12 and Vulkan 2x. `version.dll`: "Revision 8"; `winmm.dll`: the sample did not start, so it is not a role any more.
- CUDA m8n8k16 (sm_75): exact at all three sizes; 0.153 ms at 744x464 and 0.196 at 960x544 (m16n8k16 0.156-0.172, 0.208-0.212). The same PTX links for sm_75, 80, 86, 89, 90, 100, 120 (the driver's linker); it
  was run on the RTX 4070 only.
- Cooperative matrices int8, fp16, bf16 forced: harness 61 of 61 each, self-check passed.
- Other devices: WARP (D3D12, no vendor): dot4 chosen, 61 of 61. Mesa lavapipe (Vulkan on the CPU): 61 of 61 (the minimal-host repeat was not run to its end).
- HDR10: PQ UI scenes pass; a moving object over a 200-1000 nit background 46.7 dB of PQ signal. scRGB was 38.0 dB on the same scene (the 8-bit stages clip at 80 nits); support
  for it was removed by decision.
- The 3x frame 1.93 dB below PyTorch: race_spaceship_002, a near-static clip; the three worst 3x frames are its, at 54.4-54.8 dB against 56.0-56.7 (under half an 8-bit code). Medians are 0.
- Harness now: 124 checks (61 an API, 2 between), 53 outputs; strict 53 of 53 identical; shipped 0 failed.

# Revision 8 (5a2b8c8; RTX 4070, driver 610.88, 2026-11-25; shipped = `build-ship`, strict = `build-strict`)

First table: 5a2b8c8. The second, on 2ac5a8e, replaces its A1, A2, A7, A9 and A10 rows.

| Row | Result |
|---|---|
| A1 Vulkan | 744x464, five inputs (random, all -128, all 127, alternating, signs of the first layer's weights) x int8, fp16, bf16, dot4 (two forms), scalar: 0 of 690,432 bytes differ in each. Largest partial sum any channel can reach: 7,021,312 (limit 16,777,216) |
| A1 D3D12 | int32, dot4, CUDA exact at 64x64 (23 cases), 744x464 and 960x544 (5 each). Tensor cores by name: 15 names right (T4 and RTX yes; GTX 16, T400, T400 4GB, T600, T1000, T1200, MX450 no) |
| Network times, 744x464, back to back | Vulkan fp16 0.26 ms, bf16 0.28, int8 0.33, dot4 0.68, scalar 1.4; D3D12 CUDA 0.22, dot4 0.67, int32 1.25 (this run's clocks). fp16 is first on NVIDIA since |
| A2a, A2b | 5 sizes / formats x 2x, 4x, 6x: strict 0 bytes; shipped worst 86.3 dB |
| A5 | 64 recorded frames an API, 2x and 6x: strict 0 bytes; shipped 75.5 dB or better, at most 0.0002% over 1 code |
| A9, A10 | 144 checks (71 an API, 2 between), 0 failed, shipped and strict. Strict 59 of 59 outputs identical; shipped worst 96.3 dB, none over 1 code. Inputs in NON_PIXEL_SHADER_RESOURCE and outputs in UNORDERED_ACCESS on D3D12; on Vulkan GENERAL, and read-only inputs in SHADER_READ_ONLY_OPTIMAL (minimal host). New scenes: output subrects, two viewports on one back buffer (58.1 dB), the disable word, frame ids, inverted axes, camera motion excluded (57.4 dB), dynamic resolution (57.6 dB), an internal size (57.5 dB) |
| Validation | Khronos + synchronisation, 1.3 host and minimal 1.2 host: 0 messages |
| A7 | coopmat_int8 forced 71 of 71; WARP (dot4) 71 of 71. Soak 400 evaluates an API: flat |
| A15 | Package: the DLL (4.0 MB), INSTALL.txt, THIRD_PARTY.md, licenses/ (Apache-2.0, Arm model license, D3D12MA, VMA, NRI, Neural Graphics SDK, NFRUX). Search of the repository for the source project's name: none. Sample at 1920x1080 with only dbghelp.dll, then only version.dll: 4x on D3D12 (519-536, 550-558 fps) and Vulkan (442-458, 474-480 fps); removing the file removes it |
| A6 a | OptiScaler 0.9.5, 1500 rendered frames: 2x on D3D12 (360 fps) and Vulkan (300 fps, scalar: features unknown there) |

| 2ac5a8e | Result |
|---|---|
| A1 | Vulkan 64x64 and 744x464, all rungs and inputs exact; D3D12 exact |
| A2a, A2b | 9 sizes / formats (720p to 8K, ultra-wide, odd, BGRA8, RGB10A2, inverted depth) x 2x, 4x, 6x, 4 frames, both builds: 54 runs, strict 0 bytes, shipped worst 86.3 dB |
| A9, A10 | 154 checks (76 an API, 2 between), 0 failed; strict 61 of 61 identical. Dynamic resolution with jittered vectors: exact (70.5 dB before the fix, the control). Soak 2000 an API flat. Khronos + synchronisation validation: 0 messages, both hosts |
| A7 | Forced: CUDA, dot4, scalar (D3D12), int8, fp16, bf16, dot4, scalar (Vulkan), WARP: 76 of 76 each. Injected self-check failures: N = 1 all pass on the next rung; N = 2, 3 fall rung by rung as logged (fp16, int8, dot4, scalar; CUDA, dot4, scalar, none). The harness then reports failures (15 checks listed for N = 3 on Vulkan): its 4-frame scenes end while the copies are still shown, so the reference scene is copies; inferred from which checks fail, not separately shown. Not run: nvapi / nvcuda absent; the stale marker at this commit |

A3, A14, A8 were not re-run: the default path's passes, constants and shaders are unchanged (A2 compares the two APIs only); on Vulkan the network kernel is now fp16, exact by A1.
Guide gaps that remain, on purpose: MvecInvalidValue, ClipToLensClip, 4x4 UI blocks, float back buffers.
## A4, 1920x1080, release Streamline DLLs, one file (dbghelp.dll), window closed after 315-320 s

| API | multiplier | DLL | log lines at the multiplier (the first, before start, left out) | evaluations generated |
|---|---|---|---|---|
| D3D12 | 2x | 2ac5a8e | 388 of 388 | 44,905 of 45,000 |
| D3D12 | 6x | 2ac5a8e | 241 of 241 | 143,540 of 144,000 |
| Vulkan | 2x | 5a2b8c8 | 397 of 397 | 44,983 of 45,000 |
| Vulkan | 6x | 5a2b8c8 | 208 of 208 | 122,920 of 123,000 |

The two D3D12 runs were on dot4, not CUDA: the failure-injection test before them had left the crash marker set (by design), found afterwards. CUDA in the sample: the short runs below and the older CUDA
long runs (293ca58). Exit code 0 each. Runs with interruptions, same day: Vulkan on 2ac5a8e 2x 331 of 362 and 6x 200 of 207, the rest at 1x, Streamline's log saying "window not focused" in both; D3D12 on 5a2b8c8
2x 367 of 400 and 6x 118 of 321, cause not shown (the log was overwritten). Further long runs were skipped (2026-11-25). The D3D12 debug layer: the last section.

## db5dc31 (after the last review's two points)

Harness 158 checks (78 an API, 2 between), strict and shipped 0 failed, strict 63 of 63 identical; Khronos + synchronisation validation, both hosts: 0 messages, the moved output region
without the flag included. Sample, 900 frames at 1920x1080 4x: both APIs. The long runs above predate it; it changes the int8 retry after a failed float kernel and the moved output region.

## Remaining rows (DLL: db5dc31 plus NFRUX_TEST_ABSENT, marker cleared, CUDA on D3D12)

- A7: `NFRUX_TEST_ABSENT=nvapi64.dll` and `=nvcuda.dll` (the load is refused): dot4 chosen, 78 of 78 each. Without it: CUDA, 78 of 78. After a forced self-check failure CUDA stays off until the marker is
  deleted or the driver changes; seen in practice here.
- A4 between neighbours (`tests/between_neighbours.py`, 64-frame 4x dumps of the sample): D3D12 183 of 183, Vulkan 165 of 165; the worst frame's larger distance is 1.6 codes below the neighbours' own distance
  (mean 10.6 to 10.8); limit +2.
- A5 replay of those dumps (60 frames D3D12, 55 Vulkan): 4x, final-d3d12 build-strict exit 0   177 generated frames: worst PSNR 138.13 dB, off by more than 1: 0.00000%, by more than 3: 0.00000%; final-d3d12 build-ship exit 0   177 generated frames: worst PSNR 76.04 dB, off by more than 1: 0.00009%, by more than 3: 0.00000%; final-vk build-strict exit 0   162 generated frames: worst PSNR 138.13 dB, off by more than 1: 0.00000%, by more than 3: 0.00000%; final-vk build-ship exit 0   162 generated frames: worst PSNR 75.05 dB, off by more than 1: 0.00011%, by more than 3: 0.00000%.
- A8 video memory, MB (display; depth half-size), the same at 2x, 4x and 6x: D3D12 CUDA 340 / 417 / 598, D3D12 dot4 291 / 368 / 549, Vulkan 258 / 312 / 432 (1080p / 1440p / 4K). It is the process's growth.
- Sample, 1200 frames at 1920x1080, D3D12 CUDA: 2x 315 fps, 4x 558, 6x 694. OptiScaler 0.9.5 with this DLL: Vulkan 2x (288 fps); D3D12 CUDA 2x in 4 of 5 starts (378-403 fps); one start presented 1x for its whole
  length with the feature created; the cause is not shown (no focus line in the log). An earlier start in dot4 was 2x.
- The 1x OptiScaler start on D3D12 did not come back: 12 further starts (6 under the first one's conditions, window focus sampled every 0.4 s and held), 2x in all 12. Cause still not shown.

# Debug layers, and NFRUX beside DLSS-G (2026-11-25; 93cff40, shipped build, release Streamline 2.14.16, unless said)

D3D12 debug layer (Graphics Tools; `tools\run_d3d12_debug.ps1` runs the default harness; the others by `NFRUX_TEST_DIRECT`, `NFRUX_ML`, `NFRUX_TEST_INPUT_STATE`, `NFRUX_TEST_WRONG_STATE`).
Counted: corruption, error and warning messages. Streamline plugins loaded: production 2.14.16 (over the air, C:\ProgramData\NVIDIA\NGX\models\sl_common_0\versions\134672; its sl.common is byte-identical to the SDK's release one and
holds the clone name).
- Control (inputs in a wrong state, not told): 1,094 messages. Inputs only; no control for outputs or the disable word.
- Harness, 78 checks: 0 messages on COMPUTE lists with inputs in NON_PIXEL, COMMON, COPY_SOURCE; on DIRECT lists in NON_PIXEL, COMMON, 0xC0. (States the harness cannot create or upload in: not run.)
  Earlier the same day, before the fixes below (2ac5a8e or later): dot4 and scalar, and GPU-based validation on CUDA and dot4: 0 messages.
- Sample `-debug` (messages logged by NFRUX, at most six an id): 4x 79 s, 2x 13 s, 6x 18 s: no error; one warning (a fence wait for zero, id 1424, also with NVIDIA's DLSS-G), and live objects at exit
  (also with NVIDIA's).
- Not run: GPU-based validation in the sample; the long runs since these changes (skipped).

Found by the sample and not by the harness (which waits after every evaluate and keeps the guide's states), fixed: the disable word is in COPY_DEST (527); tagged inputs are not transitioned (538, 527); the self-check's
readback buffer was released when the completion token was seen, before the host's list had ended (921, corruption; that run stopped with a fence timeout; it may or may not be the one bad OptiScaler start);
NRI's shader-resource state on DIRECT lists is 0xC0 and the host's 0x40 (568 messages in the harness). NVIDIA's own DLSS-G gives no errors either, but its kernels are launched outside the layer's sight.

Vulkan, Khronos + synchronisation in the sample, 4x, 1500 frames: 64 messages (limited repeats). By function: draws and ray tracing, graphics pipeline creation, present / acquire / latency calls on several threads,
Streamline's present semaphore, device destruction (five live objects, as with NVIDIA's DLSS-G; those named are ImGui's), and vkCmdCopyImage from GBufferDepth to Streamline's depth copy. None is in a dispatch, the only
shader work NFRUX records, and the copies name only those images. (NFRUX's own objects were given names for this run; that the names took effect is not shown.) NVIDIA's DLSS-G (2x) gives every class of
message NFRUX's earlier runs did.

NFRUX and NVIDIA's DLSS-G (Streamline SDK's DLL through the driver's NGX core, on a DIRECT list; NFRUX on COMPUTE), 2x, the same inputs (`ngx_harness --replay`): a 60-frame sample recording with every
second frame removed (vectors chained by nearest pixel, matrices multiplied; `halve_recording.py`), the removed frames the truth; 28 frames, right of the UI, 1280x701. `compare_providers.py`:

| | dB | SSIM | worst frame dB | opaque UI, mean difference from the frame (codes) |
|---|---|---|---|---|
| NFRUX | 27.33 | 0.840 | 25.84 | 0.00 |
| DLSS-G | 27.12 | 0.831 | 25.87 | 1.72 |
| blend | 22.85 | 0.499 | 22.25 | 1.13 |

One recording, one scene (noisy ray tracing), one GPU: 0.2 dB is not a ranking, and whether about 27 dB is a ceiling there was not measured. The UI in the recording is one opaque-ish panel that
changes little; opaque means over 90 codes from the HUD-less picture. Picture: `build\compare.png` (not in the repository).
Re-run on 93cff40: GPU-based validation CUDA and dot4, scalar (D3D12), int8 / dot4 / scalar (Vulkan): 78 of 78 each, 0 messages; A5 replays at 4x (60 and 55 frames): strict 0 bytes, shipped 76.0 and 75.1 dB; parity 1366x767 inverted 6x and 4K
BGRA8 4x: strict 0 bytes, shipped 98.8 dB or better. On the commit after the agreement: harness strict and shipped, 2000-evaluate soak, sample 4x on both APIs (D3D12 under the debug layer).

An OptiScaler build with FGNvngxReplacement (FGInput=nvngxfg), sample 1280x701: on D3D12 it ends with an access violation in ntdll with another provider selected, as with NFRUX and
with none: not NFRUX's. Vulkan, 2x / 4x: NFRUX selected by name ("NFRUX"), integer network 229 / 226 fps; with NFRUX_VK_ASSUME_FEATURES=1: 2x 375, 4x 419, 6x 510.

Fast Vulkan network by default on NVIDIA where the host's features are unknown: OptiScaler 0.9.5 2x 380 fps (fp16 cooperative matrix), with NFRUX_VK_ASSUME_FEATURES=0 287 (integers). A marker at two unfinished starts: integers, said in the log.
A start killed early leaves the count at one; the next full run clears it. Harness 158 checks, strict 63 of 63 identical; minimal host with nothing assumed, under validation, 78 of 78; with assumption 78 of 78.

D3D12 with that OptiScaler build (d6b02459), sample 1280x701, 4x, 500 frames, 4 starts each, NFRUX and another provider: with LogLevel=0 (trace, LogToFile=true) all run: NFRUX 664-679 fps. With LogLevel=5 all 8 starts end
in the access violation; with LogLevel=2 and no file NFRUX's four do, and the other provider's four fail too (two of them may be this session's kills). So the crash depends on OptiScaler's own timing or logging, whichever provider is
selected; the cause is not found. 2x and 6x with LogLevel=0, one start each, 700 frames (a87945a's build): 427 and 805 fps, exit 0.

# CUDA kernels on Vulkan, batches interleaved (2026-11-25, RTX 4070; cd6be72, shipped build unless said; "before" is 69c8586)

Conformance: exact on five inputs at 744x464 and 992x544. GPU time a network: 744x464 0.183 ms (three interleaved 0.118 each) against fp16 cooperative matrix 0.26; 992x544 0.229 (0.189) against 0.37-0.38 (0.36 interleaved).
(The timed interleaved figure has three networks writing one output buffer. Exactness of interleaving: three networks with three inputs and their own outputs on Vulkan, 0 of 2,071,296 bytes against their references; and strict
6x batches at 720p, CUDA against the integer network, every output and intermediate identical on D3D12 and on Vulkan.)
`ngx_harness --time` (submitted and waited for, medians, ms a rendered frame, 1280x701):

| | 2x | 4x | 6x |
|---|---|---|---|
| Vulkan before / after | 1.13 / 0.98 | 2.53 / 2.08 | 3.98 / 3.19 |
| D3D12 before / after | 0.81 / 0.81 | 2.07 / 1.85 | 3.26 / 2.92 |

Wall time with a submission and a wait for each evaluate, machine otherwise idle.
Harness 158 checks with CUDA on both APIs, strict 63 of 63 identical; Vulkan validation and D3D12 debug layer 0 messages (they see the launches, not the memory the kernels touch: exactness is what supports those); every rung forced 78 of 78; injected failures go CUDA, fp16, int8, dot4, scalar; soak 1000 flat; replays 6x and 4K: strict 0 bytes.
Not run: the loader's new extension and the OptiScaler route in the sample (no windows opened).

After the review: the markers wait for the real size. A process ended at the first full-size run, twice: the third start uses fp16 cooperative matrix (Vulkan) / dot4 (D3D12), said in the log. Two sample launches on Vulkan (loader; OptiScaler)
presented 1x: Streamline created the feature and never evaluated it (no interpolation line in its log); focus is the likely cause, not shown. So the sample has not exercised CUDA on Vulkan.

CUDA on Vulkan in the sample (789ba49, 1280x701, 2500 rendered frames, window focus sampled every 0.4 s and held throughout): through the loader 4x 651 fps (583 with fp16 earlier); OptiScaler 0.9.5 2x 395 fps
(380 with fp16). Both chose "CUDA kernels, int8 tensor cores (Vulkan)", passed the self-check, exited 0 and left no marker. One start each; the sample under validation with CUDA was not run.

Weights changed from v14 to the early v1 fine-tune (2026-11-25). On the halved sample recording at 2x (28 frames, shipped build, D3D12 replay, LPIPS lower is better): v1 27.12 dB, LPIPS 0.0972; v14 27.33 dB, 0.1092; Arm v2 27.04 dB, 0.0929.
PyTorch, test_new, few pairs, LPIPS v1 / v14 / Arm: 2x .0553 / .0598 / .0544; 3x .0400 / .0436 / .0410; 4x .0482 / .0506 / .0473; 6x .0414 / .0446 / .0433. PSNR preferred v14 because it rewards mixing candidates.

Weights changed from v1 to v15 (QAT of the LPIPS-weighted full run, `output\nfru_mfg_v15\qatf`; 2026-11-27). Each model built into the DLL and replayed on the
sample recording at 2x, frames 3-29 against the true frames with the UI (PSNR / LPIPS): blend 23.64 / .137, v1 27.45 / .0835, v15 27.18 / .0744, v16 (shrink-and-perturb)
27.11 / .0751. v1 rebuilt by the same route gave the shipped DLL's bytes (29/29 frames). Held-out PyTorch scorecard, LPIPS v1 / v15 / v16:
test_new 2x .0491 / .0463 / .0488, 3x .0363 / .0349 / .0363, 4x .0468 / .0459 / .0460, 6x .0407 / .0390 / .0411; test_bal 2x .0529 / .0499 / .0535, 3x .0442 / .0418 / .0433,
4x .0309 / .0305 / .0306, 6x .0592 / .0554 / .0579; v15 ~0.3 dB below v1 in PSNR, v16 up to 1 dB below at 3x/6x and streaky on fast foliage.
With v1: parity D3D12 against Vulkan 102 dB worst of 14 frames, ML conformance exact on both APIs, NGX harness 158 checks, A3 D3D12 within 0.1 dB of PyTorch on average at 2x-6x (2 pairs a clip). Earlier A3/A14 figures in this file are v14's.
nfrux.ini (2026-11-25, headless harness, RTX 4070): every Network value on both APIs, 78 checks and 0 failed each; what cannot be made (coopmat on D3D12, CUDA without nvcuda.dll, an unknown word) falls back and says so; an environment variable wins; two crashes stop a CUDA forced by the file until IgnoreCrashMarkers=1; MaxMultiplier 2 to 6 through NFRUX_MaxFrames.

Shaders (2026-11-25, Vulkan timestamps, medians a dispatch over the harness's --time run, RTX 4070, training running): postprocess 0.0430 -> 0.0399 ms (the UI test one fetch a pixel through group memory), preprocess 0.0594 -> 0.0522 (gathers), one flow match dispatch 0.0164 -> 0.0143 (seven offsets between barriers). Output files of the recorded (24 an API) and synthetic (27 an API) scenes are byte-identical to before on both APIs in the shipped build; strict build: D3D12 and Vulkan identical, 63 of 63. NFRUX_TEST_TIMES now logs medians.

Network, CUDA (2026-11-26, ml_conformance_d3d12, RTX 4070 idle with clocks held up, 6 runs, median): a layer under 20k pixels runs 1 output column x 1 channel tile a warp instead of 2 x 8, so the small U-Net layers fill the GPU. 744x464 0.166 -> 0.153 ms, 960x544 0.229 -> 0.218 ms. Splitting every layer that way, or reordering layers by depth (already in order), did not help. Exact on both APIs; NGX harness 158 checks, 0 failed. An idle RTX 40 stays at 210 MHz through a short test: earlier timings taken idle or beside training were unusable.

## Review fixes and A3 / A14 with v1 (2026-11-26)

Two design reviews (architecture: approve with changes; code accuracy: accurate with fixes). Fixed: the input quantisation constants were still v14's (1/255, 0.498039), now v1's metadata (0.003921557, 0.4980377); cooperative matrices off NVIDIA behind a crash marker of their own (refused: dot4; an ini-forced coopmat falls back, logged; tested on the RTX 4070 with NFRUX_TEST_MAT_MARKER, NGX harness Vulkan 78 checks, 0 failed, each way); markers and logs use wide paths and log their file (a user name outside the code page lost CUDA silently); Log=true / yes is a flag; the loader's log is beside the DLL. DESIGN corrected where the reviews found it wrong (ini table, MultiFrameCountMax, CUDA layer class and layout, bf16, markers, package).

Strict build after the change: A5 replay of both 64-frame recordings at 2x, 4x and 6x, D3D12 against Vulkan, 0 differing bytes in every intermediate and output; A2a synthetic scenes (1080p, 4K, 8K, 3440x1440, 1277x719 at 2x/4x/6x; BGRA8 and RGB10A2; inverted depth): 22 of 22 identical.

A3 / A14, test_new, 4 pairs a clip, shipped D3D12 build (`tests/quality_reference.py`, LPIPS added):

| | 2x | 3x | 4x | 5x | 6x |
|---|---|---|---|---|---|
| NFRUX PSNR / LPIPS | 38.15 / .0414 | 39.16 / .0322 | 40.35 / .0395 | 33.56 / .0586 | 37.21 / .0373 |
| torch v1 | 38.05 / .0416 | 39.23 / .0320 | 40.35 / .0395 | 33.51 / .0587 | 37.12 / .0373 |
| Arm as released | 38.05 / .0395 | 38.79 / .0328 | 40.52 / .0374 | 33.44 / .0637 | 36.52 / .0387 |
| Arm with prior | 38.05 / .0395 | 38.94 / .0320 | 40.56 / .0368 | 33.75 / .0589 | 36.76 / .0371 |

A3 passes: NFRUX within 0.1 dB of torch on every multiplier's mean (limit 0.2; worst single frame -1.66 dB at 3x on a 57 dB pair). A14 (LPIPS, not above Arm as released) fails at 2x and 4x for v1, passes at 3x, 5x and 6x; PSNR says the same at 4x. The v15 retrain (LPIPS 0.5, no oracle loss) beat Arm at 2x-6x in its 4k-step probe and is the candidate to close it.


# First-release baseline (2026-11-27, a77b2c6..441bae9, shipped build, RTX 4070, idle GPU)

`ngx_harness --time` (each evaluate submitted and waited for; ms a rendered frame, medians), the automatic rung (CUDA kernels on both APIs):

| | 1280x701 | 1080p | 1440p | 4K |
|---|---|---|---|---|
| D3D12 2x / 4x / 6x | 0.84 / 1.97 / 2.94 | 1.15 / 2.14 / 3.40 | 1.44 / 2.48 / 3.77 | 1.61 / 3.26 / 5.04 |
| Vulkan 2x / 4x / 6x | 0.94 / 2.13 / 3.19 | 1.15 / 2.29 / 3.58 | 1.36 / 2.71 / 4.08 | 1.86 / 3.70 / 5.66 |

Other rungs, forced (the paths of other GPUs, run on this one), 1080p 2x / 4K 4x: Vulkan coopmat 1.32 / 4.89, Vulkan dot4 (tiled, now the NVIDIA
default) 1.45 / 4.72 (untiled before: 1.94 / 6.71), D3D12 dot4 1.52 / 5.66, Vulkan scalar 3.21 / 9.65.
GPU time at 4K 6x (Vulkan timestamps, a rendered frame): post-process 5 x 0.29 ms (near the card's bandwidth), network 1.09, preprocess 0.26,
downsamples 0.25, flow warp 0.18; the copy into the host's output 0.12 a generated frame (kept: under the bar for a direct-write path).
Rung timing on the first start: D3D12 cuda 0.150 / dot4 0.603 ms, Vulkan cuda 0.177 / coopmat 0.249 / dot4 0.603 (640x360): cuda cached, reused next start.
Video memory (`--vram`, the same at 2x-6x): D3D12 cuda 393 / 469 / 651 MB at 1080p / 1440p / 4K, dot4 291 / 368 / 549; Vulkan cuda 295 / 350 / 470,
dot4 256 / 311 / 431. Soak 3000 evaluates a API: memory and handles flat. Harness 78/78 a API (Vulkan dot4 forced too), conformance exact,
parity worst 99.8 dB, rungs_test 15/15. Not run here: AMD and Intel GPUs (their plans are tested by rungs_test only).
DLSS-G for comparison (`ngx_harness _nvngx.dll --time`, the driver's NGX core with the Streamline SDK's nvngx_dlssg 310.9, its default preset;
2x only on this card), D3D12, ms a rendered frame: 1280x701 1.47, 1080p 1.76, 1440p 2.83, 4K 3.63; NFRUX 0.84 / 1.15 / 1.44 / 1.61. `--time` now
also prints the CPU time of the evaluate calls a rendered frame (NFRUX 1080p: 0.18 ms at 2x, 0.54-0.66 at 6x).

# CUDA network: tiled kernels, one launch a layer (2026-11-27, RTX 4070 idle; e8ed750 against 373db67)

GPU time by layer (`ml_conformance_d3d12 layers`: a barrier and a timestamp after each layer, 992x544, separate inputs, medians; ms), the row kernels before / tiled after:

| layer | 0 | 1 | 2 | 3-8 | 9-13 (7x7, each) | 14-18 | 19 | 20 | 21 | sum |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 network | .021 / .021 | .028 / .014 | .015 / .009 | .029 / .024 | .009 / .005 | .027 / .020 | .017 / .008 | .012 / .007 | .022 / .008 | .215 / .135 |
| 5 networks | .177 / .156 | .129 / .054 | .060 / .031 | .091 / .052 | .046 / .010 | .099 / .048 | .073 / .031 | .048 / .021 | .092 / .029 | 1.000 / .472 |

Layer 0 (20-byte pixels, 10.8 MB of input a network) stays memory-bound. Network pass, Vulkan timestamps (NFRUX_TEST_TIMES): 1080p 2x 0.284 -> 0.166 ms, 6x 1.090 -> 0.561.
`ngx_harness --time`, ms a rendered frame (submitted and waited for, medians; both builds run alternately):

| | D3D12 before / after | Vulkan before / after |
|---|---|---|
| 1080p 2x | 1.03 / 0.94 | 1.13 / 1.05 |
| 1080p 6x | 3.32 / 2.69 | 3.58 / 3.00 |
| 4K 6x | 5.07 / 4.42 | 5.62 / 5.02 |

CPU of the evaluate calls at 6x: 0.57 -> 0.48 ms (D3D12), 0.52 -> 0.48 (Vulkan), from one launch a layer for all timesteps instead of one a timestep.
The rung trial now times both variants (first start here: cuda 0.104, cuda/1 0.152, dot4 0.603 ms at 640x360). What did not help: ldmatrix for the input
fragments (no change, layer 0 slower with its pixels padded to 32 bytes), coalesced word-by-word staging for layer 0 (0.156 -> 0.176), larger tiles at
stride 2 (layer 0 0.30). Fusing layers 20 and 21 was not tried: without the timing barriers they already run side by side. Exact: ml_conformance on both
APIs and both variants, parity worst 99.82 dB, ngx_harness 78 / 78 on each API with each variant as the default, rungs_test passes.
Tuned on the RTX 4070 only (sm_89): the tile shapes and the 100k-pixel threshold. sm_75 gets the tiled kernels with m8n8k16 and the earlier kernels as the alternative; the trial picks per GPU.

# Vulkan: only features known to be enabled (2026-11-27)

The "assumed features" option (`AssumeFeatures`, `NFRUX_VK_ASSUME_FEATURES`, on by default on NVIDIA where the host's features were unknown) is gone.
Sources of truth now (DESIGN, "Vulkan features"):
1. The loader: what `vkCreateDevice` enabled and the instance's apiVersion.
2. Otherwise Streamline's contract: Vulkan 1.3, buffer addresses, and binary import confirmed by its commands resolving.
3. Otherwise refusal.

Storage-image formats are checked per format. A forced Vulkan `cuda` needs the launch feature known as well. RTX 4070, shipped build:

| Host | Result |
|---|---|
| Loader-like (features reported) | 78/78, CUDA, Vulkan 1.3 |
| Minimal (Vulkan 1.2, nothing enabled) | 78/78 on the integer network, Khronos validation 0 messages |
| `--streamline` (sl.interposer loaded, nothing reported) | 78/78, CUDA on Vulkan |
| `--unknown` | refused, process ends normally |

Forced rungs on Vulkan: dot4, scalar, coopmat and cuda each 78/78. D3D12 78/78. Conformance exact on both APIs, parity 99.82 dB, rungs_test passes.

# Passes, barriers and copies (2026-11-27/28, RTX 4070 idle; e43258b -> 8ac6f2d, shipped build)

The network was left as tuned; this round is the rest of the frame on both APIs, D3D12 first. Every change keeps every byte: parity dumps of
every output and intermediate, both APIs, shipped and strict builds, compared with e43258b's (synthetic 1080p 2x and 6x, 4K from 1080p render, 4K
from 4K render with inverted depth, 1277x719 RGB10A2 inverted, 1280x720 BGRA8 from 640x360, 1600x900 RGBA16F, the 64-frame sample recording at 6x);
the sample recording through the front end (`ngx_harness --replay`, 2x / 4x / 6x, 108 images); ML conformance exact; `ngx_harness` 78/78 on each
API with each rung forced; strict parity 0 bytes between the APIs; rungs_test; Vulkan synchronisation validation and the D3D12 debug layer silent.

`ngx_harness --time`, ms a rendered frame (submitted and waited for, medians of 3 alternate runs; calls = CPU of the evaluates):

| | 1080p 2x / 6x | 1440p 2x / 6x | 4K 2x / 6x | 1280x701 2x / 6x |
|---|---|---|---|---|
| D3D12 before | 0.93 / 2.67 | 1.11 / 3.03 | 1.49 / 4.37 | 0.71 / 2.24 |
| D3D12 after | **0.77 / 2.40** | **0.92 / 2.72** | **1.16 / 3.86** | **0.62 / 2.03** |
| Vulkan before | 1.03 / 2.98 | 1.25 / 3.45 | 1.72 / 4.94 | 0.77 / 2.56 |
| Vulkan after | **0.92 / 2.72** | **1.09 / 3.07** | **1.47 / 4.47** | **0.73 / 2.34** |

CPU of the evaluates, 1080p 2x / 6x: D3D12 0.18 / 0.46 -> 0.16 / 0.45 ms, Vulkan 0.18 / 0.47 -> 0.18 / 0.44. Vulkan's 1080p 2x varies between
runs of the harness (0.88 to 0.97 for the same DLL); D3D12's within 0.01.

GPU time by pass, 1080p 2x (NFRUX_TEST_TIMES; D3D12 needed the first commit below to read anything; ms; "before" is e43258b with that fix).
Passes recorded in one group run side by side, so the parts are read together and `(recording)` is the total:

| | D3D12 before | D3D12 after | Vulkan before | Vulkan after |
|---|---|---|---|---|
| optical flow: hints, pyramid, levels 5-2 (and two copies before) | 0.245 | 0.153 | 0.244 | 0.155 |
| (of which the match at level 2) | 0.103 | 0.048 | 0.092 | 0.045 |
| motion vectors: clear, downsample-all, warp | 0.081 | 0.080 | 0.086 | 0.098 |
| timestep chain: warp TM1, fill, preprocess, inpaint (6) | 0.086 | 0.076 | 0.091 | 0.084 |
| network (kernels and the copies to the parameter textures) | 0.142 | 0.137 | 0.180 | 0.180 |
| composite (after: with the history store) | 0.078 | 0.082 | 0.082 | 0.086 |
| copies at the end (history; the host's output) | 0.046 | 0 | 0.046 | 0.027 |
| **(recording)** | **0.677** | **0.529** | **0.728** | **0.627** |

The changes, in order (commit messages have each one's numbers):

| Commit | What | Effect (RTX 4070) |
|---|---|---|
| b8ac05c | NFRUX_TEST_TIMES on D3D12 (the wrapped device has no queue: frequency from a compute queue made for it); `(recording)` | measurement |
| 4961861 | Flow match: row sums reuse the search texels across a row of offsets, SADs as exact float sums, Lucas-Kanade from shared memory | D3D12 1080p 2x 0.93 -> 0.85, 4K 2x 1.49 -> 1.41 |
| 443922f | Dispatches recorded in groups (barriers once before a group of independent dispatches); flow split so the motion vectors' passes go between its parts; the hints read where they are (two copies gone) | 6x: D3D12 2.59 -> 2.48, Vulkan 2.90 -> 2.81 |
| dc43723 | D3D12: frame 1 written by the composite into the host's output (unordered access, the composite's format, whole) | D3D12 4K 2x 1.40 -> 1.30, 1080p 0.83 -> 0.81 |
| 79b45fb | The first composite stores the history (no copy at the end) | D3D12 4K 2x 1.30 -> 1.25 |
| 76c10d0 | The flow's median filter inside the bilateral filter (4 dispatches less) | 1080p 2x 0.81 -> 0.80 |
| ef1917b | NFRUX_TEST_TIMES=2: passes by place | measurement |
| dde0487 | Match: sliding row sums | level 2 0.051 -> 0.048 ms |
| 67fe9c8 | Vulkan BGRA8 history stored by the composite through an RGBA8 view | Vulkan 4K 2x 1.58 -> 1.51 |
| 7821a3b | CUDA on D3D12: one UAV barrier a layer instead of one a tensor and network | 1080p 2x 0.80 -> 0.79, network 0.141 -> 0.137 ms |
| 5af8003 | Preprocess 16 x 16 workgroups (from 32 x 16) | the pass 0.059 -> 0.051 ms (both APIs) |
| 9b4ce5f | Composite: no UI test without a UI source (it compared the colour with itself) | the pass -0.005 ms |
| 8e4919b | Motion warp: 3x3 depth neighbourhoods by gathers | the pass -0.001 (D3D12) / -0.003 (Vulkan) ms |
| 8ac6f2d | Downsample-all: 2x2 / 4x4 depth regions by gathers | 4K: the pass 0.139 -> 0.129 (D3D12), 0.177 -> 0.170 (Vulkan) |

Not kept (no measurable gain, or not exact enough to keep): the pyramid's levels two to a dispatch (2-3, 4-5: 18 x 18 recomputed a tile; no change);
workgroups of 32 x 8, 16 x 8 or 8 x 8 for the 16 x 16 passes (post-process 0.088 -> 0.091 at 32 x 8; the others within a timestamp tick);
preprocess at 64 x 8 (0.060) or 32 x 32 (0.073); the inpaint passes and warp TM1 in other shapes (no change: they cost their dispatch); the
network's layers as one CUDA launch chain on D3D12 (CPU 0.050 -> 0.034 ms, exact in conformance, but the ordering of a chain's launches is not
specified, so not relied on); on Vulkan, narrower barriers between the CUDA layers, the parameter textures kept in the GENERAL layout, or their
copies from an undefined layout (no change). The CPU of a dispatch was measured (QueryPerformanceCounter around NRI's calls, 1080p 2x):
D3D12 about 0.6 us of its own and 1.7 a group (barriers 0.8, binding and the dispatch); Vulkan 0.7 and 3.2 (vkCmdBindDescriptorSets 1.4
alone); the CUDA network's recording on D3D12 0.050 ms at 2x and 0.087 at 6x, most of it NVAPI's launch calls (one a layer).

Where Vulkan's extra time is, 1080p 2x (0.92 against D3D12's 0.77): GPU 0.10 ms, of which the CUDA kernels 0.025 and their copies into the
parameter textures 0.015 (the same kernels and copies; launched by VK_NVX_binary_import and vkCmdCopyBufferToImage), the copy into the host's
output 0.027 (Vulkan outputs: their usage is not known, so the composite cannot write them; their BGRA8 is also written blue first into RGBA8),
the downsample passes (at 4K: downsample of the colour 0.077 against 0.030); and about 0.03 a recording in submitting and waiting (a recording
of one copy: 0.19 against 0.16 ms on D3D12), which `--time` counts for each of a batch's evaluates.

# Timesteps in one dispatch (2026-11-28, RTX 4070 idle; 0d5e9fa -> ebda42d, shipped build)

VARIANTS gap 2: at 4x-6x every pass after the flow ran once a generated frame. Now each runs once a batch from 3x, and the D3D12
networks without CUDA run side by side. Every byte kept: parity dumps of every output and intermediate (both APIs, shipped and strict
builds, the eight configurations of the last round) equal to 0d5e9fa's, and with dot4 forced equal to them too; the sample recording
through the front end (2x / 4x / 6x, 108 images) equal; ML conformance exact; `ngx_harness` 78/78 on each API with each rung forced, in
both builds; rungs_test; Vulkan synchronisation validation and the D3D12 debug layer silent.

`ngx_harness --time`, ms a rendered frame (submitted and waited for, medians of 2 alternate runs; the automatic rung, CUDA):

| | 1080p 2x / 4x / 6x | 1440p 2x / 4x / 6x | 4K 2x / 4x / 6x |
|---|---|---|---|
| D3D12 before | 0.77 / 1.60 / 2.40 | 0.92 / 1.89 / 2.72 | 1.17 / 2.61 / 3.86 |
| D3D12 after | 0.77 / **1.58 / 2.35** | 0.92 / **1.87 / 2.69** | 1.17 / 2.61 / **3.81** |
| Vulkan before | 0.94 / 1.72 / 2.71 | 1.11 / 2.09 / 3.07 | 1.47 / 3.00 / 4.47 |
| Vulkan after | 0.94 / **1.67 / 2.59** | **1.08** / **2.00 / 2.95** | 1.47 / **2.97 / 4.29** |

Vulkan 2x from a 4-run alternation (the same DLL reads 0.86-0.94 at 1080p and 1.03-1.11 at 1440p from one session to the next); 2x runs
the same one-timestep passes as before. CPU of the evaluates at 6x, before / after: D3D12 1080p 0.45 / 0.41 ms, 4K 0.50 / 0.47; Vulkan
1080p 0.44 / 0.34, 4K 0.51 / 0.38. GPU time of a 1080p 6x recording (NFRUX_TEST_TIMES, `(recording)`): D3D12 1.486 -> 1.457 ms, Vulkan
1.691 -> 1.566. The last round's estimate for the CPU alone was 0.06 ms (D3D12) and 0.14 (Vulkan) at 6x; measured: 0.05 and 0.12 a frame.

D3D12 with the network forced to dot4 (`NFRUX_ML=dot4`, the path of D3D12 games on AMD and Intel GPUs), 1080p, before / after: 2x 1.27 /
1.27, 4x 3.15 / 2.41, 6x 5.05 / 3.56.

| Commit | What | Effect (RTX 4070) |
|---|---|---|
| 8e2e7c5 | Warp, warp TM1, fill, preprocess, inpaint seed / mips / resolve, composite (and the warp's clear): from 3x one dispatch for all timesteps, z = timestep (`...Batch` shader variants: each timestep's own resources at binding + 16 x timestep, picked by the workgroup's z; timesteps 1-4 in the constants; the warp's shared outputs and the history stored by timestep 0). 60 dispatches -> 12 at 6x, 24 -> 12 at 3x | 1080p 6x: D3D12 2.39 -> 2.34, Vulkan 2.72 -> 2.60 |
| ebda42d | D3D12 dot4 / integer networks: a batch's networks layer by layer (each layer's barriers for all at once, dispatches side by side) instead of one network after the other | dot4 1080p 6x 5.05 -> 3.56 |

Not kept: the batched passes at 2x as well (all five bindings and the picking with one timestep): D3D12 1080p 2x 0.77 -> 0.80, Vulkan 0.86
-> 0.94; the preprocess 0.052 -> 0.059 ms. So 2x keeps the one-timestep passes (the shaders are built both ways). The preprocess's 3x3 motion
neighbourhood picked load by load: D3D12 6x 0.236 ms (5 x 0.047) -> 0.250; loaded in one pick: 0.241. The DLL grows from 4.4 to 5.5 MB with the batch
variants.

What still runs once a timestep: the networks' work itself (CUDA: one launch a layer for the batch), the copies of each network's two
outputs into the textures the composite filters (a buffer cannot be filtered with the same bytes), the composite's UI test (the same
answer for every timestep), and the copy out of each frame after the first (the host's output for frame k comes with evaluate k). Vulkan
also copies frame 1 out (gap 1).

Kit report [bench/nfrux_bench_NVIDIA-GeForce-RTX-4070_20261128-0220.md](bench/nfrux_bench_NVIDIA-GeForce-RTX-4070_20261128-0220.md) (ebda42d): correctness ok on
every check; the automatic choice is cuda/0 on both APIs (its 2x rows and the forced cuda/0 rows differ by run-to-run noise, up to 0.03 ms).
Against the 2026-11-27 report (both rounds' changes): D3D12 auto 1080p 6x 2.66 -> 2.37 ms, 4K 6x 4.38 -> 3.82; Vulkan auto 1080p 6x
3.00 -> 2.59, 4K 6x 5.27 -> 4.28; D3D12 dot4 1080p 6x 5.38 -> 3.57, 4K 6x 7.24 -> 4.98; Vulkan coopmat 1080p 6x 4.23 -> 3.88, dot4 tiled
4.43 -> 3.99.