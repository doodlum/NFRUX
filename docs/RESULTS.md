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

`loader/probe.cpp` as `dbghelp.dll` beside the sample (sl.interposer imports it): IAT patch of GetProcAddress / LoadLibraryExW in every non-Windows
module, LdrRegisterDllNotification for later ones. NGX core loaded from the DriverStore; exports of interest wrapped; no system setting changed.
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

D3D12 (rhi_d3d12, SM 6.4 dot4 network) against Vulkan (rhi_nri on NRI v192, int8 cooperative matrix): **every generated frame and every
intermediate compared byte for byte, 0 differ**, at 1280x720 (2x, 3x, 5x, 6x), and 4x at 1920x1080, 1920x1080 from 1280x720, 1280x800 from
742x464, 2560x1440 from 1707x960, 1366x767. Mean PSNR against the true in-between frames 43-45 dB (the same on both, by construction).

Faults the comparison found, all fixed in the shared source or backend:
- `findMSB(0)`: -1 in GLSL, unsigned ~0 after SPIRV-Cross (zero-flow texels got exponent 15 on D3D12);
- SPIRV-Cross emits the 32-bit `InterlockedMin` for 64-bit raw buffer atomics (D3D12's 64-bit scatter was wrong; `InterlockedMin64` now);
- Vulkan: an image's first layout transition needs earlier work as its source, else the first interpolated frame was corrupted at 5x and
  more (not reported by synchronization validation, whose activation was not proven; mechanism not established);
- `log` / `exp` written as `log2` / `exp2`.

Not covered yet: NRI on D3D12 (crashes, being looked at), HDR formats, BGRA8, inverted depth, UI inputs, real captures, 4K, other GPUs, the
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
"no more than 1 dB below the inputs" part met. The cross-replay part of A5 is not done.