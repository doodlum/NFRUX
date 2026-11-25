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