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
