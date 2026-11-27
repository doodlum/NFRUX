# NFRUX bench: NVIDIA GeForce RTX 4070

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070, VEN_10DE DEV_2786 |
| Driver | 32.0.16.1088 |
| Windows | 10.0.26300 |
| NFRUX | 1.0.0 (bfbb00f) |
| Run | 20261127-1730 |

## Correctness

| Test | Result | |
|---|---|---|
| ngx_harness d3d12 | 78 checks, 0 failed | ok |
| ngx_harness vulkan | 78 checks, 0 failed | ok |
| ml_conformance_d3d12 | PASS | ok |
| ml_conformance_vulkan | ALL PASSED | ok |
| parity | 14 generated frames: worst PSNR 99.82 dB, off by more than 1: 0.00000%, by more than 3: 0.00000% | ok |

## What NFRUX chose

```
NFRUX 1.0.0 (bfbb00f)
D3D12 (NRI): NVIDIA GeForce RTX 4070, shader model 68, dot4 1
rungs: nvidia-rtx, order cuda, coopmat, dot4, scalar
network 64x64: CUDA kernels, int8 tensor cores, tiled kernels
rung timing at 640x360: 3 rungs, 6 recordings
rung timing: cuda 0.105, cuda/1 0.165, dot4 0.602 ms: cuda cached for the next start
Vulkan (NRI): NVIDIA GeForce RTX 4070, dot4 1, cooperative matrix 1
network 64x64: CUDA kernels, int8 tensor cores (Vulkan), tiled kernels
rung timing at 640x360: 4 rungs, 6 recordings
rung timing: cuda 0.138, cuda/1 0.190, coopmat 0.250, dot4 0.350 ms: cuda cached for the next start
```

## Timing (ms a rendered frame; first = the evaluate that runs the network; cpu = NFRUX's calls)

```csv
api,variant,size,mult,ms,first,cpu,note
d3d12,auto,1280x720,2x,0.72,0.72,0.17,
d3d12,auto,1280x720,4x,1.54,1.28,0.3,
d3d12,auto,1280x720,6x,2.24,1.71,0.45,
d3d12,auto,1920x1080,2x,0.88,0.88,0.18,
d3d12,auto,1920x1080,4x,1.75,1.46,0.32,
d3d12,auto,1920x1080,6x,2.66,2.09,0.47,
d3d12,auto,2560x1440,2x,1.09,1.09,0.18,
d3d12,auto,2560x1440,4x,2.05,1.73,0.32,
d3d12,auto,2560x1440,6x,3.03,2.39,0.47,
d3d12,auto,3840x2160,2x,1.49,1.49,0.21,
d3d12,auto,3840x2160,4x,2.92,2.45,0.38,
d3d12,auto,3840x2160,6x,4.38,3.49,0.52,
d3d12,cuda/0 tiled,1280x720,2x,0.67,0.67,0.17,
d3d12,cuda/0 tiled,1280x720,4x,1.55,1.27,0.31,
d3d12,cuda/0 tiled,1280x720,6x,2.25,1.71,0.46,
d3d12,cuda/0 tiled,1920x1080,2x,0.87,0.87,0.18,
d3d12,cuda/0 tiled,1920x1080,4x,1.73,1.45,0.32,
d3d12,cuda/0 tiled,1920x1080,6x,2.68,2.1,0.47,
d3d12,cuda/0 tiled,2560x1440,2x,1.09,1.09,0.18,
d3d12,cuda/0 tiled,2560x1440,4x,2.27,1.89,0.42,
d3d12,cuda/0 tiled,2560x1440,6x,3.04,2.4,0.48,
d3d12,cuda/0 tiled,3840x2160,2x,1.49,1.49,0.22,
d3d12,cuda/0 tiled,3840x2160,4x,2.91,2.45,0.38,
d3d12,cuda/0 tiled,3840x2160,6x,4.39,3.48,0.53,
d3d12,cuda/1 rows,1280x720,2x,0.78,0.78,0.16,
d3d12,cuda/1 rows,1280x720,4x,1.74,1.46,0.31,
d3d12,cuda/1 rows,1280x720,6x,2.65,2.11,0.45,
d3d12,cuda/1 rows,1920x1080,2x,0.97,0.97,0.17,
d3d12,cuda/1 rows,1920x1080,4x,1.99,1.7,0.31,
d3d12,cuda/1 rows,1920x1080,6x,3.08,2.49,0.46,
d3d12,cuda/1 rows,2560x1440,2x,1.18,1.18,0.18,
d3d12,cuda/1 rows,2560x1440,4x,2.31,1.99,0.34,
d3d12,cuda/1 rows,2560x1440,6x,3.44,2.8,0.48,
d3d12,cuda/1 rows,3840x2160,2x,1.6,1.6,0.21,
d3d12,cuda/1 rows,3840x2160,4x,3.16,2.7,0.38,
d3d12,cuda/1 rows,3840x2160,6x,4.81,3.9,0.53,
d3d12,dot4,1280x720,2x,1.21,1.21,0.14,
d3d12,dot4,1280x720,4x,3.11,2.82,0.33,
d3d12,dot4,1280x720,6x,4.91,4.36,0.49,
d3d12,dot4,1920x1080,2x,1.41,1.41,0.15,
d3d12,dot4,1920x1080,4x,3.37,3.06,0.34,
d3d12,dot4,1920x1080,6x,5.38,4.77,0.52,
d3d12,dot4,2560x1440,2x,1.59,1.59,0.16,
d3d12,dot4,2560x1440,4x,3.7,3.36,0.35,
d3d12,dot4,2560x1440,6x,5.77,5.13,0.52,
d3d12,dot4,3840x2160,2x,2.07,2.07,0.19,
d3d12,dot4,3840x2160,4x,4.53,4.06,0.4,
d3d12,dot4,3840x2160,6x,7.24,6.2,0.59,
d3d12,scalar,1920x1080,2x,2.01,2.01,0.16,
vulkan,auto,1280x720,2x,0.78,0.78,0.17,
vulkan,auto,1280x720,4x,1.66,1.34,0.31,
vulkan,auto,1280x720,6x,2.56,1.88,0.46,
vulkan,auto,1920x1080,2x,0.98,0.98,0.19,
vulkan,auto,1920x1080,4x,1.97,1.58,0.33,
vulkan,auto,1920x1080,6x,3,2.26,0.48,
vulkan,auto,2560x1440,2x,1.19,1.19,0.2,
vulkan,auto,2560x1440,4x,2.41,1.97,0.37,
vulkan,auto,2560x1440,6x,3.59,2.76,0.55,
vulkan,auto,3840x2160,2x,1.76,1.76,0.26,
vulkan,auto,3840x2160,4x,3.37,2.77,0.41,
vulkan,auto,3840x2160,6x,5.27,4.05,0.64,
vulkan,cuda/0 tiled,1280x720,2x,0.78,0.78,0.18,
vulkan,cuda/0 tiled,1280x720,4x,1.77,1.43,0.32,
vulkan,cuda/0 tiled,1280x720,6x,2.57,1.89,0.46,
vulkan,cuda/0 tiled,1920x1080,2x,0.98,0.98,0.19,
vulkan,cuda/0 tiled,1920x1080,4x,2,1.62,0.36,
vulkan,cuda/0 tiled,1920x1080,6x,3.07,2.29,0.51,
vulkan,cuda/0 tiled,2560x1440,2x,1.26,1.26,0.2,
vulkan,cuda/0 tiled,2560x1440,4x,2.32,1.92,0.34,
vulkan,cuda/0 tiled,2560x1440,6x,3.48,2.7,0.49,
vulkan,cuda/0 tiled,3840x2160,2x,1.74,1.74,0.24,
vulkan,cuda/0 tiled,3840x2160,4x,3.5,2.88,0.49,
vulkan,cuda/0 tiled,3840x2160,6x,4.98,3.87,0.53,
vulkan,cuda/1 rows,1280x720,2x,0.89,0.89,0.17,
vulkan,cuda/1 rows,1280x720,4x,2,1.67,0.32,
vulkan,cuda/1 rows,1280x720,6x,2.98,2.29,0.46,
vulkan,cuda/1 rows,1920x1080,2x,1.09,1.09,0.19,
vulkan,cuda/1 rows,1920x1080,4x,2.2,1.83,0.33,
vulkan,cuda/1 rows,1920x1080,6x,3.62,2.75,0.53,
vulkan,cuda/1 rows,2560x1440,2x,1.3,1.3,0.2,
vulkan,cuda/1 rows,2560x1440,4x,2.59,2.18,0.35,
vulkan,cuda/1 rows,2560x1440,6x,3.91,3.11,0.5,
vulkan,cuda/1 rows,3840x2160,2x,1.86,1.86,0.24,
vulkan,cuda/1 rows,3840x2160,4x,3.59,3.02,0.41,
vulkan,cuda/1 rows,3840x2160,6x,5.43,4.3,0.57,
vulkan,coopmat,1280x720,2x,1.06,1.06,0.21,
vulkan,coopmat,1280x720,4x,2.43,2.07,0.35,
vulkan,coopmat,1280x720,6x,3.81,3.11,0.52,
vulkan,coopmat,1920x1080,2x,1.21,1.21,0.19,
vulkan,coopmat,1920x1080,4x,2.7,2.32,0.37,
vulkan,coopmat,1920x1080,6x,4.23,3.48,0.53,
vulkan,coopmat,2560x1440,2x,1.43,1.43,0.2,
vulkan,coopmat,2560x1440,4x,3.1,2.68,0.39,
vulkan,coopmat,2560x1440,6x,4.75,3.92,0.56,
vulkan,coopmat,3840x2160,2x,2.01,2.01,0.22,
vulkan,coopmat,3840x2160,4x,4.08,3.51,0.43,
vulkan,coopmat,3840x2160,6x,6.28,5.13,0.63,
vulkan,coopmat_int8,1280x720,2x,1.11,1.11,0.18,
vulkan,coopmat_int8,1280x720,4x,2.71,2.35,0.35,
vulkan,coopmat_int8,1280x720,6x,4.26,3.54,0.52,
vulkan,coopmat_int8,1920x1080,2x,1.31,1.31,0.19,
vulkan,coopmat_int8,1920x1080,4x,3.11,2.7,0.45,
vulkan,coopmat_int8,1920x1080,6x,4.74,3.93,0.55,
vulkan,coopmat_int8,2560x1440,2x,1.52,1.52,0.19,
vulkan,coopmat_int8,2560x1440,4x,3.54,3.09,0.49,
vulkan,coopmat_int8,2560x1440,6x,5.22,4.39,0.58,
vulkan,coopmat_int8,3840x2160,2x,2.17,2.17,0.28,
vulkan,coopmat_int8,3840x2160,4x,4.35,3.78,0.44,
vulkan,coopmat_int8,3840x2160,6x,6.75,5.56,0.63,
vulkan,coopmat_fp16,1280x720,2x,1,1,0.18,
vulkan,coopmat_fp16,1280x720,4x,2.43,2.08,0.35,
vulkan,coopmat_fp16,1280x720,6x,3.81,3.11,0.53,
vulkan,coopmat_fp16,1920x1080,2x,1.26,1.26,0.23,
vulkan,coopmat_fp16,1920x1080,4x,2.71,2.32,0.37,
vulkan,coopmat_fp16,1920x1080,6x,4.25,3.48,0.54,
vulkan,coopmat_fp16,2560x1440,2x,1.42,1.42,0.2,
vulkan,coopmat_fp16,2560x1440,4x,3.08,2.67,0.38,
vulkan,coopmat_fp16,2560x1440,6x,4.89,4.07,0.65,
vulkan,coopmat_fp16,3840x2160,2x,2,2,0.24,
vulkan,coopmat_fp16,3840x2160,4x,4.28,3.63,0.55,
vulkan,coopmat_fp16,3840x2160,6x,6.42,5.24,0.68,
vulkan,coopmat_bf16,1280x720,2x,1.03,1.03,0.18,
vulkan,coopmat_bf16,1280x720,4x,2.52,2.14,0.36,
vulkan,coopmat_bf16,1280x720,6x,3.91,3.19,0.53,
vulkan,coopmat_bf16,1920x1080,2x,1.22,1.22,0.19,
vulkan,coopmat_bf16,1920x1080,4x,2.89,2.48,0.44,
vulkan,coopmat_bf16,1920x1080,6x,4.51,3.7,0.63,
vulkan,coopmat_bf16,2560x1440,2x,1.45,1.45,0.21,
vulkan,coopmat_bf16,2560x1440,4x,3.18,2.75,0.4,
vulkan,coopmat_bf16,2560x1440,6x,4.81,4,0.55,
vulkan,coopmat_bf16,3840x2160,2x,2.04,2.04,0.26,
vulkan,coopmat_bf16,3840x2160,4x,4.14,3.56,0.45,
vulkan,coopmat_bf16,3840x2160,6x,6.34,5.2,0.63,
vulkan,dot4 tiled,1280x720,2x,1.16,1.16,0.18,
vulkan,dot4 tiled,1280x720,4x,2.59,2.22,0.35,
vulkan,dot4 tiled,1280x720,6x,4,3.29,0.53,
vulkan,dot4 tiled,1920x1080,2x,1.36,1.36,0.19,
vulkan,dot4 tiled,1920x1080,4x,2.87,2.47,0.38,
vulkan,dot4 tiled,1920x1080,6x,4.43,3.68,0.55,
vulkan,dot4 tiled,2560x1440,2x,1.57,1.57,0.2,
vulkan,dot4 tiled,2560x1440,4x,3.74,3.2,0.73,
vulkan,dot4 tiled,2560x1440,6x,4.92,4.09,0.54,
vulkan,dot4 tiled,3840x2160,2x,2.16,2.16,0.25,
vulkan,dot4 tiled,3840x2160,4x,4.22,3.65,0.44,
vulkan,dot4 tiled,3840x2160,6x,6.63,5.36,0.65,
vulkan,dot4 untiled,1280x720,2x,1.61,1.61,0.19,
vulkan,dot4 untiled,1280x720,4x,4.07,3.69,0.35,
vulkan,dot4 untiled,1280x720,6x,6.5,5.77,0.52,
vulkan,dot4 untiled,1920x1080,2x,1.79,1.79,0.19,
vulkan,dot4 untiled,1920x1080,4x,4.31,3.92,0.37,
vulkan,dot4 untiled,1920x1080,6x,7.16,6.32,0.56,
vulkan,dot4 untiled,2560x1440,2x,2.08,2.08,0.24,
vulkan,dot4 untiled,2560x1440,4x,4.74,4.29,0.38,
vulkan,dot4 untiled,2560x1440,6x,7.43,6.59,0.56,
vulkan,dot4 untiled,3840x2160,2x,2.47,2.47,0.24,
vulkan,dot4 untiled,3840x2160,4x,5.68,5.1,0.43,
vulkan,dot4 untiled,3840x2160,6x,9.2,7.98,0.65,
vulkan,scalar,1920x1080,2x,2.86,2.86,0.2,
```
