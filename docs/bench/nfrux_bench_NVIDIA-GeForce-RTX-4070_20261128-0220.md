# NFRUX bench: NVIDIA GeForce RTX 4070

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070, VEN_10DE DEV_2786 |
| Driver | 32.0.16.1088 |
| Windows | 10.0.26300 |
| NFRUX | 0.0.1 (ebda42d) |
| Run | 20261128-0220 |

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
NFRUX 0.0.1 (ebda42d)
D3D12 (NRI): NVIDIA GeForce RTX 4070, shader model 68, dot4 1
rungs: nvidia-rtx, order cuda, coopmat, dot4, scalar
network 64x64: CUDA kernels, int8 tensor cores, tiled kernels
rung timing at 640x360: 3 rungs, 6 recordings
rung timing: cuda 0.101, cuda/1 0.159, dot4 0.601 ms: cuda cached for the next start
Vulkan (NRI): NVIDIA GeForce RTX 4070, dot4 1, cooperative matrix 1
network 64x64: CUDA kernels, int8 tensor cores (Vulkan), tiled kernels
rung timing at 640x360: 4 rungs, 6 recordings
rung timing: cuda 0.137, cuda/1 0.189, coopmat 0.251, dot4 0.349 ms: cuda cached for the next start
```

## Timing (ms a rendered frame; first = the evaluate that runs the network; cpu = NFRUX's calls)

```csv
api,variant,size,mult,ms,first,cpu,note
d3d12,auto,1280x720,2x,0.62,0.62,0.15,
d3d12,auto,1280x720,4x,1.35,1.1,0.28,
d3d12,auto,1280x720,6x,1.97,1.44,0.4,
d3d12,auto,1920x1080,2x,0.76,0.76,0.19,
d3d12,auto,1920x1080,4x,1.59,1.31,0.29,
d3d12,auto,1920x1080,6x,2.37,1.77,0.43,
d3d12,auto,2560x1440,2x,0.95,0.95,0.19,
d3d12,auto,2560x1440,4x,1.93,1.62,0.33,
d3d12,auto,2560x1440,6x,2.74,2.08,0.46,
d3d12,auto,3840x2160,2x,1.17,1.17,0.2,
d3d12,auto,3840x2160,4x,2.61,2.14,0.35,
d3d12,auto,3840x2160,6x,3.82,2.91,0.48,
d3d12,cuda/0 tiled,1280x720,2x,0.59,0.59,0.16,
d3d12,cuda/0 tiled,1280x720,4x,1.36,1.1,0.29,
d3d12,cuda/0 tiled,1280x720,6x,1.97,1.44,0.39,
d3d12,cuda/0 tiled,1920x1080,2x,0.76,0.76,0.19,
d3d12,cuda/0 tiled,1920x1080,4x,1.57,1.29,0.29,
d3d12,cuda/0 tiled,1920x1080,6x,2.38,1.78,0.42,
d3d12,cuda/0 tiled,2560x1440,2x,0.92,0.92,0.19,
d3d12,cuda/0 tiled,2560x1440,4x,1.92,1.6,0.32,
d3d12,cuda/0 tiled,2560x1440,6x,2.71,2.07,0.46,
d3d12,cuda/0 tiled,3840x2160,2x,1.17,1.17,0.2,
d3d12,cuda/0 tiled,3840x2160,4x,2.62,2.15,0.35,
d3d12,cuda/0 tiled,3840x2160,6x,3.81,2.91,0.47,
d3d12,cuda/1 rows,1280x720,2x,0.69,0.69,0.15,
d3d12,cuda/1 rows,1280x720,4x,1.54,1.29,0.28,
d3d12,cuda/1 rows,1280x720,6x,2.38,1.84,0.41,
d3d12,cuda/1 rows,1920x1080,2x,0.84,0.84,0.16,
d3d12,cuda/1 rows,1920x1080,4x,1.78,1.48,0.3,
d3d12,cuda/1 rows,1920x1080,6x,2.78,2.18,0.42,
d3d12,cuda/1 rows,2560x1440,2x,1.07,1.07,0.19,
d3d12,cuda/1 rows,2560x1440,4x,2.1,1.77,0.33,
d3d12,cuda/1 rows,2560x1440,6x,3.14,2.47,0.46,
d3d12,cuda/1 rows,3840x2160,2x,1.29,1.29,0.2,
d3d12,cuda/1 rows,3840x2160,4x,2.85,2.38,0.36,
d3d12,cuda/1 rows,3840x2160,6x,4.23,3.31,0.47,
d3d12,dot4,1280x720,2x,1.13,1.13,0.14,
d3d12,dot4,1280x720,4x,2.21,1.93,0.29,
d3d12,dot4,1280x720,6x,3.2,2.63,0.44,
d3d12,dot4,1920x1080,2x,1.28,1.28,0.15,
d3d12,dot4,1920x1080,4x,2.43,2.12,0.31,
d3d12,dot4,1920x1080,6x,3.57,2.97,0.45,
d3d12,dot4,2560x1440,2x,1.45,1.45,0.18,
d3d12,dot4,2560x1440,4x,2.73,2.39,0.34,
d3d12,dot4,2560x1440,6x,3.94,3.27,0.49,
d3d12,dot4,3840x2160,2x,1.77,1.77,0.18,
d3d12,dot4,3840x2160,4x,3.37,2.91,0.35,
d3d12,dot4,3840x2160,6x,4.98,4.08,0.5,
d3d12,scalar,1920x1080,2x,1.87,1.87,0.15,
vulkan,auto,1280x720,2x,0.69,0.69,0.16,
vulkan,auto,1280x720,4x,1.51,1.19,0.24,
vulkan,auto,1280x720,6x,2.21,1.54,0.32,
vulkan,auto,1920x1080,2x,0.89,0.89,0.2,
vulkan,auto,1920x1080,4x,1.74,1.38,0.26,
vulkan,auto,1920x1080,6x,2.59,1.87,0.34,
vulkan,auto,2560x1440,2x,1.06,1.06,0.18,
vulkan,auto,2560x1440,4x,2.04,1.67,0.27,
vulkan,auto,2560x1440,6x,2.94,2.18,0.34,
vulkan,auto,3840x2160,2x,1.48,1.48,0.21,
vulkan,auto,3840x2160,4x,2.98,2.41,0.31,
vulkan,auto,3840x2160,6x,4.28,3.18,0.38,
vulkan,cuda/0 tiled,1280x720,2x,0.69,0.69,0.16,
vulkan,cuda/0 tiled,1280x720,4x,1.54,1.22,0.25,
vulkan,cuda/0 tiled,1280x720,6x,2.19,1.54,0.32,
vulkan,cuda/0 tiled,1920x1080,2x,0.89,0.89,0.2,
vulkan,cuda/0 tiled,1920x1080,4x,1.72,1.38,0.26,
vulkan,cuda/0 tiled,1920x1080,6x,2.6,1.88,0.34,
vulkan,cuda/0 tiled,2560x1440,2x,1.03,1.03,0.18,
vulkan,cuda/0 tiled,2560x1440,4x,2.09,1.69,0.27,
vulkan,cuda/0 tiled,2560x1440,6x,2.96,2.18,0.35,
vulkan,cuda/0 tiled,3840x2160,2x,1.48,1.48,0.21,
vulkan,cuda/0 tiled,3840x2160,4x,2.98,2.41,0.31,
vulkan,cuda/0 tiled,3840x2160,6x,4.27,3.18,0.38,
vulkan,cuda/1 rows,1280x720,2x,0.79,0.79,0.16,
vulkan,cuda/1 rows,1280x720,4x,1.71,1.39,0.25,
vulkan,cuda/1 rows,1280x720,6x,2.63,1.94,0.33,
vulkan,cuda/1 rows,1920x1080,2x,0.99,0.99,0.19,
vulkan,cuda/1 rows,1920x1080,4x,1.95,1.59,0.26,
vulkan,cuda/1 rows,1920x1080,6x,3.01,2.28,0.34,
vulkan,cuda/1 rows,2560x1440,2x,1.15,1.15,0.19,
vulkan,cuda/1 rows,2560x1440,4x,2.24,1.86,0.27,
vulkan,cuda/1 rows,2560x1440,6x,3.35,2.58,0.35,
vulkan,cuda/1 rows,3840x2160,2x,1.59,1.59,0.21,
vulkan,cuda/1 rows,3840x2160,4x,3.11,2.55,0.31,
vulkan,cuda/1 rows,3840x2160,6x,4.7,3.59,0.39,
vulkan,coopmat,1280x720,2x,0.91,0.91,0.16,
vulkan,coopmat,1280x720,4x,2.22,1.87,0.28,
vulkan,coopmat,1280x720,6x,3.44,2.75,0.38,
vulkan,coopmat,1920x1080,2x,1.12,1.12,0.2,
vulkan,coopmat,1920x1080,4x,2.44,2.07,0.29,
vulkan,coopmat,1920x1080,6x,3.88,3.13,0.42,
vulkan,coopmat,2560x1440,2x,1.29,1.29,0.21,
vulkan,coopmat,2560x1440,4x,2.76,2.35,0.31,
vulkan,coopmat,2560x1440,6x,4.24,3.45,0.45,
vulkan,coopmat,3840x2160,2x,1.74,1.74,0.22,
vulkan,coopmat,3840x2160,4x,3.59,3.03,0.35,
vulkan,coopmat,3840x2160,6x,5.49,4.39,0.45,
vulkan,coopmat_int8,1280x720,2x,1.01,1.01,0.16,
vulkan,coopmat_int8,1280x720,4x,2.49,2.13,0.28,
vulkan,coopmat_int8,1280x720,6x,3.88,3.19,0.38,
vulkan,coopmat_int8,1920x1080,2x,1.19,1.19,0.17,
vulkan,coopmat_int8,1920x1080,4x,2.71,2.34,0.29,
vulkan,coopmat_int8,1920x1080,6x,4.26,3.52,0.38,
vulkan,coopmat_int8,2560x1440,2x,1.38,1.38,0.2,
vulkan,coopmat_int8,2560x1440,4x,3.01,2.61,0.3,
vulkan,coopmat_int8,2560x1440,6x,4.61,3.82,0.4,
vulkan,coopmat_int8,3840x2160,2x,1.84,1.84,0.21,
vulkan,coopmat_int8,3840x2160,4x,3.86,3.29,0.35,
vulkan,coopmat_int8,3840x2160,6x,5.93,4.84,0.45,
vulkan,coopmat_fp16,1280x720,2x,0.91,0.91,0.16,
vulkan,coopmat_fp16,1280x720,4x,2.22,1.87,0.28,
vulkan,coopmat_fp16,1280x720,6x,3.47,2.77,0.39,
vulkan,coopmat_fp16,1920x1080,2x,1.12,1.12,0.2,
vulkan,coopmat_fp16,1920x1080,4x,2.45,2.07,0.29,
vulkan,coopmat_fp16,1920x1080,6x,3.83,3.09,0.39,
vulkan,coopmat_fp16,2560x1440,2x,1.28,1.28,0.2,
vulkan,coopmat_fp16,2560x1440,4x,2.8,2.38,0.32,
vulkan,coopmat_fp16,2560x1440,6x,4.34,3.46,0.43,
vulkan,coopmat_fp16,3840x2160,2x,1.76,1.76,0.23,
vulkan,coopmat_fp16,3840x2160,4x,3.81,3.18,0.47,
vulkan,coopmat_fp16,3840x2160,6x,5.52,4.41,0.47,
vulkan,coopmat_bf16,1280x720,2x,0.98,0.98,0.19,
vulkan,coopmat_bf16,1280x720,4x,2.28,1.92,0.29,
vulkan,coopmat_bf16,1280x720,6x,3.53,2.83,0.38,
vulkan,coopmat_bf16,1920x1080,2x,1.13,1.13,0.19,
vulkan,coopmat_bf16,1920x1080,4x,2.49,2.12,0.29,
vulkan,coopmat_bf16,1920x1080,6x,3.9,3.16,0.39,
vulkan,coopmat_bf16,2560x1440,2x,1.28,1.28,0.18,
vulkan,coopmat_bf16,2560x1440,4x,2.8,2.4,0.31,
vulkan,coopmat_bf16,2560x1440,6x,4.25,3.46,0.4,
vulkan,coopmat_bf16,3840x2160,2x,1.76,1.76,0.22,
vulkan,coopmat_bf16,3840x2160,4x,3.7,3.12,0.37,
vulkan,coopmat_bf16,3840x2160,6x,5.58,4.48,0.45,
vulkan,dot4 tiled,1280x720,2x,1.06,1.06,0.16,
vulkan,dot4 tiled,1280x720,4x,2.36,2.01,0.28,
vulkan,dot4 tiled,1280x720,6x,3.62,2.92,0.38,
vulkan,dot4 tiled,1920x1080,2x,1.24,1.24,0.17,
vulkan,dot4 tiled,1920x1080,4x,2.58,2.2,0.29,
vulkan,dot4 tiled,1920x1080,6x,3.99,3.25,0.39,
vulkan,dot4 tiled,2560x1440,2x,1.44,1.44,0.2,
vulkan,dot4 tiled,2560x1440,4x,2.88,2.48,0.3,
vulkan,dot4 tiled,2560x1440,6x,4.34,3.55,0.39,
vulkan,dot4 tiled,3840x2160,2x,1.89,1.89,0.21,
vulkan,dot4 tiled,3840x2160,4x,3.74,3.18,0.36,
vulkan,dot4 tiled,3840x2160,6x,5.67,4.57,0.45,
vulkan,dot4 untiled,1280x720,2x,1.51,1.51,0.17,
vulkan,dot4 untiled,1280x720,4x,3.83,3.47,0.29,
vulkan,dot4 untiled,1280x720,6x,6.1,5.4,0.38,
vulkan,dot4 untiled,1920x1080,2x,1.67,1.67,0.17,
vulkan,dot4 untiled,1920x1080,4x,4.04,3.66,0.29,
vulkan,dot4 untiled,1920x1080,6x,6.46,5.72,0.39,
vulkan,dot4 untiled,2560x1440,2x,1.86,1.86,0.19,
vulkan,dot4 untiled,2560x1440,4x,4.35,3.94,0.3,
vulkan,dot4 untiled,2560x1440,6x,6.82,6.02,0.4,
vulkan,dot4 untiled,3840x2160,2x,2.37,2.37,0.22,
vulkan,dot4 untiled,3840x2160,4x,5.18,4.62,0.34,
vulkan,dot4 untiled,3840x2160,6x,8.16,7.05,0.45,
vulkan,scalar,1920x1080,2x,2.73,2.73,0.17,
```
