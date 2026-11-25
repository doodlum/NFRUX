# Third-party material

| Part | Origin | License |
|---|---|---|
| Network weights (core/model) | **Modified** from Arm's NFRU v2 (neural-frame-rate-upscaling): shrunk, perturbed, retrained on 20 input channels, quantised (nfru_mfg_v14_int8) | Arm AI Model Community License 1.0, `licenses/Arm_AI_Model_Community_License_v1_0_PRE-1154.pdf` |
| Pass shaders, pass order, network runtime | Arm Neural Graphics SDK 1.1.1 (`core/UPSTREAM.md`), modified | MIT (`core/LICENSE-MIT.txt`; in the package `licenses/Neural-Graphics-SDK-MIT.txt`) |
| Optical flow shaders (`arm_*`) | Arm's ML Emulation Layer for Vulkan, modified (each file says so) | Apache-2.0 (`licenses/Apache-2.0.txt`) |
| NRI 993db1a (statically linked), with Vulkan Memory Allocator and D3D12 Memory Allocator | NVIDIA-RTX/NRI; GPUOpen | MIT; MIT; MIT |

Anyone receiving the weights or a product containing them is bound by clause 4 (Acceptable Use) of the Arm license above; a distribution
must carry that license, these notices and clause 4 as an enforceable term. Arm does not endorse this project.

No NVAPI or CUDA header is used (function ids and structures are NFRUX's own declarations). Streamline, OptiScaler and
NVIDIA DLLs are used for testing only and are not distributed.