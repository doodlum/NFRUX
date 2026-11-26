cmake_minimum_required(VERSION 3.20)
project(NFRUX CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)  # nothing to install beside the DLL
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
add_link_options(/DEBUG /OPT:REF)
add_compile_options(/Zi /MP /EHsc /wd4244 /wd4267)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
option(NFRUX_STRICT_MATH "No fused or reordered shader arithmetic: D3D12 and Vulkan give the same bytes (the proof build). Off: the shipped build" OFF)
set(GEN ${CMAKE_BINARY_DIR}/shaders)
file(GLOB_RECURSE SHADER_DEPS core/gpu/*.h core/shaders/*.glsl core/shaders/*.h core/shaders/*.hlsl core/shaders/*.hlsli core/shaders/*.comp core/ml/*.hlsl)
add_custom_command(OUTPUT ${GEN}/nfrux_shaders.h
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/build_shaders.py ${CMAKE_SOURCE_DIR}/core ${GEN} ${NFRUX_STRICT_MATH}
    DEPENDS ${SHADER_DEPS} tools/build_shaders.py VERBATIM)
# AMD FSR FG4's weights: the compressed blob in the repository expanded into a header (tools/embed_weights.py).
add_custom_command(OUTPUT ${GEN}/amd_mlfg_weights.h
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/embed_weights.py ${CMAKE_SOURCE_DIR}/core/model/amd_mlfg_weights.xz ${GEN}/amd_mlfg_weights.h
    DEPENDS core/model/amd_mlfg_weights.xz tools/embed_weights.py VERBATIM)
add_custom_target(nfrux_shaders DEPENDS ${GEN}/nfrux_shaders.h ${GEN}/amd_mlfg_weights.h)
set(NRI_DIR "${CMAKE_SOURCE_DIR}/third_party/NRI" CACHE PATH "NRI, from tools/build_nri.ps1")
add_library(nfrux_core STATIC core/log.cpp core/marker.cpp core/model.cpp core/ml/tosa_graph.cpp core/ml/cpu_reference.cpp core/ml/d3d12_network.cpp core/ml/cuda_network.cpp core/ml/cuda_vulkan.cpp core/ml/vulkan_network.cpp core/interpolator.cpp core/mlfg/mlfg_network.cpp core/mlfg/mlfg_cuda.cpp core/mlfg/mlfg_interpolator.cpp rhi/rhi_nri.cpp)
add_dependencies(nfrux_core nfrux_shaders)
target_include_directories(nfrux_core PUBLIC core core/ml core/model core/mlfg core/shaders/mlfg rhi ${GEN} $ENV{VULKAN_SDK}/Include ${NRI_DIR}/Include)
add_executable(ml_conformance_d3d12 tests/ml_conformance_d3d12.cpp)
target_link_libraries(ml_conformance_d3d12 nfrux_core d3d12 dxgi dxguid)
add_executable(ml_conformance_vulkan tests/ml_conformance_vulkan.cpp)
target_link_libraries(ml_conformance_vulkan nfrux_core $ENV{VULKAN_SDK}/Lib/vulkan-1.lib)
add_executable(nfrux_caps tools/gpu_caps.cpp)
target_link_libraries(nfrux_caps d3d12 dxgi dxguid)

add_executable(parity tests/parity.cpp)
target_include_directories(parity PRIVATE rhi core)
set(NRI_LIBS ${NRI_DIR}/_Bin/Release/NRI.lib ${NRI_DIR}/_build/Release/NRI_Shared.lib ${NRI_DIR}/_build/Release/NRI_D3D12.lib ${NRI_DIR}/_build/Release/NRI_VK.lib ${NRI_DIR}/_build/Release/NRI_Validation.lib)
target_link_libraries(parity nfrux_core ${NRI_LIBS} d3d12 dxgi dxguid $ENV{VULKAN_SDK}/Lib/vulkan-1.lib)

add_library(nvngx_dlssg SHARED frontend/ngx_exports.cpp frontend/frame_generator.cpp loader/loader.cpp)
target_include_directories(nvngx_dlssg PRIVATE rhi core)
target_link_libraries(nvngx_dlssg nfrux_core ${NRI_LIBS} d3d12 dxgi dxguid $ENV{VULKAN_SDK}/Lib/vulkan-1.lib)
set_target_properties(nvngx_dlssg PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/NFRUX)
target_link_options(nvngx_dlssg PRIVATE /PDBALTPATH:%_PDB% /OPT:ICF /DELAYLOAD:d3d12.dll /DELAYLOAD:dxgi.dll /DELAYLOAD:vulkan-1.dll)
target_link_libraries(nvngx_dlssg delayimp)
# FSR FG4 passes (core/shaders/mlfg) in isolation on both APIs, for tests/mlfg_passes.py.
add_executable(mlfg_passes tests/mlfg_passes.cpp)
add_dependencies(mlfg_passes nfrux_shaders)
target_include_directories(mlfg_passes PRIVATE rhi ${GEN} $ENV{VULKAN_SDK}/Include)
target_link_libraries(mlfg_passes d3d12 dxgi dxguid $ENV{VULKAN_SDK}/Lib/vulkan-1.lib)
# FSR FG4 interpolator on raw frames, for tests/mlfg_net.py.
add_executable(mlfg_run tests/mlfg_run.cpp)
target_link_libraries(mlfg_run nfrux_core ${NRI_LIBS} d3d12 dxgi dxguid $ENV{VULKAN_SDK}/Lib/vulkan-1.lib)
add_executable(ngx_harness tests/ngx_harness.cpp)
target_link_libraries(ngx_harness nfrux_core ${NRI_LIBS} d3d12 dxgi dxguid $ENV{VULKAN_SDK}/Lib/vulkan-1.lib)
set(NFRUX_AGILITY "" CACHE STRING "Agility SDK number (620): ngx_harness then uses .\\D3D12\\D3D12Core.dll and its debug layers")
if(NFRUX_AGILITY)
    target_compile_definitions(ngx_harness PRIVATE NFRUX_AGILITY=${NFRUX_AGILITY})
endif()
add_executable(cuda_launch experiments/cuda_launch.cpp)
target_link_libraries(cuda_launch d3d12 dxgi dxguid)
add_executable(directml_convolution experiments/directml_convolution.cpp)
target_link_libraries(directml_convolution d3d12 dxgi dxguid DirectML)
add_executable(optical_flow_hardware experiments/optical_flow_hardware.cpp)
target_include_directories(optical_flow_hardware PRIVATE $ENV{VULKAN_SDK}/Include)
target_link_libraries(optical_flow_hardware $ENV{VULKAN_SDK}/Lib/vulkan-1.lib)
# Records the GPU work of AMD's FSR Frame Generation 4 DLL (docs/FFX_MLFG.md).
add_executable(ffx_capture tools/ffx_extract/ffx_capture.cpp)
target_include_directories(ffx_capture PRIVATE $ENV{VULKAN_SDK}/Include third_party/ffx_api)
target_link_libraries(ffx_capture $ENV{VULKAN_SDK}/Lib/vulkan-1.lib)

# What is handed out: the one DLL, the texts its licenses ask for, INSTALL.txt and the optional nfrux.ini. Nothing else.
add_custom_target(package
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/package
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nvngx_dlssg> ${CMAKE_BINARY_DIR}/package/nvngx_dlssg.dll
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/licenses ${CMAKE_BINARY_DIR}/package/licenses
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/LICENSE ${CMAKE_BINARY_DIR}/package/licenses/NFRUX.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/core/LICENSE-MIT.txt ${CMAKE_BINARY_DIR}/package/licenses/Neural-Graphics-SDK-MIT.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${NRI_DIR}/LICENSE.txt ${CMAKE_BINARY_DIR}/package/licenses/NRI.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${NRI_DIR}/_build/_deps/vma-src/LICENSE.txt ${CMAKE_BINARY_DIR}/package/licenses/VulkanMemoryAllocator.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${NRI_DIR}/_build/_deps/d3d12ma-src/LICENSE.txt ${CMAKE_BINARY_DIR}/package/licenses/D3D12MemoryAllocator.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/THIRD_PARTY.md ${CMAKE_SOURCE_DIR}/INSTALL.txt ${CMAKE_SOURCE_DIR}/nfrux.ini ${CMAKE_BINARY_DIR}/package
    DEPENDS nvngx_dlssg VERBATIM)
