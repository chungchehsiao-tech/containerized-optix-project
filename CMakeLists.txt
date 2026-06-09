cmake_minimum_required(VERSION 3.18)

# Enable both C++ and CUDA languages natively
project(ContainerizedOptixProject LANGUAGES CXX CUDA)

# Force C++ standards to CUDA 10.0 and OptiX 7.2
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# 1. SDK PATH CONFIGURATION
# ------------------------------------------------------------------
# Set a variable for the OptiX headers. 
# In local development, points to our deps/ folder, which is in .gitignore and .dockerignore
# Need to override this to point to the cloned optix-dev repo in our CI/CD pipeline
set(OPTIX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/OptiX_SDK_7.2/include" CACHE PATH "Path to OptiX SDK headers")

# ------------------------------------------------------------------
# 2. DEVICE CODE (CUDA -> PTX Compilation)
# ------------------------------------------------------------------
# OptiX requires device code to be compiled to PTX (Parallel Thread Execution)
add_library(DeviceCode OBJECT src/shader.cu)

set_target_properties(DeviceCode PROPERTIES 
    CUDA_PTX_COMPILATION ON         # Tells NVCC to output .ptx instead of .obj/.o
    CUDA_ARCHITECTURES "75"         # e.g., 75 = Turing GPUs (RTX 20-series). Adjust as needed.
)
target_include_directories(DeviceCode PRIVATE ${OPTIX_INCLUDE_DIR})

# ------------------------------------------------------------------
# 3. HOST CODE (C++ Executable)
# ------------------------------------------------------------------
add_executable(OptixApp src/main.cpp)

# Give the host code access to the OptiX headers
target_include_directories(OptixApp PRIVATE ${OPTIX_INCLUDE_DIR})

# Find the CUDA Toolkit to link the CUDA Runtime (cudart) for host memory management
find_package(CUDAToolkit REQUIRED)
target_link_libraries(OptixApp PRIVATE CUDA::cudart)