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

# Define output to go inside build folder
set(PTX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/ptx")
file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

# Force the path to strictly use forward slashes, escaping it for Windows
file(TO_CMAKE_PATH "${PTX_OUTPUT_DIR}" SAFE_PTX_PATH)
set(PTX_FILE "${PTX_OUTPUT_DIR}/shader.ptx")

add_executable(OptixApp src/main.cpp)

# The absolute path of the PTX folder directly into the C++ code
target_compile_definitions(OptixApp PRIVATE "PTX_DIR=\"${PTX_OUTPUT_DIR}\"")

# Give the host code access to the OptiX headers
target_include_directories(OptixApp PRIVATE ${OPTIX_INCLUDE_DIR})

# ------------------------------------------------------------------------------
# OPTIX PTX COMPILATION
# Force the NVIDIA compiler to generate PTX assembly for our shaders.
# ------------------------------------------------------------------------------



set(PTX_FILE "${PTX_OUTPUT_DIR}/shader.ptx")

# Create the exact terminal command to compile the shader
add_custom_command(
    OUTPUT ${PTX_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER} 
            -ptx                               # Stop at PTX assembly
            -lineinfo                          # for Nsight Debugging
            -I${CMAKE_SOURCE_DIR}/src          # Tell it where LaunchParams.h is
            -I${OPTIX_INCLUDE_DIR}             # Tell it where optix.h is
            ${CMAKE_SOURCE_DIR}/src/shader.cu  # The file to compile
            -o ${PTX_FILE}                     # Where to save it
    DEPENDS ${CMAKE_SOURCE_DIR}/src/shader.cu
            ${CMAKE_SOURCE_DIR}/src/LaunchParams.h
    COMMENT "Compiling shader.cu to PTX..."
)

# Create a fake "Target" to hold
add_custom_target(CompilePTX DEPENDS ${PTX_FILE})

# Force main C++ program to wait for the PTX to finish building
add_dependencies(OptixApp CompilePTX)

# Find the CUDA Toolkit to link the CUDA Runtime (cudart) for host memory management
find_package(CUDAToolkit REQUIRED)
target_link_libraries(OptixApp PRIVATE CUDA::cudart_static)