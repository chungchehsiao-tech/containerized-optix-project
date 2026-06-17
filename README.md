# Containerized OptiX 7 Ray Tracing Engine
![C++](https://img.shields.io/badge/C++-14-blue.svg)
![OptiX](https://img.shields.io/badge/OptiX-7.2+-brightgreen.svg)
![CUDA](https://img.shields.io/badge/CUDA-10.0+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)

A hardware-accelerated, containerized ray tracing engine built from scratch using C++, CUDA, and the NVIDIA OptiX 7.2 API. This project serves as a structural showcase of modern host-device memory management, Shader Binding Table (SBT) architecture, and reproducible Docker containerization for GPU build environments.

## 🖼️ Render Output

The engine outputs the final frame buffer as a 1920x1080 .ppm file (converted to .png below for web reference).
![alt text](ppmOutput.png)  
## 🏗️ Architecture: The Dual-Pipeline Strategy

This project uses a split-pipeline architecture to navigate the limitations of Windows Subsystem for Linux (WSL2) GPU passthrough.

### 1. Local Execution (Native Windows CMake)

**Used for: Active Development, Debugging, and Rendering**  
While containerized development is an industry standard, executing the NVIDIA OptiX graphics runtime through a Docker container on Windows is fundamentally blocked by WSL2 virtualization limits. 
* **The WSL2 Limitation:** While compute APIs (CUDA) pass through seamlessly, the Windows DirectX Graphics Kernel (Dxgkrnl) bridge often fails to pass the OptiX entry symbols (`libnvoptix.so.1`) to the container OS, resulting in `OptiX Error 7805 (OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND)`.  
* **The Solution:** We bypass virtualization entirely for execution. Local development relies on the native Windows MSVC compiler via CMake. This ensures the compiled `.exe` has direct access to the host `nvoptix.dll` display driver.  

### 2. Docker Containerization  

**Used for: Automated Build Verification & Cloud Testing in the future**  
Despite the execution limitation on Windows, this project maintains a multi-stage `Dockerfile`.   
* **The Linux Usage:** The Docker environment is used strictly as a proof-of-concept. It validates that the C++ host code, CUDA `.ptx` payloads, and CMake targets successfully compile in a clean Linux environment (`nvidia/cuda:11.8.0`).  
* **EULA Compliance:** To adhere to the NVIDIA OptiX 7.2 EULA regarding the redistribution of proprietary SDK headers, the Docker pipeline builds the container only if the local user provides their own copy of the SDK. The Docker image will refuse to build if the SDK is missing.  
By ensuring the CMake build system targets both legacy local environments (CUDA 10.x) and modern deployment targets (CUDA 11.8), the architecture guarantees forward compatibility.  

## 🛠️ Prerequisites  

To run this engine, your host machine requires:  
1. An NVIDIA GPU with physically active RT Cores (RTX 20-series or newer).  
2. [NVIDIA Display Drivers](https://www.nvidia.com/Download/index.aspx) installed on the host OS.  
3. CMake (3.18+) and a C++14 compliant compiler (MSVC for Windows, GCC for Linux).  
4. Docker Desktop (with WSL2 backend if on Windows).  
5. The NVIDIA OptiX 7.2 SDK (placed in the appropriate directory as expected by CMake).  

### 🛠️ Build Instructions  

**For Windows Developers:**  
Do not run the application via Docker Desktop. Compile and execute natively:  
```bash  
cmake -B build -S .  
cmake --build build --config Debug  
./build/Debug/OptixApp.exe  
```

**For Native Linux Developers:**  
You may execute the application entirely within the container (as native Linux bypasses the WSL2 virtualization bridge):  
```bash  
docker build -t optix-app .  
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all optix-app  
```