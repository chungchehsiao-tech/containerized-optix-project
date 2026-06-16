#pragma once
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <optix.h>

// CUDA_CHECK and OPTIX_CHECK MACROS
// -----------------------------------------------------------------------------
// ERROR HANDLING MACROS
// Wrap API calls ensure hardware failures don't fail silently.
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

#define OPTIX_CHECK(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::cerr << "OptiX Error: " << res \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)


// -----------------------------------------------------------------------------
// OPTIX LOG CALLBACK
// Allow the NVIDIA driver to send internal warnings/errors to our terminal
// -----------------------------------------------------------------------------
inline static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}


