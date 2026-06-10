#include <optix.h>
#include "LaunchParams.h"

// -----------------------------------------------------------------------------
// CONSTANT MEMORY
// -----------------------------------------------------------------------------
extern "C" __constant__ LaunchParams optixLaunchParams;

// -----------------------------------------------------------------------------
// RAY GENERATION PROGRAM
// Entry point. One thread per pixel.
// -----------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
    // Find out which pixel to look at for the thread
    const uint3 idx = optixGetLaunchIndex();
    
    // Boundary check if it is outside the window
    if (idx.x >= optixLaunchParams.width || idx.y >= optixLaunchParams.height) {
        return;
    }

    // Calculate normalized UV coordinates (0.0 to 1.0)
    float u = static_cast<float>(idx.x) / static_cast<float>(optixLaunchParams.width);
    float v = static_cast<float>(idx.y) / static_cast<float>(optixLaunchParams.height);

    // Create color gradient in red and green channel
    int r = static_cast<int>(255.0f * u);
    int g = static_cast<int>(255.0f * v);
    int b = 100; // do not change blue channel now...
    int a = 255; // Opaque

    // Convert 2D pixel coordinates (x, y) into a 1D memory array index
    const uint32_t fb_index = idx.y * optixLaunchParams.width + idx.x;

    // 6. Write the color to our VRAM buffer
    optixLaunchParams.resultBuffer[fb_index] = make_uchar4(r, g, b, a);
}