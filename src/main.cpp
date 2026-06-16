#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <optix.h>

#include <vector>
#include "LaunchParams.h"
#include "render/OptiXRenderer.h"
#include "util/ImageSaver.h"

// -----------------------------------------------------------------------------
// MAIN PROGRAM
// -----------------------------------------------------------------------------
int main() {
    
    OptixRenderer renderer; // Constructor

    renderer.buildPipeline();
    renderer.buildSBT();
    renderer.bufferAlloc();
    renderer.renderFrame();

    std::vector<uchar4> image_pixels(1920 * 1080);
    renderer.downloadImage(image_pixels);
    savePPM("output.ppm", image_pixels, 1920, 1080);
    
    return 0;
}