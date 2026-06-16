#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <fstream>
#include <sstream>
#include <vector>
#include "LaunchParams.h"
#include "render/OptiXRenderer.h"
#include "util/ImageSaver.h"

// -----------------------------------------------------------------------------
// MAIN PROGRAM
// -----------------------------------------------------------------------------
int main() {
    
    OptixRenderer renderer; // Constructor fires
    renderer.buildPipeline();
    renderer.buildSBT();
    renderer.bufferAlloc();
    renderer.renderFrame();


/*
    // -------------------------------------------------------------------------
    // IMAGE RETRIEVAL FROM BUFFER & SAVING
    // -------------------------------------------------------------------------
    std::cout << "Downloading image from GPU...\n";
    
    // Create an empty CPU array to hold the image
    std::vector<uchar4> host_pixels(width * height);

    // Download the VRAM buffer back across the PCIe bus to CPU RAM
    CUDA_CHECK(cudaMemcpy(
        host_pixels.data(),
        d_resultBuffer,
        width * height * sizeof(uchar4),
        cudaMemcpyDeviceToHost
    ));

    // To guarantee zero external dependencies, we use the raw PPM image format.
    std::cout << "Saving image to output.ppm...\n";
    std::ofstream file("output.ppm");
    file << "P3\n" << width << " " << height << "\n255\n"; // PPM Header
    for (int i = 0; i < width * height; ++i) {
        file << (int)host_pixels[i].x << " " 
             << (int)host_pixels[i].y << " " 
             << (int)host_pixels[i].z << "\n";
    }
    file.close();

    std::cout << "[SUCCESS] Image successfully rendered and saved to the project folder!\n";
*/
    

    return 0;
}