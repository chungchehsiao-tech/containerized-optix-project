#pragma once
#include <optix.h>
#include <vector>
#include <optix_stubs.h>

#include <fstream>
#include <sstream>
#include "../LaunchParams.h"

class OptixRenderer {
public:
    // OptiX constructor and destructor
    OptixRenderer();
    ~OptixRenderer();

    // Load PTX, Create Programs, Link Pipeline 
    void buildPipeline(); 
    // Pack SbtRecords 
    void buildSBT();
    // Memory allocation for buffers
    void bufferAlloc();
    //optixLaunch, cudaStreamSynchronize
    void renderFrame();
    // Our image loading from GPU
    void downloadImage(std::vector<uchar4>& host_pixels);

private:
    OptixDeviceContext optix_context = nullptr;
    OptixPipeline      pipeline      = nullptr;
    CUstream           stream        = 0;

    // Create Program Groups (Raygen, Miss, Hitgroup)
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroup programGroups[3]; // We now need 3 programs!

    CUdeviceptr        d_launchParams = 0;
    // Allocate the blank image canvas on the GPU
    uchar4* d_resultBuffer = nullptr;
    OptixShaderBindingTable sbt = {};

    // For simple project we fix the size in this scope
    // May wish to make size change adjustable later
    int width  = 1920;
    int height = 1080;
};
