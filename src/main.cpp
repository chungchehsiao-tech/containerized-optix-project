#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <fstream>
#include <sstream>
#include <vector>

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
static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

// -----------------------------------------------------------------------------
// MAIN PROGRAM
// -----------------------------------------------------------------------------
int main() {
    std::cout << "Starting OptiX Engine Initialization...\n";

    // Step 1: Initialize the CUDA Runtime
    // Call cudaFree(0) to force the CUDA driver wake up and build a context on the default GPU.
    CUDA_CHECK(cudaFree(0));
    std::cout << "[SUCCESS] CUDA Runtime initialized.\n";

    // Step 2: Initialize OptiX function pointers
    // Dynamically loads the OptiX API from the physical display driver.
    OPTIX_CHECK(optixInit());
    std::cout << "[SUCCESS] OptiX Driver loaded.\n";

    // Step 3: Create the OptiX Device Context
    // Main handle, will use later to create buffers and pipelines.
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel    = 4; // 4 captures everything up to standard warnings
    
    CUcontext cu_ctx = 0; // Using the primary CUDA context
    OptixDeviceContext optix_context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &optix_context));

    std::cout << "[SUCCESS] OptiX Device Context created successfully! The CPU is talking to the GPU.\n";

    std::cout << "Building OptiX Pipeline...\n";

    //Setup Pipeline Options
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur        = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues      = 2;
    pipelineCompileOptions.numAttributeValues    = 2;
    pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_DEBUG; 
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    // Setup Module Options
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    // Read PTX File 
    std::string ptx_path = std::string(PTX_DIR) + "/shader.ptx";
    std::ifstream ptx_file(ptx_path);

    if (!ptx_file.good()) {
        std::cerr << "ERROR: Could not open PTX file. Did CMake build it?\n";
        exit(1);
    }

    std::stringstream ptx_buffer;
    ptx_buffer << ptx_file.rdbuf();
    std::string ptx_source = ptx_buffer.str();

    // Create the Module 
    OptixModule module = nullptr;
    OPTIX_CHECK(optixModuleCreateFromPTX(
        optix_context,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptx_source.c_str(),
        ptx_source.size(),
        nullptr, nullptr,
        &module
    ));

    //Create Program Group (Raygen)
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module            = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    OptixProgramGroup raygenProgramGroup = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context,
        &raygenDesc,
        1,             // Number of program groups
        &pgOptions,
        nullptr, nullptr,
        &raygenProgramGroup
    ));

    //Link the Pipeline
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;
    pipelineLinkOptions.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OptixPipeline pipeline = nullptr;
    OPTIX_CHECK(optixPipelineCreate(
        optix_context,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        &raygenProgramGroup,
        1,             // Number of program groups
        nullptr, nullptr,
        &pipeline
    ));

    std::cout << "[SUCCESS] OptiX Pipeline Built!\n";


    // Clean up memory before exiting
    optixDeviceContextDestroy(optix_context);
    std::cout << "Engine shut down cleanly.\n";

    return 0;
}