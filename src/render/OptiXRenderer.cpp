#include "../util/Logger.h"
#include "OptiXRenderer.h"
#include <iostream>

// OptiX requires a specific aligned struct to hold shader headers
    template <typename T>
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data; // Empty for now, used for materials later
    };
    typedef SbtRecord<int> RaygenRecord;

// Create Program Groups (Raygen, Miss, Hitgroup)
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroup programGroups[3]; // We now need 3 programs!

// For simple project we fix the size in this scope
// May wish to make size change adjustable later
int width = 1920;
int height = 1080;

/* CUDA, OptiX initiate and Context create */
OptixRenderer::OptixRenderer() { 
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
    //OptixDeviceContext optix_context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &optix_context));

    std::cout << "[SUCCESS] OptiX Device Context created successfully! The CPU is talking to the GPU.\n";

}

/* Load PTX, Create Programs, Link Pipeline */
void OptixRenderer::buildPipeline() { 
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

    

    // Raygen Program
    OptixProgramGroupDesc raygenDesc    = {};
    raygenDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module            = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__renderFrame";
    OPTIX_CHECK(optixProgramGroupCreate(optix_context, &raygenDesc, 1, &pgOptions, nullptr, nullptr, &programGroups[0]));

    // Miss Program
    OptixProgramGroupDesc missDesc      = {};
    missDesc.kind                       = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module                = module;
    missDesc.miss.entryFunctionName     = "__miss__dummy";
    OPTIX_CHECK(optixProgramGroupCreate(optix_context, &missDesc, 1, &pgOptions, nullptr, nullptr, &programGroups[1]));

    // Hitgroup Program
    OptixProgramGroupDesc hitgroupDesc  = {};
    hitgroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupDesc.hitgroup.moduleCH            = module;
    hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__dummy";
    OPTIX_CHECK(optixProgramGroupCreate(optix_context, &hitgroupDesc, 1, &pgOptions, nullptr, nullptr, &programGroups[2]));

    // Link the Pipeline
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;
    pipelineLinkOptions.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    //OptixPipeline pipeline = nullptr;
    OPTIX_CHECK(optixPipelineCreate(
        optix_context,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups,
        3,             // <--- Tell the pipeline to link all 3 programs
        nullptr, nullptr,
        &pipeline
    ));

    std::cout << "[SUCCESS] OptiX Pipeline Built!\n";
}

/* Pack SbtRecords and cudaMemcpy to VRAM */ 
void OptixRenderer::buildSBT() { 
    // -------------------------------------------------------------------------
    // SHADER BINDING TABLE (SBT)
    // -------------------------------------------------------------------------

    typedef SbtRecord<int> EmptyRecord;

    // Raygen Record
    EmptyRecord rgSbtRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[0], &rgSbtRecord));
    CUdeviceptr d_raygenRecord = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord), sizeof(EmptyRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygenRecord), &rgSbtRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice));

    // Dummy Miss Record
    EmptyRecord msSbtRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[1], &msSbtRecord));
    CUdeviceptr d_missRecord = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missRecord), sizeof(EmptyRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_missRecord), &msSbtRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice));

    // Dummy Hitgroup Record
    EmptyRecord hgSbtRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[2], &hgSbtRecord));
    CUdeviceptr d_hitgroupRecord = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroupRecord), sizeof(EmptyRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroupRecord), &hgSbtRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice));

    // Hand the memory pointers to the OptiX API
    //OptixShaderBindingTable sbt = {};
    
    sbt.raygenRecord                = d_raygenRecord;
    sbt.missRecordBase              = d_missRecord;
    sbt.missRecordStrideInBytes     = sizeof(EmptyRecord);
    sbt.missRecordCount             = 1;
    
    sbt.hitgroupRecordBase          = d_hitgroupRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.hitgroupRecordCount         = 1;

}

void OptixRenderer::bufferAlloc(){
    // -------------------------------------------------------------------------
    // MEMORY ALLOCATION 
    // -------------------------------------------------------------------------
    std::cout << "Allocating memory\n";
    

    // Allocate the blank image canvas on the GPU
    //uchar4* d_resultBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_resultBuffer, width * height * sizeof(uchar4)));

    // Create the LaunchParams on the CPU, and fill it with our data
    LaunchParams hostParams = {};
    hostParams.width = width;
    hostParams.height = height;
    hostParams.resultBuffer = d_resultBuffer;

    // Allocate memory for the LaunchParams on the GPU, and copy it over
    //CUdeviceptr d_launchParams = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launchParams), sizeof(LaunchParams)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_launchParams),
        &hostParams,
        sizeof(LaunchParams),
        cudaMemcpyHostToDevice
    ));
}

/* optixLaunch, cudaStreamSynchronize */ 
void OptixRenderer::renderFrame() { 
    // -------------------------------------------------------------------------
    // LAUNCH THREADS
    // -------------------------------------------------------------------------
    std::cout << "Shooting Rays...\n";
    
    // Create an asynchronous CUDA execution queue
    //CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    OPTIX_CHECK(optixLaunch(
        pipeline,
        stream,
        d_launchParams,
        sizeof(LaunchParams),
        &sbt,
        width,    // Launch width  (X)
        height,   // Launch height (Y)
        1         // Launch depth  (Z - just 1 for flat images)
    ));

    // Force the CPU to wait until the GPU finishes tracing all rays
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
void OptixRenderer::downloadImage(std::vector<uchar4>& host_pixels) { /* cudaMemcpyDeviceToHost */ }

/* optixDeviceContextDestroy, cudaFree */ 
OptixRenderer::~OptixRenderer() { 
    // Clean up memory before exiting
    optixDeviceContextDestroy(optix_context);
    std::cout << "Engine shut down cleanly.\n";
}