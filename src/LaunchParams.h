#pragma once
#include <cuda_runtime.h>

struct LaunchParams {
    uchar4* resultBuffer;
    int     width;
    int     height;
};