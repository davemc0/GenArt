#pragma once

#include "cuda_runtime.h"

#include <Util/Assert.h>

#ifndef __CUDACC__

#include <iostream>

#define CUDA_SAFE_CALL_NO_SYNC(call)                                                                                    \
    {                                                                                                                   \
        cudaError err = call;                                                                                           \
        if (cudaSuccess != err) {                                                                                       \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw(DMcError(cudaGetErrorString(err)));                                                                   \
        }                                                                                                               \
    }

#define CUDA_SAFE_CALL(call) CUDA_SAFE_CALL_NO_SYNC(call);

void checkCUDAError(const char* msg);

void checkCUDABusy(cudaStream_t& cstream, const char* msg);
void setCUDADevice(int devid);

void getCUDADeviceInfo();
void getCUDAMemInfo();
void finishCUDA();

#endif
