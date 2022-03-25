#include "CUDAHelpers.h"

#include "GL/glew.h"
#include "cuda_runtime.h"

#include <cstdio>

void setCUDADevice(int devid)
{
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    checkCUDAError("cudaGetDeviceCount");

    ASSERT_R(devid < deviceCount);

    cudaSetDevice(devid);

    checkCUDAError("cudaSetDevice");
}

void checkCUDAError(const char* msg)
{
    cudaError_t Err = cudaGetLastError();
    if (cudaSuccess != Err) {
        fprintf(stderr, "CUDA error: %s: %s.\n", msg, cudaGetErrorString(Err));
        exit(EXIT_FAILURE);
    }
}

void checkCUDABusy(cudaStream_t& cstream, const char* msg)
{
    cudaError_t Err = cudaStreamQuery(cstream);
    if (Err == cudaErrorNotReady)
        std::cerr << msg << " busy\n";
    else if (Err != cudaSuccess)
        checkCUDAError(msg);
    else
        std::cerr << msg << " done\n";
}

void getCUDAMemInfo()
{
    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);

    std::cerr << "CUDA Free Memory = " << mem_free << " CUDA Total Memory = " << mem_total << '\n';
}

void getCUDADeviceInfo()
{
    int i, n, dv, rv;
    const char* suffix = "B";

    printf("\nCUDA Device Info\n");
    cudaError_t a = cudaDriverGetVersion(&dv);
    cudaError_t b = cudaRuntimeGetVersion(&rv);

    printf("Driver version=%d\nRuntime version=%d\n", dv, rv);

    cudaError_t goe = cudaGetDeviceCount(&n);

    printf("There are %d cuda devices\n", n);

    for (i = 0; i < n; i++) {
        printf("\nCUDA device %d:\n", i);
        cudaDeviceProp prop;

        cudaError_t joe = cudaGetDeviceProperties(&prop, i);
        if (prop.totalGlobalMem > 10 * 1024 * 1024) {
            prop.totalGlobalMem /= 1024 * 1024;
            suffix = "MB";
        }

        printf("char name[256]=\"%s\";\n"
               "size_t totalGlobalMem=%lld %s;\n"
               "size_t sharedMemPerBlock=%lld;\n"
               "int regsPerBlock=%d;\n"
               "int warpSize=%d;\n"
               "size_t memPitch=%lld;\n"
               "int maxThreadsPerBlock=%d;\n"
               "int maxThreadsDim[3]=[%d %d %d];\n"
               "int maxGridSize[3]=[%d %d %d];\n"
               "int clockRate=%d;\n"
               "size_t totalConstMem=%lld;\n"
               "int major=%d;\n"
               "int minor=%d;\n"
               "size_t textureAlignment=%lld;\n"
               "int deviceOverlap=%d;\n"
               "int multiProcessorCount=%d\n"
               "int kernelExecTimeoutEnabled=%d\n"
               "int integrated=%d\n"
               "int canMapHostMemory=%d\n"
               "int computeMode=%d\n"
               "int maxTexture1D=%d\n"
               "int maxTexture2D[2]=[%d %d]\n"
               "int maxTexture3D[3]=[%d %d %d]\n"
               "int maxTexture1DLayered[2]=[%d %d]\n"
               "int maxTexture2DLayered[3][%d %d %d]\n"
               "size_t surfaceAlignment=%lld\n"
               "int concurrentKernels=%d\n"
               "int ECCEnabled=%d\n"
               "int pciBusID=%d\n"
               "int pciDeviceID=%d\n"
               "int tccDriver=%d\n"
               "int asyncEngineCount=%d\n"
               "int unifiedAddressing=%d\n"
               "int memoryClockRate=%d\n"
               "int memoryBusWidth=%d\n"
               "int l2CacheSize=%d\n"
               "int maxThreadsPerMultiProcessor=%d\n",
               prop.name, prop.totalGlobalMem, suffix, prop.sharedMemPerBlock, prop.regsPerBlock, prop.warpSize, prop.memPitch, prop.maxThreadsPerBlock,
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2],
               prop.clockRate, prop.totalConstMem, prop.major, prop.minor, prop.textureAlignment, prop.deviceOverlap, prop.multiProcessorCount,
               prop.kernelExecTimeoutEnabled, prop.integrated, prop.canMapHostMemory, prop.computeMode, prop.maxTexture1D, prop.maxTexture2D[0],
               prop.maxTexture2D[1], prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2], prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1],
               prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2], prop.surfaceAlignment, prop.concurrentKernels,
               prop.ECCEnabled, prop.pciBusID, prop.pciDeviceID, prop.tccDriver, prop.asyncEngineCount, prop.unifiedAddressing, prop.memoryClockRate,
               prop.memoryBusWidth, prop.l2CacheSize, prop.maxThreadsPerMultiProcessor);
    }
}

void finishCUDA()
{
    // Must call cudaDeviceReset before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();
}
