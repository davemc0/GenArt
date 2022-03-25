// The main CUDA rendering kernel for MathStyle

#include "Evaluator.h"

// ColMapTexRef is used in RenderHelpers.h.
texture<float4, 1, cudaReadModeElementType> ColMapTexRef; // Using a 2D texture is a HACK since the ColorMap is 1D. I think I did it to address a CUDA bug.

#include "RenderHelpers.h"

// Put clock count in alpha channel
//#define CLOCKS
// Rotate the Halton sequence. Costs 10% perf. 2 regs.
//#define ROTATION
// If only one sample is rendered, don't jitter it so that lines come out straight. 1 reg.
//#define NO_JITTER_ONE_SAMPLE
// Experiment with a hardcoded equation: 16 regs for main loop + color spaces
// #define HARDCODED
// Making these templated saves 3 regs and speeds it up.
#define TEMPLATED_COLORSPACE

// Also need ColorMap, sample coordinate table, tokenized expressions
__constant__ int ConstantTokenStream[MAX_TOKENS];
__constant__ float2 ConstantSampleLocs[NUM_SAMPLE_LOCS];
__constant__ float2 ConstantRandomNums[NUM_SAMPLE_LOCS]; // Try in global memory since threads access different elements at same time.

// 12..16 main loop + color transform
// 13 regs main loop
// 20 regs main loop + color transform + evaluate
// 20 regs main loop + evaluate

#ifdef TEMPLATED_COLORSPACE
template<int ColorSpace>
__global__
void RenderKernel(uchar4 *uc4DevPtr, int pitch, int wid, int hgt,
                  int yofs, float stepxy, float Xmin, float Ymin, float MinSamples)
#else
// Regs += 3
__global__
void RenderKernel(uchar4 *uc4DevPtr, int pitch,
                  int wid, int hgt, int yofs, float stepxy, float Xmin, float Ymin,
                  float MinSamples, int ColorSpace)
#endif
{
#ifdef CLOCKS
    clock_t clock0 = clock();
#endif

    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y + yofs;
    if(xi >= wid || yi >= hgt) return;

    int index = yi * pitch + xi;
    
    float lxp = (xi+0.5f) * stepxy + Xmin; // Pixel center on image plane
    float lyp = (yi+0.5f) * stepxy + Ymin; // Centered on the half step

#ifdef ROTATION
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // Thread index within thread block

    // Rotate lxp,lyp randomly here
    // These memory reads only happen once per thread and are coherent. 
    int RandInd = tid % NUM_SAMPLE_LOCS;
    float2 SampOfs = ConstantRandomNums[RandInd]; // On 0..1
#endif

    f3Pixel Sum = f3Pixel(0,0,0);

    //for (int NSamples = 0; NSamples < MinSamples; NSamples++) {
    for (int NSamples = MinSamples-1; NSamples >= 0; NSamples--) {
            float2 SampLoc = ConstantSampleLocs[NSamples]; // On -0.5..0.5

#ifdef ROTATION
        float xo = SampLoc.x + SampOfs.x; if(xo>0.5f) xo-=1; // Map -0.5..1.5 to -0.5..0.5
        float yo = SampLoc.y + SampOfs.y; if(yo>0.5f) yo-=1;
#else
        float xo = SampLoc.x;
        float yo = SampLoc.y;
#endif

#ifdef NO_JITTER_ONE_SAMPLE
        if(MinSamples <= 1) xo = yo = 0;
#endif

        float x = lxp + xo * stepxy; // could premultiply table by this.
        float y = lyp + yo * stepxy;
        float r = sqrtf(x*x + y*y);

        float red, grn, blu;

#ifdef HARDCODED
#if 1
        // Rendering this equation at -size 2048 2048 -qual 100 100 100 
        // Hard coded = 0.159s Evaluator = 3.720s CPU 4 core not hard coded  = 112.3s
        // 0.248 s vs. 2.6 s
        red = 0.197278;
        grn = 0.647491 + (eLn(eMod(0.4343 * eLn(eSin(-eExp(eExp(eLn(eSin(eExp(eExp(eSin(x * y))))) + (eSin(2.0723 * x) - eTan(x))))) + (eSin(eExp(eExp(eSin(eSqrt(r))))) + eDiv(eXOr(eMod(-1.99781 * (eMod(ePow(x, 0.725418), eClamp(eSqrt(y)))), ePow(0.156434, eSin(2.02173 * x))), x * y), eExp(eSin(eExp(eExp(eSin(eExp(eExp(eSin(eSqrt(y))))) - y))))))), 0.987066)) + (eDiv(eXOr(eCube(2.45321 - 33.3313 * r), eMod(-2.07658 * (eMod(ePow(x, 0.884499), eClamp(eSqrt(y)))), 1)), eExp(eSin(eExp(eExp(eSin(2.95912 * x) - y))))) + ePow(x - (eSqr(y) + 0.4343 * eLn(0.114559 - (x + y))), x)));
        blu = 0.52584 + (eLn(eMod(0.4343 * eLn(eSin(-eExp(eExp(eLn(eSin(eExp(eExp(eSin(x * y))))) + (eSin(2 * x) - eTan(x))))) + (eSin(eExp(eExp(eSin(eSqrt(r))))) + eDiv(eXOr(eMod(-2.16671 * (eMod(ePow(x, 0.787501), eClamp(eSqrt(y)))), ePow(0.178835, eSin(2.02401 * x))), x * y), eExp(eSin(eExp(eExp(eSin(eExp(eExp(eSin(eSqrt(y))))) - y))))))), 1)) + (eDiv(eXOr(eCube(2.43771 - 33.2563 * r), eMod(-2.16671 * (eMod(ePow(x, 0.787501), eClamp(eSqrt(y)))), ePow(0.178835, eSin(2.02401 * x)))), eExp(eSin(eExp(eExp(eSin(2.88095 * x) - y))))) + ePow(x - (eSqr(y) + 0.4343 * eLn(0.278156 - (x + y))), x)));
#else
        // Experiment to see how many regs Evaluate takes
        // This is basically a noop
        red = float(ConstantTokenStream[int(x+r)]);
        grn = float(ConstantTokenStream[int(x+y)]);
        blu = float(ConstantTokenStream[int(y+r)]);
#endif
#else
        EvaluateTokenized(ConstantTokenStream, x, y, r, 0, red, grn, blu);
#endif

        f3Pixel FinalVal = ColorTransform(red, grn, blu, ColorSpace);
        // Experiment to see how many regs ColorTransform takes
        // If hardcoded, some of these up it from 17 to 19. If not hardcoded, still 23.
        // f3Pixel FinalVal(red, grn, blu);

        Sum.x += FinalVal.x;
        Sum.y += FinalVal.y;
        Sum.z += FinalVal.z;
    }

    float Sc = 255.0f / MinSamples;
    f3Pixel SumSc = f3Pixel(Sum.x * Sc, Sum.y * Sc, Sum.z * Sc);
#ifdef CLOCKS
    clock_t clockspan = clock() - clock0;
    int clk = clockspan >> 17;
    if(clk > 255) clk = 255;
    unsigned char uclk = (unsigned char) clk;
    // uchar4 uc4P = make_uchar4((unsigned char)SumSc.x, (unsigned char)SumSc.y, uclk, (unsigned char)SumSc.z);
    uchar4 uc4P = make_uchar4(uclk, uclk, uclk, 0xff);
#else
    uchar4 uc4P = make_uchar4((unsigned char)SumSc.x, (unsigned char)SumSc.y, (unsigned char)SumSc.z, 0xff);
#endif
    uc4DevPtr[index] = uc4P; // pitch and index are in uc4s.
}

inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void InvokeRenderKernel(uchar4 *uc4DevPtr, size_t pitch, int wid, int hgt, int GridHgtInPix, int yofs,
                        float BoxWid, float Xmin, float Ymin, float MinSamples, ColorSpace_t ColorSpace)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(wid, BLOCKDIM_X), iDivUp(GridHgtInPix, BLOCKDIM_Y));
	int pitchin = static_cast<int>(pitch)/4;

    // If the image is non-square I've chosen to keep the pixels square and map the BoxWid to the width.
    float stepxy = BoxWid / float(wid);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

#ifdef TEMPLATED_COLORSPACE
    switch(ColorSpace) {
    case 0:
        RenderKernel<0><<<grid, threads, 0>>>(uc4DevPtr, pitchin, wid, hgt, yofs,
            stepxy, Xmin, Ymin, MinSamples);
        break;
    case 1:
        RenderKernel<1><<<grid, threads, 0>>>(uc4DevPtr, pitchin, wid, hgt, yofs,
            stepxy, Xmin, Ymin, MinSamples);
        break;
    case 2:
        RenderKernel<2><<<grid, threads, 0>>>(uc4DevPtr, pitchin, wid, hgt, yofs,
            stepxy, Xmin, Ymin, MinSamples);
        break;
    case 3:
        RenderKernel<3><<<grid, threads, 0>>>(uc4DevPtr, pitchin, wid, hgt, yofs,
            stepxy, Xmin, Ymin, MinSamples);
        break;
    case 4:
        RenderKernel<4><<<grid, threads, 0>>>(uc4DevPtr, pitchin, wid, hgt, yofs,
            stepxy, Xmin, Ymin, MinSamples);
        break;
    case 5:
        RenderKernel<5><<<grid, threads, 0>>>(uc4DevPtr, pitchin, wid, hgt, yofs,
            stepxy, Xmin, Ymin, MinSamples);
        break;
    }
#else
    RenderKernel<<<grid, threads, 0>>>(uc4DevPtr, pitchin, wid, hgt, yofs,
        stepxy, Xmin, Ymin, MinSamples, ColorSpace);
#endif
}

void loadColorMapTexture(cudaArray* ColMapArray)
{
    ColMapTexRef.normalized = true;
    ColMapTexRef.filterMode = cudaFilterModeLinear;
    ColMapTexRef.addressMode[0] = cudaAddressModeClamp;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    ColMapTexRef.channelDesc = channelDesc;

    // Bind the array to the texture
    cudaBindTextureToArray(ColMapTexRef, ColMapArray, channelDesc);
}

void loadTokensToConstant(const int *HostTokens, const int sizeofHostTokens)
{
    cudaMemcpyToSymbol(ConstantTokenStream, HostTokens, sizeofHostTokens);
}

void loadSampleLocsToConstant(const float2 *Locs, const int sizeofLocs, const float2 *Nums, const int sizeofNums)
{
    cudaMemcpyToSymbol(ConstantSampleLocs, Locs, sizeofLocs);
    cudaMemcpyToSymbol(ConstantRandomNums, Nums, sizeofNums);
}

///////////////////////////////////////////////////////////
// Test code

#if 0
__global__ 
void TestKernel(float *fDevPtr, int wid, float v)
{
   // fDevPtr[0] = roundf(v);
   // fDevPtr[1] = rintf(v);
   // fDevPtr[2] = floorf(v);
   // fDevPtr[3] = ceilf(v);
   // fDevPtr[4] = truncf(v);
   // fDevPtr[5] = float(int(v));
   // fDevPtr[6] = roundf(-v);
   // fDevPtr[7] = rintf(-v);
   // fDevPtr[8] = floorf(-v);
   // fDevPtr[9] = ceilf(-v);
   // fDevPtr[10]= truncf(-v);
   // fDevPtr[11]= float(int(-v));
}
#endif

void InvokeTestKernel(float *devFloats, int wid, float v)
{
    dim3 threads(BLOCKDIM_X);
    dim3 grid((wid+BLOCKDIM_X-1) / BLOCKDIM_X);

  //  TestKernel<<<grid, threads>>>(devFloats, wid, v);
}
