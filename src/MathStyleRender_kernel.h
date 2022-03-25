#pragma once

// These all have functionality that must be compiled with nvcc.

extern void InvokeRenderKernel(uchar4* uc4DevPtr, size_t pitch, int wid, int hgt, int GridHgtInPix, int yofs, float BoxWid, float Xmin, float Ymin, float MinSamples, ColorSpace_t ColorSpace);

extern void loadColorMapTexture(cudaArray* ColMapArray);

extern void loadSampleLocsToConstant(const float2* Locs, const int sizeofLocs, const float2* Nums, const int sizeofNums);

extern void loadTokensToConstant(const int* HostTokens, const int sizeofHostTokens);

extern void InvokeTestKernel(float* devFloats, int wid, float v);
