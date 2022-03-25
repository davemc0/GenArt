// Use CUDA to Render an individual

#pragma once

#include "CUDAHelpers.h"
#include "CUDARender.h"

class Individual;

const int MAX_COLORMAP_ENTRIES = 64;

class MathStyleCUDARender : public CUDARender
  {
public:
    MathStyleCUDARender(const int deviceId_);

    ~MathStyleCUDARender();

    // Render an image of this individual and store it in the Individual
    // Can only have one host thread call this since each host thread is a different CUDA context.
    // Must have an upgrade lock on Pop, so that it can get an exclusive lock when needed.
    virtual void Render(Individual* ind_, uc4DImage* Im, const int w_, const int h_, const Quality_t Q_);

private:
    cudaArray* m_ColorMapCUDAArray;
    float m_totalElapsedTime;
    float m_deviceOpsPerMS;

    // Set up the sample table
    void InitSampleTable();

    int WorkEstimate(int *HostTokens, int TokenCount);

    float computeDeviceOpsPerMS();

    int getGridHeight(int w, int h, float estOpsPerSample, int minSamples);
    void testGridHeight();
};
