#include "MathStyleCUDARender.h"

#include "CUDAHelpers.h"
#include "ExprImplementations.h"
#include "ExprTools.h"
#include "Math/Halton.h"
#include "Math/MiscMath.h"
#include "Math/Random.h"
#include "MathIndividual.h"
#include "MathStyleRender_kernel.h"

#include <iostream>

CUDARender::CUDARender(const int deviceId_) : m_deviceId(deviceId_) { setCUDADevice(m_deviceId); }

MathStyleCUDARender::MathStyleCUDARender(const int deviceId_) : CUDARender(deviceId_)
{
    m_totalElapsedTime = 0;
    m_deviceOpsPerMS = computeDeviceOpsPerMS();

    cudaChannelFormatDesc ColMapChannelDesc = cudaCreateChannelDesc<float4>();
    CUDA_SAFE_CALL(cudaMallocArray(&m_ColorMapCUDAArray, &ColMapChannelDesc, MAX_COLORMAP_ENTRIES, 1));
    checkCUDAError("MathStyleCUDARenderEngine");

    InitSampleTable();
}

MathStyleCUDARender::~MathStyleCUDARender()
{
    freeArray(); // Don't pull this function implementation in here because it throws.
}

void MathStyleCUDARender::Render(Individual* ind_, uc4DImage* Im, const int w_, const int h_, const Quality_t Q_)
{
    MathIndividual* ind(dynamic_cast<MathIndividual*>(ind_));
    ASSERT_R(ind);

    checkCUDAError("MathStyleCUDARender::Render");

    ASSERT_R(ind->G);
    ASSERT_R(w_ > 0 && h_ > 0);

    // std::cerr << "MathStyleCUDARender::Render: " << ind->IDNum << ' ' << w_ << 'x' << h_ << " Quality=" << Q_ << std::endl;

    Im->SetSize(w_, h_);

    checkCUDAError("Here");

    ////////////////////////////////////////////
    // Set up the ColorMap
    ColorMap<f3Pixel> CMapFullSize(ind->CMap, MAX_COLORMAP_ENTRIES);

    float4 TmpCMap[MAX_COLORMAP_ENTRIES];
    for (size_t i = 0; i < CMapFullSize.size(); i++) {
        TmpCMap[i].x = CMapFullSize[i].r();
        TmpCMap[i].y = CMapFullSize[i].g();
        TmpCMap[i].z = CMapFullSize[i].b();
    }

    CUDA_SAFE_CALL(cudaMemcpyToArray(m_ColorMapCUDAArray, 0, 0, TmpCMap, sizeof(TmpCMap), cudaMemcpyHostToDevice));
    m_ColorMapTexObj = loadColorMapTexture(m_ColorMapCUDAArray);

    ////////////////////////////////////////////
    // Set up the tokenized expressions
    int HostTokens[MAX_TOKENS];
    int TokenCnt = 1; // Leave slot 0 for the length
    TokenCnt += Tokenize(ind->R, HostTokens + TokenCnt, MAX_TOKENS - TokenCnt);
    TokenCnt += Tokenize(ind->G, HostTokens + TokenCnt, MAX_TOKENS - TokenCnt);
    TokenCnt += Tokenize(ind->B, HostTokens + TokenCnt, MAX_TOKENS - TokenCnt);
    HostTokens[0] = TokenCnt;

    loadTokensToConstant(HostTokens, sizeof(HostTokens));

    ////////////////////////////////////////////
    // Invoke the kernel and read back the result

    ASSERT_R(Q_.MinSamples <= SampleTable_t::TABLE_SIZE);

    int GridHgtInPix = getGridHeight(w_, h_, WorkEstimate(HostTokens, TokenCnt), Q_.MinSamples);

    uchar4* uc4DevPtr = static_cast<uchar4*>(Im->map());

    Im->renderTimerStart();

    // std::cerr << "GridHgtInPix: " << GridHgtInPix << std::endl;

    // Launch the render as an async series of kernel launches to allow user interaction during long renders
    int grids = 0;
    for (int yofs = 0; yofs < h_; yofs += GridHgtInPix) {
        InvokeRenderKernel(uc4DevPtr, Im->Pitch(), w_, h_, GridHgtInPix, yofs, ind->BoxWid, ind->Xmin, ind->Ymin, Q_.MinSamples, m_ColorMapTexObj,
                           ind->ColorSpace);
        checkCUDAError("InvokeRenderKernel");
        // std::cerr << '.';
        grids++;
    }
    // std::cerr << "grids = " << grids << '\n';

    Im->renderTimerStop(); // Doesn't block.

    Im->unmap();
}

void MathStyleCUDARender::InitSampleTable()
{
    SampleTable_t SampTab;
    SampTab.ScaleHalton(1, 1); // Return values on -0.5 .. 0.5.

    // A table of random offsets to add to the table of sample locations
    ASSERT_R(SampTab.TABLE_SIZE == NUM_SAMPLE_LOCS);
    float RandTable[NUM_SAMPLE_LOCS * 2];
    for (int i = 0; i < NUM_SAMPLE_LOCS * 2; i++) RandTable[i] = frand(0, 1);

    ASSERT_R(sizeof(RandTable) == sizeof(SampTab.tab));
    ASSERT_R(sizeof(RandTable) == sizeof(float2) * NUM_SAMPLE_LOCS);

    checkCUDAError("InitSampleTable0");

    loadSampleLocsToConstant((const float2*)SampTab.tab, sizeof(SampTab.tab), (const float2*)RandTable, sizeof(RandTable));
    checkCUDAError("InitSampleTable4");
}

int MathStyleCUDARender::WorkEstimate(int* HostTokens, int TokenCount)
{
    int TokensInIFS = 0, OtherTokens = 0;
    for (int i = 0; i < TokenCount; i++) {
        if ((HostTokens[i] & 0xff) == IFS_e) {
            int cntL = (HostTokens[i] >> 24) & 0xff;
            int cntR = (HostTokens[i] >> 16) & 0xff;

            TokensInIFS += cntL + cntR;
        }

        if ((HostTokens[i] & 0xff) != Var_e && (HostTokens[i] & 0xff) != Const_e) OtherTokens++;
    }

    // The 8 accounts for some overhead.
    int Count = 8 + OtherTokens + TokensInIFS * 10; // Three channels, IFS is most of the channel; average of 10 IFS iterations
    return Count;
}

float MathStyleCUDARender::computeDeviceOpsPerMS()
{
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, CUDARender::m_deviceId));
    checkCUDAError("cudaGetDeviceProperties");

    int clocksPerMS = deviceProp.clockRate > 0 ? deviceProp.clockRate : 1000000; // For AModel
    float scaleFactorSM = deviceProp.multiProcessorCount > 8 ? 2.0f : 1.0f;      // Fast SM?
    float clocksPerOp = 5.0f;
    float devOpsPerMS = float(clocksPerMS) * float(deviceProp.multiProcessorCount) * scaleFactorSM / clocksPerOp;

    std::cerr << "DeviceOpsPerMS = " << devOpsPerMS << " = clockRate = " << clocksPerMS << " multiProcCount = " << deviceProp.multiProcessorCount
              << " scaleFactorSM = " << scaleFactorSM << " / clocksPerOp = " << clocksPerOp << std::endl;

    return devOpsPerMS;
}

void MathStyleCUDARender::testGridHeight()
{
    while (1) {
        int w = randn(2000) + 100;
        int h = randn(2000) + 100;

        float estOpsPerSample = randn(1000) + 10;
        int minSamples = randn(20);

        int GridHgtInPix = getGridHeight(w, h, estOpsPerSample, minSamples);

        int waste = 0;
        int Grids = (h + GridHgtInPix - 1) / GridHgtInPix;
        int OGrids = Grids;

        do {
            int full = h / GridHgtInPix;
            int rem = h % GridHgtInPix;

            waste = rem ? (GridHgtInPix - rem) : 0;

            std::cerr << GridHgtInPix << "*" << full << "+" << rem << "=" << h << " waste = " << waste
                      << (waste > Grids * BLOCKDIM_Y ? " vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n" : "\n");

            GridHgtInPix -= BLOCKDIM_Y;
            Grids = GridHgtInPix ? (h + GridHgtInPix - 1) / GridHgtInPix : 99999;
        } while (waste >= 0 && GridHgtInPix > 0 && Grids == OGrids);
    }
}

int MathStyleCUDARender::getGridHeight(int w, int h, float estOpsPerSample, int minSamples)
{
    // return h; // Uncomment to force one launch per image

    float TargetMSPerGrid = 1000; // Based on two second timeout
    float TargetOpsPerGrid = m_deviceOpsPerMS * TargetMSPerGrid;

    float EstSamplesPerPixel = minSamples;
    float EstOpsPerPixel = EstSamplesPerPixel * estOpsPerSample;

    float TargetPixelsPerGrid = TargetOpsPerGrid / EstOpsPerPixel;

    float Grids = (w * h) / TargetPixelsPerGrid;
    float HgtInBlocks = h / (float)BLOCKDIM_Y;
    float GridHgtInBlocks = HgtInBlocks / Grids;
    int GridHgtInPix = 0;

    if (Grids < 5.0f) { // We really care about the correct number of grids to avoid timeouts.
        int iGrids = (int)(Grids + 0.5f);
        if (iGrids < 1) iGrids = 1;
        float GridHgtInBlocks = HgtInBlocks / (float)iGrids;
        int iGridHgtInBlocks = (int)GridHgtInBlocks; // Round down
        GridHgtInPix = iGridHgtInBlocks * BLOCKDIM_Y;
        while (iGrids * GridHgtInPix < h) GridHgtInPix += BLOCKDIM_Y; // Round up if necessary
        ASSERT_R(iGrids * GridHgtInPix >= h);
    } else { // Don't care so much about the correct number of grids. Try to save work. Round GridHgtInBlocks up so grids goes down.
        float GridHgtInBlocks = HgtInBlocks / Grids;
        int iGridHgtInBlocks = (int)(GridHgtInBlocks + 0.99f); // Round up
        GridHgtInPix = iGridHgtInBlocks * BLOCKDIM_Y;
        ASSERT_R(GridHgtInPix > 0);
        int iGrids = (h + GridHgtInPix - 1) / GridHgtInPix;
        ASSERT_R(iGrids * GridHgtInPix >= h);
        while (iGrids * GridHgtInPix - h >= iGrids * BLOCKDIM_Y) GridHgtInPix -= BLOCKDIM_Y; // Round down if possible
    }

    return GridHgtInPix;
}

void MathStyleCUDARender::freeArray()
{
    CUDA_SAFE_CALL(cudaFreeArray(m_ColorMapCUDAArray));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(m_ColorMapTexObj));
}
