/////////////////////////////////////////////////////////////////////////////////////////////////
// Some test code

#include "CUDAHelpers.h"
#include "ExprImplementations.h"
#include "ExprTools.h"
#include "IntervalImplementations.h"
#include "Math/Halton.h"
#include "Math/Random.h"
#include "MathIndividual.h"
#include "NonaryExprSubclasses.h"

#include <Util/Timer.h>

////////////////////////////////////////////////////////////////////////////////
// Test CUDA Kernel

void TestKernel()
{
    std::cerr << "uo1\n";

    std::cerr << "uo2\n";
    float* devFloats = NULL;
    const int NFLOATS = 12;
    float HostFloats[NFLOATS];
    std::cerr << "uo3\n";

    checkCUDAError("TestKernel0");
    printf("devFloats=%p\n", devFloats);
    printf("&devFloats=%p\n", &devFloats);
    cudaMalloc((void**)&devFloats, NFLOATS * sizeof(float));
    printf("devFloats=%p\n", devFloats);
    checkCUDAError("TestKernel1");
    std::cerr << "uo4\n";
    // Can't declare this in an anonymous namespace.
    extern void InvokeTestKernel(float* devFloats, int wid, float v);
    InvokeTestKernel(devFloats, NFLOATS, 1.3f);
    std::cerr << "uo5\n";
    checkCUDAError("TestKernel2");
    std::cerr << "uo6\n";

    cudaMemcpy(HostFloats, devFloats, NFLOATS * sizeof(float), cudaMemcpyDeviceToHost);
    std::cerr << "uo7\n";

    for (int i = 0; i < NFLOATS; i++) std::cerr << HostFloats[i] << ",";
    std::cerr << std::endl;
    std::cerr << "uo8\n";

    checkCUDAError("TestKernel3");
}

namespace {
////////////////////////////////////////////////////////////////////////////////
// Test the app infrastructure

void TestImageStuff()
{
    ASSERT_RM(0, "Renovate this");
    // float score = ImgCompare(AnImage, TestImg);
    // float score = ImgHistScore(EvolveToImg);
    // std::cerr << "Score = " << score << std::endl;
}

// An IndivID is based on the app start time in seconds plus a serial number.
void TestCreateIndivID()
{
    while (1) {
        // std::cerr << CreateIndivID() << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Test Math Routines

// What if I do a more braindead mod?
void TestMod()
{
    int i = 0;
    std::cerr << "Testing mod accuracy.\n";
    while (1) {
        float x = DRandf(-100.0f, 100.0f);
        float y = DRandf(-100.0f, 100.0f);
        float res = fmod(x, y);
        float rese = eMod(x, y);
        if (fabsf((res - rese) / res) > 0.00001f && (fabs(res - rese) > 0.00001f)) {
            printf("%d: %f,%f: %f != %f\n", i, x, y, res, rese);
            // float res = fmod(x, y);
            // float rese = eMod(x, y);
        }
        i++;
    }
}

// What if I do a more braindead mod?
void TimeMod()
{
    Timer T;
    {
        T.Reset();
        T.Start();
        float x = 1123.545342f;
        float y = 12325.25f;
        float z = 0;
        for (int i = 0; i < 1000000000; i++) { z += fmod(x, y); }
        T.Stop();
        std::cerr << z << "fmod: " << T.Read() << std::endl;
    }

    {
        T.Reset();
        T.Start();
        float x = 1123.545342f;
        float y = 12325.25f;
        float z = 0;
        for (int i = 0; i < 1000000000; i++) { z += eMod(x, y); }
        T.Stop();
        std::cerr << z << "tmod: " << T.Read() << std::endl;
    }
}

inline f3Pixel exYCrCbtoRGB(const f3Pixel& YCrCb)
{
    float r = YCrCb[1] + YCrCb[0];
    float b = YCrCb[2] + YCrCb[0];
    float g = (YCrCb[0] - 0.299f * r + 0.114f * b) * 1.7035775f;

    return f3Pixel(r, g, b);
}

// MINMAX RGB if YCrCb is 0..1: [0.0157301, -0.463345, 0.0117844][1.99501, 1.55366, 1.98428]
void TestColorSpace()
{
    f3Pixel Mn(0.5), Mx(0.5);
    while (1) {
        f3Pixel RndYCrCb(DRandf(), DRandf(), DRandf());
        f3Pixel RndRGB(exYCrCbtoRGB(RndYCrCb));
        std::cerr << RndYCrCb << " -> " << RndRGB << std::endl;
        Mn = Min(RndRGB, Mn);
        Mx = Max(RndRGB, Mx);
        std::cerr << "MINMAX: " << Mn << Mx << std::endl;
    }
}

// Test the Hammersley and Halton sequences
void TestSamplingSequence()
{
    for (int N = 1; N < 100; N++) {
        std::cerr << N << std::endl;
        for (int i = 0; i < N; i++) {
            // vector<float> H = Hammersley(i, 2, N);
            std::vector<float> H = Halton(i, 2);

            for (size_t p = 0; p < H.size(); p++) std::cerr << H[p] << ", ";
            std::cerr << std::endl;
        }
    }
    exit(0);
}

// Is this atan2 accurate enough?
void TestFastATan2()
{
    std::cerr << "Testing transcendental accuracy.\n";
    int total = 0, escapes = 0;
    while (1) {
        float x = DRandf(-100.0f, 100.0f);
        float y = DRandf(-100.0f, 100.0f);
        float at2 = atan2f(y, x);
        float eat2 = eATan2(y, x);
        if (fabsf((at2 - eat2) / at2) > 0.57f) {
            printf("%f,%f: %f != %f (%d/%d)\n", x, y, at2, eat2, escapes, total);
            escapes++;
        }
        total++;
    }
}

// Why do I do -sin(-fmod(x)) instead of sin(x)?
// relerr < 0.0001: 24641
// relerr < 0.001: 202277
// relerr < 0.01: 1034621049
// relerr < 0.1: 524065342
// relerr < 0.5: 8216525
void TestFastSine()
{
    const int NBINS = 6;
    int Hist[NBINS];
    for (int i = 0; i < 6; i++) Hist[i] = 0;

    std::cerr << "Testing transcendental accuracy.\n";
    int total = 0;
    while (1) {
        float x = DRandf(-100.0f, 100.0f);
        float tanx = atanf(x);
        float etanx = eATan(x);

        float relerr = fabsf((tanx - etanx) / tanx);

        if (relerr < 0.0001f)
            Hist[0]++;
        else if (relerr < 0.001f)
            Hist[1]++;
        else if (relerr < 0.01f)
            Hist[2]++;
        else if (relerr < 0.1f)
            Hist[3]++;
        else if (relerr < 0.5f)
            Hist[4]++;
        else
            Hist[5]++;

        if (relerr > 0.5f) { printf("%f: %f != %f \t{%d, %d, %d, %d, %d %d}\n", x, tanx, etanx, Hist[0], Hist[1], Hist[2], Hist[3], Hist[4], Hist[5]); }
    }
}

void TestExprCanBeDeleted()
{
    VarVals_t VV;
    InitVVals(VV);
    Expr* Jo(RandExpr(12, &VV));

    delete Jo;
}

void TestMathIndividualCanBeCreatedAndDeleted()
{
    MathIndividual* ind(new MathIndividual("+ 1 x", "- 0 y", "* x y", NULL, SPACE_RGB, .1f, 12345, 1, -1, -1, 0, 0, 1));
}

void TestFindLeadingOnes()
{
    while (1) {
        unsigned int v = 0x7fff; // & LRand();
        printf("0x%x 0x%x\n", v, FindLeadingOnes(v));
    }
}

void TestToIntToFloat()
{
    while (1) {
        float f = DRandf();
        unsigned int i = ToInt(f);
        float ff = ToFloat(i);
        unsigned int ii = ToInt(ff);
        if (i != ii) printf("0x%x 0x%x\n", i, ii);
    }
}

void TestIntervalImpl()
{
    float inf = std::numeric_limits<float>::infinity();
    // interval lv(-inf, 0);
    // interval rv(0, inf);
    // interval lv(-83.773,-0.0f);
    // interval rv(-0.856163,0.0f);
    // interval lv(-3.14159,3.14159);
    // interval rv(0,2);
    interval lv(0, 0);
    interval rv(0, 0);

    std::cerr << "iAbs       " << tostring(iAbs(lv)) << '\n';
    std::cerr << "iACos      " << tostring(iACos(lv)) << '\n';
    std::cerr << "iASin      " << tostring(iASin(lv)) << '\n';
    std::cerr << "iATan      " << tostring(iATan(lv)) << '\n';
    std::cerr << "iBitNot    " << tostring(iBitNot(lv)) << '\n';
    std::cerr << "iCbrt      " << tostring(iCbrt(lv)) << '\n';
    std::cerr << "iClamp     " << tostring(iClamp(lv)) << '\n';
    std::cerr << "iCos       " << tostring(iCos(lv)) << '\n';
    std::cerr << "iCube      " << tostring(iCube(lv)) << '\n';
    std::cerr << "iExp       " << tostring(iExp(lv)) << '\n';
    std::cerr << "iLn        " << tostring(iLn(lv)) << '\n';
    std::cerr << "iRound     " << tostring(iRound(lv)) << '\n';
    std::cerr << "iSin       " << tostring(iSin(lv)) << '\n';
    std::cerr << "iSqr       " << tostring(iSqr(lv)) << '\n';
    std::cerr << "iSqrt      " << tostring(iSqrt(lv)) << '\n';
    std::cerr << "iTan       " << tostring(iTan(lv)) << '\n';
    std::cerr << "iUnaryMinus" << tostring(iUnaryMinus(lv)) << '\n';

    std::cerr << "iATan2     " << tostring(iATan2(lv, rv)) << '\n';
    std::cerr << "iAnd       " << tostring(iAnd(lv, rv)) << '\n';
    std::cerr << "iDiv       " << tostring(iDiv(lv, rv)) << '\n';
    std::cerr << "iMinus     " << tostring(iMinus(lv, rv)) << '\n';
    std::cerr << "iMod       " << tostring(iMod(lv, rv)) << '\n';
    std::cerr << "iMult      " << tostring(iMult(lv, rv)) << '\n';
    std::cerr << "iOr        " << tostring(iOr(lv, rv)) << '\n';
    std::cerr << "iPlus      " << tostring(iPlus(lv, rv)) << '\n';
    std::cerr << "iPow       " << tostring(iPow(lv, rv)) << '\n';
    std::cerr << "iXOr       " << tostring(iXOr(lv, rv)) << '\n';
}

float randVal()
{
    union
    {
        float f;
        unsigned int i;
    } u;

    u.i = {(unsigned int)LRand()};

    if (IsNaN(u.f)) return randVal();

    switch (randn(16)) {
    case 0: return std::numeric_limits<float>::infinity();
    case 1: return 0.0f;
    case 2: return 1.0f;
    case 3: return 2.0f;
    case 4: return E_PI;
    case 5: return DRandf();
    case 6: return DRandf(1.0f, 1000.0f);
    case 7: return DRandf(1000.0f, 1e9f);
    case 8: return -std::numeric_limits<float>::infinity();
    case 9: return -0.0f;
    case 10: return -1.0f;
    case 11: return -2.0f;
    case 12: return -E_PI;
    case 13: return -DRandf();
    case 14: return -DRandf(1.0f, 1000.0f);
    default: return -DRandf(1000.0f, 1e9f);
    }
}

float relerror(float c1, float c0)
{
    float abserr = fabsf(c1 - c0);
    float denom = std::max(fabsf(c0), fabsf(c1));
    float d = fabsf(abserr / denom);

    return d;
}

void TestFastPowF()
{
    while (1) {
        float a = DRand(0, 100); // randVal();
        float b = DRand(-10, 10);

        float c0 = pow(a, b);
        float c1 = mypowf(a, b);
        float c2 = ePow(a, b);
        float d = relerror(c1, c0);

        printf("%15.5e %15.5e %15.5e \t%f,%f \t%f\n", c0, c1, c2, a, b, d);
    }
}

float fastPowOpt(float a, float b, int v)
{
    union
    {
        float f;
        int x;
    } u = {a};

    int t = u.x - v;
    float bt = b * t;
    int bti = (int)bt;
    u.x = bti + v;

    return u.f;
}

void TestFastPowOpt()
{
    double vsum = 0;
    double vcount = 0;

    while (1) {
        float a = -DRandf();
        float b = 2.0f;
        float c0 = powf(a, b);

        if (c0 > (float)0x7fffffff || c0 == 0.0f) continue;

        int bestv = 0;
        float bestd = std::numeric_limits<float>::max();
        float bestc = 0;
        for (long long int vl = 0; vl <= 2000000000; vl += 1) {
            int v = (int)vl;
            float c1 = fastPowOpt(a, b, v);

            float abserr = fabsf(c1 - c0);
            float denom = std::max(fabsf(c0), fabsf(c1));
            float d = fabsf(abserr / denom);

            if (c1 > 0 && d < bestd) {
                bestd = d;
                bestv = v;
                bestc = c1;
                printf("%f =? %f %f,%f %d %f\n", c0, bestc, a, b, bestv, bestd);
            }
        }

        vsum += bestv;
        vcount++;
        int vavg = (int)(vsum / vcount);

        printf("%16.5f =? %16.5f %9.5f,%9.5f \t%09d %09d \t%8.5f\n", c0, bestc, a, b, bestv, vavg, bestd);
    }
}
}; // namespace

void Test()
{
    std::cerr << "Test\n";
    // TestFastPowOpt();
    // TestFastPow();
    // TestFastPowF();
    TestFastATan2();
    exit(0);
    TestIntervalImpl();
    TestFindLeadingOnes();
    TestToIntToFloat();
    TestExprCanBeDeleted();
    TestMathIndividualCanBeCreatedAndDeleted();
    TestFastSine();
    TestKernel();
    TimeMod();
    TestMod();
    TestSamplingSequence();
    TestCreateIndivID();
    TestColorSpace();
}
