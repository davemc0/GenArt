#pragma once

#ifdef __CUDACC__

#define DMC_LOC __device__ static
#define DMC_SH_LOC __shared__
#define DMC_VARLOC __device__
#define DMC_RESTRICT __restrict__

#else

#include <cmath>
#include <float.h>

#ifndef DMC_LOC
#define DMC_LOC inline
#define DMC_SH_LOC
#define DMC_VARLOC
#define DMC_RESTRICT
#endif

#endif

// This caused precision problems as a #define. This way is better.
DMC_VARLOC const float E_E = 2.71828182845904523536028747135f;
DMC_VARLOC const float E_PI = 3.141592653589793238462f;

namespace {

DMC_LOC float Clampf(const float minv, const float d, const float maxv)
{
#ifdef __CUDACC__
    return fminf(maxv, fmaxf(d, minv));
#else
    return d <= minv ? minv : (d >= maxv ? maxv : d);
#endif
}

DMC_LOC float Saturate(const float d)
{
#ifdef __CUDACC__
    return __saturatef(d);
#else
    return d <= 0 ? 0 : (d >= 1 ? 1 : d);
#endif
}

DMC_LOC int IsNaN(const float d)
{
#ifdef __CUDACC__
    return isnan(d);
#else
#ifdef _WIN32
    return _isnan(d);
#else
    return isnan(d);
#endif
#endif
}

DMC_LOC int Finite(const float d)
{
#ifdef __CUDACC__
    return isfinite(d);
#else
#ifdef _WIN32
    return _finite(d);
#else
    return finite(d);
#endif
#endif
}

// Fix any NaNs, etc.
DMC_LOC float MakeFinite(const float tmpv)
{
#ifdef __CUDACC__
    if (isinf(tmpv) || isnan(tmpv))
        return 0.0f;
    else
        return tmpv;
#else
    if (!Finite(tmpv) || IsNaN(tmpv))
        return 0.0f;
    else
        return tmpv;
#endif
}

DMC_LOC unsigned int ToInt(const float l)
{
    float Cl = Saturate(l);
    float Sc = Cl * 65535.0f;
    return (unsigned int)(Sc);
}

DMC_LOC float ToFloat(const unsigned int l)
{
    return float(l & 0xffff) / 65535.0f;
}

DMC_LOC unsigned int FindLeadingOnes(unsigned int v)
{
    for (unsigned int mask = 0x8000; mask != 0xffff; mask |= mask >> 1u) {
        if ((mask & v) != mask) return (mask << 1u) & 0xffff;
    }

    return 0xffff;
}

// Wildly inaccurate but very fast atan2. Uses five fewer regs than real atan2.
DMC_LOC float myatan2f(float y, float x)
{
    float angle;
    const float pi14 = E_PI / 4;
    const float pi34 = 3 * pi14;
    float abs_y = fabsf(y) + 1e-10f; // kludge to prevent 0/0 condition

    if (x >= 0) {
        float r = (x - abs_y) / (x + abs_y);
        angle = pi14 - pi14 * r;
    } else {
        float r = (x + abs_y) / (abs_y - x);
        angle = pi34 - pi14 * r;
    }

    return y < 0 ? (-angle) : angle; // negate if in quad III or IV
}

// Adequate for real bases on 0..1000 and exponents on -10..10
// Error is up to 30%. Artifacts are that curves in IFS images are jaggy.
// http://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp/
// Actually slower than real powf on CUDA. Why?
DMC_LOC float mypowf(float a, float b)
{
    const int V = 1064861783; // Empirical estimate. For double precision it's 1072632447.

    union
    {
        float f;
        int x;
    } u = {a};

    int t = u.x - V;
    float bt = b * t;
    int bti = (int)bt;
    u.x = bti + V;

    return u.f;
}

}; // namespace
