#pragma once

// Needed for ColorSpace_t
#include "MathHelpers.h"
#include "MathStyleDefines.h"

#ifndef __CUDACC__
#include <Math/Vector.h>
#endif

namespace {

#ifdef __CUDACC__
struct f3Pixel : public float3
{
#define DMC_TMPLOC __device__
    DMC_TMPLOC f3Pixel() {}
    DMC_TMPLOC f3Pixel(float r_, float g_, float b_)
    {
        x = r_;
        y = g_;
        z = b_;
    }
    DMC_TMPLOC const float& r() const { return x; }
    DMC_TMPLOC float& r() { return x; }
    DMC_TMPLOC const float& g() const { return y; }
    DMC_TMPLOC float& g() { return y; }
    DMC_TMPLOC const float& b() const { return z; }
    DMC_TMPLOC float& b() { return z; }
#undef DMC_TMPLOC
};
#endif

// Clamp to 0..1
DMC_LOC f3Pixel Saturatef3(const f3Pixel& val)
{
    return f3Pixel(Saturate(val.r()), Saturate(val.g()), Saturate(val.b()));
}

// These are YUV to RGB.
// R = Y + (1.4075 * (V - 128));
// G = Y - (0.3455 * (U - 128) - (0.7169 * (V - 128));
// B = Y + (1.7790 * (U - 128);

// Y = R * 0.299 + G * 0.587 + B * 0.114;
// U = R * -0.169 + G * -0.332 + B * 0.500 + 128;
// V = R* 0.500 + G * -0.419 + B * -0.0813 + 128;

// JFIF YCrCb
// R = Y + 1.402 (Cr-128)
// G = Y - 0.34414 (Cb-128) - 0.71414 (Cr-128)
// B = Y + 1.772 (Cb-128)

// Actually Cr,Y,Cb
DMC_LOC f3Pixel YCrCbtoRGB(const f3Pixel& YCrCb)
{
    float r = YCrCb.r() + YCrCb.g();
    float b = YCrCb.b() + YCrCb.g();
    float g = (YCrCb.g() - 0.299f * r + 0.114f * b) * 1.7035775f;

    return f3Pixel(r, g, b);
}

// Assume Foley & VanDam HSV
// Hue 0...1 does all six hues. Other values wrap to this.
// V 0..1 maps directly to one of the channels.
// S can be anything.
// Valid input range is 0..1 for all three. Can handle anything, though.
// Actually H,V,S
DMC_LOC f3Pixel HSVtoRGB(const f3Pixel& HSV)
{
    int Hi = (int)HSV.r();
    float Hif = (float)Hi;
    float H = HSV.r() - Hif;
    if (H < 0.0f) H++;

    float H6 = H * 6.0f;
    int H6i = (int)H6; // 0,1,2,3,4,5
    float H6if = (float)H6i;
    float f = H6 - H6if;

    float S = HSV.b();
    float V = HSV.g();
    float p = V * (1 - S);
    float q = V * (1 - S * f);
    float t = V * (1 - S * (1 - f));

    float r, g, b;
    switch (H6i) {
    case 1: r = q, g = V, b = p; break;
    case 2: r = p, g = V, b = t; break;
    case 3: r = p, g = q, b = V; break;
    case 4: r = t, g = p, b = V; break;
    case 5: r = V, g = p, b = q; break;
    case 0:
    default: // Numerical precision such as HSV.r() can cause i==6.
        r = V, g = t, b = p;
        break;
    }

    return f3Pixel(r, g, b);
}

// Clamps negative numbers to zero, then maps positive numbers to 0..1.
DMC_LOC float ToneMap1L(const float p)
{
    return p <= 0.0f ? 0.0f : p / (1.0f + p);
}

// Clamps negative numbers to zero, then maps positive numbers to 0..1.
DMC_LOC f3Pixel ToneMapL(const f3Pixel& P)
{
    return f3Pixel(ToneMap1L(P.r()), ToneMap1L(P.g()), ToneMap1L(P.b()));
}

// Several ways to transform any red,grn,blu color values to output on 0..1.
DMC_LOC f3Pixel ColorTransform(const float red, const float grn, const float blu,
#ifdef __CUDACC__
                               int ColorSpace)
#else
                               int ColorSpace, const ColorMap<f3Pixel>& CMap)
#endif
{
    f3Pixel Orig(red, grn, blu);

    f3Pixel FinalVal;
    switch (ColorSpace) {
    case SPACE_RGB:
        // Treat the equations as R, G, B.
        FinalVal = Saturatef3(Orig);

        break;
    case SPACE_TONEMAP_RGB:
        // Treat the equations as R, G, B but tone map them to 0..1.
        FinalVal = ToneMapL(Orig);

        break;
    case SPACE_YCRCB: { // Treat the equations as Cr, Y, and Cb.
        f3Pixel WasYCrCb(YCrCbtoRGB(Orig));
        FinalVal = Saturatef3(WasYCrCb);
    } break;
    case SPACE_TONEMAP_HSV: { // Treat the equations as H, V, and S.
        f3Pixel WasHSV(HSVtoRGB(Orig));
        FinalVal = ToneMapL(WasHSV);
    } break;
    case SPACE_COLMAP: { // Treat the G equation as an index into the ColorMap.
#ifdef __CUDACC__
        float4 val4 = tex1D(ColMapTexRef, Orig.g());
        FinalVal.r() = val4.x;
        FinalVal.g() = val4.y;
        FinalVal.b() = val4.z;
#else
        FinalVal = CMap(Orig.g()); // CMap() clamps to 0..1.
#endif
    } break;
    case SPACE_TONEMAP_COLMAP: { // Treat ToneMap(G) as an index into the ColorMap.
        float TonedVal = ToneMap1L(Orig.g());
#ifdef __CUDACC__
        float4 val4 = tex1D(ColMapTexRef, TonedVal);
        FinalVal.r() = val4.x;
        FinalVal.g() = val4.y;
        FinalVal.b() = val4.z;
#else
        FinalVal = CMap(TonedVal);
#endif
    } break;
    }

    return FinalVal;
}

}; // namespace
