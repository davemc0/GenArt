#pragma once

// Doesn't include any of the class-based system.

#include "MathHelpers.h"

const int NUM_NONARY_EXPR_OPCODES = 2;
const int NUM_UNARY_EXPR_OPCODES = 17;
const int NUM_BINARY_EXPR_OPCODES = 11;
const int NUM_EXPR_OPCODES = 30;

// WARNING!!! Change the above constants when messing with this list.
enum ExprOpcodes_t {
    Const_e = 0,
    Var_e = 1,

    Abs_e = 2,
    ACos_e = 3,
    ASin_e = 4,
    ATan_e = 5,
    BitNot_e = 6,
    Cbrt_e = 7,
    Clamp_e = 8,
    Cos_e = 9,
    Cube_e = 10,
    Exp_e = 11,
    Ln_e = 12,
    Round_e = 13,
    Sin_e = 14,
    Sqr_e = 15,
    Sqrt_e = 16,
    Tan_e = 17,
    UnaryMinus_e = 18,

    And_e = 19,
    ATan2_e = 20,
    Div_e = 21,
    IFS_e = 22,
    Minus_e = 23,
    Mod_e = 24,
    Mult_e = 25,
    Or_e = 26,
    Plus_e = 27,
    Pow_e = 28,
    XOr_e = 29
};

const int IFS_MAX_ITER = 20;

// For inline evaluation, need to know how many variables to deal with
const int MAX_VARIABLES = 3;

namespace {

DMC_LOC float eAbs(const float lv) { return fabsf(lv); }

DMC_LOC float eACos(const float lv) { return acosf(Clampf(-1.0f, lv, 1.0f)); }

DMC_LOC float eAnd(const float lv, const float rv)
{
    unsigned int ll = ToInt(lv);
    unsigned int rr = ToInt(rv);

    return ToFloat(ll & rr);
}

DMC_LOC float eASin(const float lv) { return asinf(Clampf(-1.0f, lv, 1.0f)); }

DMC_LOC float eATan(const float lv)
{
    // This uses two fewer regs than real atan.
    return myatan2f(lv, 1.0f);
}

DMC_LOC float eATan2(const float lv, const float rv) { return myatan2f(lv, rv); }

DMC_LOC float eBitNot(const float lv)
{
    unsigned int ll = ToInt(lv);

    return ToFloat(~ll);
}

DMC_LOC float eCbrt(const float lv) { return (lv > 0.0f) ? powf(lv, 1.0f / 3.0f) : 0.0f; }

DMC_LOC float eClamp(const float lv) { return Saturate(lv); }

DMC_LOC float eCos(const float lv) { return cosf(lv); }

DMC_LOC float eCube(const float lv) { return lv * lv * lv; }

DMC_LOC float eDiv(const float lv, const float rv) { return (rv != 0.0f) ? lv / rv : 0.0f; }

DMC_LOC float eExp(const float lv) { return expf(lv); }

DMC_LOC float eLn(const float lv)
{
    return (lv == 0.0f) ? 0.0f : logf(fabsf(lv)); // Note the absolute value here.
}

DMC_LOC float eMinus(const float lv, const float rv) { return lv - rv; }

DMC_LOC float eMod(const float lv, const float rv)
{
#if 0
        return (rv != 0.0f) ? fmodf(lv, rv) : 0.0f;
#else // 2X faster on CPU; 10X on GPU
    float quo = lv / rv;
    int iclampquo = (int)quo; // Fails for huge numbers. Returns inf.
    float fclampquo = (float)iclampquo;
    float remfrac = quo - fclampquo;
    float rem = remfrac * rv;
    return (rv != 0.0f) ? rem : 0.0f;
#endif
}

DMC_LOC float eMult(const float lv, const float rv) { return lv * rv; }

DMC_LOC float eOr(const float lv, const float rv)
{
    unsigned int ll = ToInt(lv);
    unsigned int rr = ToInt(rv);

    return ToFloat(ll | rr);
}

DMC_LOC float ePlus(const float lv, const float rv) { return lv + rv; }

DMC_LOC float ePow(const float lv, const float rv)
{
    // If base is negative then round the exponent toward zero. Negative base to an odd power has a negative result.
    // This should make ePow never return NaN. CUDA fast pow can't handle negative bases at all, even with whole number exponents.
    // Fails for huge rv because the int cast yields -2B. Need a higher perf fix.
    bool lneg = lv < 0;
    int r = int(rv);
    float rf = lneg ? float(r) : rv;
    float res = powf(fabsf(lv), rf);
    if (lneg && (r & 1))
        return -res; // Negative base to an odd power has a negative result.
    else
        return res;
}

DMC_LOC float eRound(const float lv) { return (lv < 0.5f) ? 0.0f : 1.0f; }

DMC_LOC float eSin(const float lv) { return sinf(lv); }

DMC_LOC float eSqr(const float lv) { return lv * lv; }

DMC_LOC float eSqrt(const float lv)
{
    return sqrtf(fabsf(lv)); // Note the absolute value here.
}

DMC_LOC float eTan(const float lv)
{
    return tanf(eMod(lv, E_PI)); // Note the fmod here. tan is very unstable without it.
}

DMC_LOC float eUnaryMinus(const float lv) { return -lv; }

DMC_LOC float eXOr(const float lv, const float rv)
{
    unsigned int ll = ToInt(lv);
    unsigned int rr = ToInt(rv);

    return ToFloat(ll ^ rr);
}
}; // namespace
