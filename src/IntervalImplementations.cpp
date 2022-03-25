#include "IntervalImplementations.h"

#include "ExprImplementations.h"

#include <algorithm>

///////////////////////////////////////////////////
// UnaryExpr

interval iAbs(const interval lv)
{
    if (lv.lower < 0) {
        if (lv.upper > 0)
            return interval(0, std::max(-lv.lower, lv.upper));
        else
            return interval(-lv.upper, -lv.lower);
    } else
        return lv;
}

interval iACos(const interval lv)
{
    return interval(eACos(lv.upper), eACos(lv.lower));
}

interval iASin(const interval lv)
{
    return interval(eASin(lv.lower), eASin(lv.upper));
}

interval iATan(const interval lv)
{
    if (lv.lower == interval::infinity() && lv.upper == interval::infinity()) return interval(0);
    if (lv.lower == -interval::infinity() && lv.upper == -interval::infinity()) return interval(0);

    return interval(eATan(lv.lower_finite()), eATan(lv.upper_finite()));
}

interval iBitNot(const interval lv)
{
    return interval(eBitNot(lv.lower), eBitNot(lv.upper));
}

interval iCbrt(const interval lv)
{
    return interval(eCbrt(lv.lower), eCbrt(lv.upper));
}

interval iClamp(const interval lv)
{
    return interval(eClamp(lv.lower), eClamp(lv.upper));
}

interval iCos(const interval lv)
{
    if (lv.lower == interval::infinity() && lv.upper == interval::infinity()) return interval(0);
    if (lv.lower == -interval::infinity() && lv.upper == -interval::infinity()) return interval(0);

    interval iout(eCos(lv.lower), eCos(lv.upper));

    if (lv.upper >= 2.0f * E_PI || lv.lower <= -2.0f * E_PI) iout = interval(-1.0f, 1.0f);
    if (lv.contains(E_PI)) iout.extend(-1.0f);
    if (lv.contains(0)) iout.extend(1.0f);
    if (lv.contains(-E_PI)) iout.extend(-1.0f);

    return iout;
}

interval iCube(const interval lv)
{
    return interval(eCube(lv.lower), eCube(lv.upper));
}

interval iExp(const interval lv)
{
    return interval(eExp(lv.lower), eExp(lv.upper));
}

interval iLn(const interval lv)
{
    if (lv.lower == 0 && lv.upper == 0) return interval(0);

    if (lv.lower < 0 && lv.upper > 0) {
        interval iout(iLn(interval(lv.lower, -0.0f)));
        iout.extend(iLn(interval(0.0f, lv.upper)));
        return iout;
    }

    ASSERT_D(!(lv.lower < 0 && lv.upper > 0));

    if (lv.lower < 0 || lv.upper < 0) return iLn(-lv);

    ASSERT_D(lv.lower >= 0 && lv.upper >= 0);

    interval iout(eLn(lv.lower), eLn(lv.upper));

    if (lv.contains(0.0f)) iout.extend(-87.3365f); // This is log(eps).

    return iout;
}

interval iRound(const interval lv)
{
    return interval(eRound(lv.lower), eRound(lv.upper));
}

interval iSin(const interval lv)
{
    if (lv.lower == interval::infinity() && lv.upper == interval::infinity()) return interval(0);
    if (lv.lower == -interval::infinity() && lv.upper == -interval::infinity()) return interval(0);

    interval iout(eSin(lv.lower), eSin(lv.upper));
    if (lv.upper >= 1.5f * E_PI || lv.lower <= -1.5f * E_PI) iout = interval(-1.0f, 1.0f);
    if (lv.contains(0.5f * E_PI)) iout.extend(1.0f);
    if (lv.contains(-0.5f * E_PI)) iout.extend(-1.0f);

    return iout;
}

interval iSqr(const interval lv)
{
    interval iout(eSqr(lv.lower), eSqr(lv.upper));
    if (lv.contains(0.0f)) iout.extend(0.0f);

    return iout;
}

interval iSqrt(const interval lv)
{
    interval iout(eSqrt(lv.lower), eSqrt(lv.upper));
    if (lv.contains(0.0f)) iout.extend(0.0f);

    return iout;
}

interval iTan(const interval lv)
{
    if (lv.lower == interval::infinity() && lv.upper == interval::infinity()) return interval(0);
    if (lv.lower == -interval::infinity() && lv.upper == -interval::infinity()) return interval(0);

    // Move both lower and upper to the canonical interval
    float lb = lv.lower - E_PI * 0.5f;
    float ub = lv.upper - E_PI * 0.5f;
    float periods = floorf(lb / E_PI);
    float offset = periods * E_PI;
    float lo = lb - offset;
    float ro = ub - offset;

    // ASSERT_D(lo >= 0 && lo <= E_PI);
    // ASSERT_D(ro >= lo);

    interval iout(eTan(lv.lower), eTan(lv.upper));
    if (ro > E_PI || !lv.finite()) // Spans a singularity
        iout = interval(-interval::infinity(), interval::infinity());

    return iout;
}

interval iUnaryMinus(const interval lv)
{
    return interval(eUnaryMinus(lv.lower), eUnaryMinus(lv.upper));
}

///////////////////////////////////////////////////
// BinaryExpr

interval iAnd(const interval lv, const interval rv)
{
    unsigned int ll = ToInt(lv.lower), lu = ToInt(lv.upper), rl = ToInt(rv.lower), ru = ToInt(rv.upper);

    unsigned int lvx = FindLeadingOnes(~(ll ^ lu));
    unsigned int rvx = FindLeadingOnes(~(rl ^ ru));
    unsigned int constmask = (lvx | rvx);

    return interval(ToFloat((ll & rl) & constmask), ToFloat(std::min(lu, ru)));
}

namespace {
void extenAT(float l, float r, interval& iout)
{
    // if (l != 0 || r != 0) // By enabling the kludge in myatan2f we can handle 0,0
    iout.extend(eATan2(l, r));
}

void extenDiv(float l, float r, interval& iout)
{
    const float inf = interval::infinity();
    if (l == inf && r == inf)
        iout.extend(interval(inf));
    else
        iout.extend(eDiv(l, r));
}
}; // namespace

interval iATan2(const interval lv, const interval rv)
{
    if (lv.lower == interval::infinity() && lv.upper == interval::infinity()) return interval(0);
    if (lv.lower == -interval::infinity() && lv.upper == -interval::infinity()) return interval(0);
    if (rv.lower == interval::infinity() && rv.upper == interval::infinity()) return interval(0);
    if (rv.lower == -interval::infinity() && rv.upper == -interval::infinity()) return interval(0);

    interval iout;

    extenAT(lv.lower_finite(), rv.lower_finite(), iout);
    extenAT(lv.lower_finite(), rv.upper_finite(), iout);
    extenAT(lv.upper_finite(), rv.lower_finite(), iout);
    extenAT(lv.upper_finite(), rv.upper_finite(), iout);

    if (lv.contains(0) && lv.lower < 0) {
        iout.extend(eATan2(0, rv.lower_finite()));
        if (Finite(rv.upper)) iout.extend(eATan2(0, rv.upper_finite()));
        iout.extend(eATan2(-interval::min_float(), rv.lower_finite())); // Need the negative sign to yield a -PI
        if (Finite(rv.upper)) iout.extend(eATan2(-interval::min_float(), rv.upper_finite()));
    }

    return iout;
}

interval iDiv(const interval lv, const interval rv)
{
    const float inf = interval::infinity();
    if (lv.lower == 0 && lv.upper == 0) return interval(0);
    if (rv.lower == 0 && rv.upper == 0) return interval(0);
    if (rv.lower == inf && rv.upper == inf) return interval(0);

    if (lv.lower < 0 && lv.upper > 0) {
        interval iout(iDiv(interval(lv.lower, -0.0f), rv));
        iout.extend(iDiv(interval(0.0f, lv.upper), rv));
        return iout;
    }

    if (rv.lower < 0 && rv.upper > 0) {
        interval iout(iDiv(lv, interval(rv.lower, -0.0f)));
        iout.extend(iDiv(lv, interval(0.0f, rv.upper)));
        return iout;
    }

    // Neither span crosses zero.
    ASSERT_D(!(lv.lower < 0 && lv.upper > 0));
    ASSERT_D(!(rv.lower < 0 && rv.upper > 0));

    if (lv.lower < 0 || lv.upper < 0) return -iDiv(-lv, rv);
    if (rv.lower < 0 || rv.upper < 0) return -iDiv(lv, -rv);

    ASSERT_D(lv.lower >= 0 && lv.upper >= 0 && rv.lower >= 0 && rv.upper >= 0);

    interval iout;

    extenDiv(lv.lower, rv.lower, iout);
    extenDiv(lv.lower, rv.upper, iout);
    extenDiv(lv.upper, rv.lower, iout);
    extenDiv(lv.upper, rv.upper, iout);

    // Even though div 0 returns 0, denominators near zero can yield arbitrarily large numbers.
    if (rv.lower == 0.0f) iout.extend(interval::infinity());

    return iout;
}

interval iMinus(const interval lv, const interval rv)
{
    interval iout;
    if (lv.lower == -interval::infinity() || rv.upper == interval::infinity())
        iout.extend(-interval::infinity());
    else
        iout.extend(eMinus(lv.lower, rv.upper));

    if (lv.upper == interval::infinity() || rv.lower == -interval::infinity())
        iout.extend(interval::infinity());
    else
        iout.extend(eMinus(lv.upper, rv.lower));

    return iout;
}

interval iMod(const interval lv, const interval rv)
{
    if (rv.lower == 0 && rv.upper == 0) return interval(0);

    if (lv.lower < 0 && lv.upper > 0) {
        interval iout(iMod(interval(lv.lower, -0.0f), rv));
        iout.extend(iMod(interval(0.0f, lv.upper), rv));
        return iout;
    }

    if (rv.lower < 0 && rv.upper > 0) {
        interval iout(iMod(lv, interval(rv.lower, -0.0f)));
        iout.extend(iMod(lv, interval(0.0f, rv.upper)));
        return iout;
    }

    // Neither span crosses zero.
    ASSERT_D(!(lv.lower < 0 && lv.upper > 0));
    ASSERT_D(!(rv.lower < 0 && rv.upper > 0));

    // fmod(x,y) has same sign as x and magnitude less than y.
    if (lv.lower < 0 || lv.upper < 0) return -iMod(-lv, rv);
    if (rv.lower < 0 || rv.upper < 0) return iMod(lv, -rv);

    ASSERT_D(lv.lower >= 0 && lv.upper >= 0 && rv.lower >= 0 && rv.upper >= 0);

    // Zeros have positive sign
    interval lvo(lv), rvo(rv);
    if (lvo.lower == 0) lvo.lower = 0.0f;
    if (lvo.upper == 0) lvo.upper = 0.0f;
    if (rvo.lower == 0) rvo.lower = 0.0f;
    if (rvo.upper == 0) rvo.upper = 0.0f;

    if (lvo.lower == 0 && lvo.upper == 0) return interval(0);
    if (lvo.lower == interval::infinity()) return interval(0);                  // HACK to deal with [inf,inf] % [whatever] = [0,0] in sampled.
    if (lvo.upper < rvo.lower || rvo.lower == interval::infinity()) return lvo; // No overlap so no modding.
    if (lvo.upper == interval::infinity())                                      // put this last to not mess up [1,inf] % [inf,inf].
        return interval(0, rvo.upper);
    if (lvo.lower == lvo.upper && rvo.lower == rvo.upper) return interval(eMod(lvo.lower, rvo.lower)); // fails on inf, so do it after

    interval iout;
    iout.extend(interval(eMod(lvo.lower, rvo.lower), eMod(lvo.upper, rvo.upper)));
    iout.extend(interval(eMod(lvo.upper, rvo.lower), eMod(lvo.upper, rvo.upper)));

    if (floorf(lvo.lower / rvo.upper) != floorf(lvo.upper / rvo.upper)) iout.extend(interval(0, rvo.upper)); // A ridge is exiting the right.upper side.

    float d = 1.0f + floorf(lvo.upper / rvo.upper);
    float ld = lvo.upper / d;
    if (rvo.lower <= ld && rvo.upper > ld) iout.extend(interval(0, ld)); // A ridge is exiting the left.upper side.

    return iout;
}

namespace {
float make_finite(float a)
{
    return (a == interval::infinity()) ? 1.0f : (a == -interval::infinity()) ? -1.0f : a;
}

float fmul(float a, float b)
{
    float m = eMult(make_finite(a), make_finite(b));

    if ((!Finite(a) || !Finite(b)) && (a != 0 && b != 0))
        return m < 0 ? -interval::infinity() : interval::infinity();
    else
        return m;
}
}; // namespace

interval iMult(const interval lv, const interval rv)
{
    // Use fmul to avoid 0 * inf = NaN and -inf * inf = NaN
    // This will mismatch the sampled one, which won't insert the NaN in the interval. But in reality the interval needs to span 0 to infinity.

    interval iout;

    iout.extend(fmul(lv.lower, rv.lower));
    iout.extend(fmul(lv.lower, rv.upper));
    iout.extend(fmul(lv.upper, rv.lower));
    iout.extend(fmul(lv.upper, rv.upper));

    return iout;
}

interval iOr(const interval lv, const interval rv)
{
    unsigned int ll = ToInt(lv.lower), lu = ToInt(lv.upper), rl = ToInt(rv.lower), ru = ToInt(rv.upper);

    unsigned int lvx = FindLeadingOnes(ll & lu) | FindLeadingOnes(~ll & ~lu);
    unsigned int rvx = FindLeadingOnes(rl & ru) | FindLeadingOnes(~rl & ~ru);
    unsigned int constmask = (lvx | rvx);

    return interval(ToFloat(std::max(ll, rl)), ToFloat((lu | ru) | ~constmask));
}

interval iPlus(const interval lv, const interval rv)
{
    interval iout;
    if (lv.lower == -interval::infinity() || rv.lower == -interval::infinity())
        iout.extend(-interval::infinity());
    else
        iout.extend(ePlus(lv.lower, rv.lower));

    if (lv.upper == interval::infinity() || rv.upper == interval::infinity())
        iout.extend(interval::infinity());
    else
        iout.extend(ePlus(lv.upper, rv.upper));

    return iout;
}

interval iPow(const interval lv, const interval rv)
{
#define CHECKL(x)                           \
    do {                                    \
        if (lv.contains(x)) {               \
            iout.extend(ePow(x, rv.lower)); \
            iout.extend(ePow(x, rv.upper)); \
        }                                   \
    } while (0)
#define CHECKR(x)                           \
    do {                                    \
        if (rv.contains(x)) {               \
            iout.extend(ePow(lv.lower, x)); \
            iout.extend(ePow(lv.upper, x)); \
        }                                   \
    } while (0)

    // A negative exponent is 1/the positive.
    // ePow rounds rv if lv is negative.
    // ePow computes pow(abs(lv), rv) and negates result if rv is odd.

    interval iout;
    iout.extend(ePow(lv.lower, rv.lower));
    iout.extend(ePow(lv.lower, rv.upper));
    iout.extend(ePow(lv.upper, rv.lower));
    iout.extend(ePow(lv.upper, rv.upper));

    if (rv.contains(0)) { iout.extend(1); }
    if (rv.contains(rv.upper - 1) && rv.upper - 1 >= 1e8f && lv.lower < -1.0f) {
        iout.extend(-ePow(lv.lower, rv.upper - 1));
        iout.extend(-ePow(lv.upper, rv.upper - 1));
    }
    if (rv.contains(rv.lower + 1) && rv.lower + 1 >= 1e8f && lv.lower < -1.0f) {
        iout.extend(-ePow(lv.lower, rv.lower + 1));
        iout.extend(-ePow(lv.upper, rv.lower + 1));
    }

    CHECKR(rv.lower + 1); // -0 ^ -1 return -inf. We need inf. The problem is that sIval still gets the -0 behavior. So stick with the -0 behavior.
    CHECKR(rv.upper - 1);
    CHECKR(-1);
    CHECKR(0);
    CHECKR(1);
    CHECKR(1000000);
    CHECKR(1000001); // Negative base to positive, odd power => -inf, but sampled can't represent large, odd numbers, so it misses it.
    CHECKR(-1000000);
    CHECKR(-1000001);

    CHECKL(std::numeric_limits<float>::min());
    CHECKL(-std::numeric_limits<float>::min());

    if (lv.contains(0)) {
        if (rv.lower < 0.0f && lv.upper > 0)   // Small positive base to a negative power. Inverting a fraction makes it large.
            iout.extend(interval::infinity()); // Needs 0 threshold to get +inf in output.

        if (rv.lower <= -1.0f && lv.lower < 0) // Small negative base to a negative power. Inverting a fraction makes it large. Exponent rounds toward zero, so start at -1.
            if (rv.span() >= 2 || (int(rv.lower) & 1) || int(rv.upper) & 1 ||
                (rv.contains(int(rv.lower + 1)) && (int(rv.lower + 1) & 1))) // Make sure there's an ODD power, or the result will be positive infinity.
                iout.extend(-interval::infinity());                          // Need -1 threshold to avoid -inf in output
            else
                iout.extend(interval::infinity());

        if (rv.upper > 1 && lv.upper > 0) // Small positive base to a positive power sends base toward zero.
            iout.extend(0);
    }

    if (lv.lower > 0) ASSERT_D(iout.lower >= 0);

    return iout;
}

interval iXOr(const interval lv, const interval rv)
{
    unsigned int ll = ToInt(lv.lower), lu = ToInt(lv.upper), rl = ToInt(rv.lower), ru = ToInt(rv.upper);

    unsigned int lvx = FindLeadingOnes(ll & lu) | FindLeadingOnes(~ll & ~lu);
    unsigned int rvx = FindLeadingOnes(rl & ru) | FindLeadingOnes(~rl & ~ru);
    unsigned int constmask = (lvx | rvx);
    interval iout(ToFloat((ll ^ rl) & constmask), ToFloat((lu ^ ru) & constmask)); // Not sure which is larger; interval constructor sorts them.

    iout.extend(ToFloat(ToInt(iout.upper) | ~constmask)); // Now that the upper is known, extend it upward for the lsbs.

    return iout;
}
