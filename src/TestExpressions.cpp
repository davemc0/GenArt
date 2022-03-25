/////////////////////////////////////////////////////////////////////////////////////////////////
// Some test code

#include "ExprTools.h"
#include "MathHelpers.h"
#include "MathStyleCUDARender.h"

#include <Math/Halton.h>
#include <Math/Random.h>
#include <Util/Timer.h>
#include <fstream>

#ifdef DMC_MACHINE_win
#include <time.h>
#endif
#include <math.h>

namespace {
////////////////////////////////////////////////////////////////////////////////
// Test Expression Optimization

// I finally arrived at the optimization rule that I would evaluate expressions with whatever modifications to the input I need, such as Sqrt(Abs(f)), rather than Sqrt(f),
// but I would optimize using theoretical rules, ignorant of the hacks on the input side.
// This causes the numerical testing of the symbolic optimization to return mismatches, but I don't care anymore.

// Raising a negative number to a fractional power is undefined. Includes sqrt(neg). With ePow if base is negative then round the exponent toward zero.
// Need to make finite before checking equivalence.
// pow 0 acos ifs x x : The rhs is non-const but evaluates to 0, so it should apply the rule "Anything to the 0 = 1", but instead applies "0 to the anything but 0 is 0".
// | y y : Simplifies to y, but mismatches because y != | y y quantized to 16-bit normalized.
// XOr simplifies to Clamp if its arg is < 0 and to BitNot if its arg is > 1.
// Sqrt evaluates Sqrt(Abs(x)), and when simplifying, the abs gets lost.
// Getting a really big intermediate value can cause error with a high-frequency function: sin * tan ln / y r / x r

// The math.h pow() function applies these rules:
// x <>  0 and y = 0.0 : 1
// x = 0.0 and y = 0.0 : 1
// x = 0.0 and y < 0.0 : INF
// x = 0.0 and y > 0.0 : 0
// x < 0.0 and y is frac | IND

// pow round x round y
//     CPU GPU
// 0^0=1   0
// 1^0=1   1
// 0^1=0   0
// 1^1=1   1

// Everything does a straightforward evaluation except the following:
// ACos:    { return acosf(Clampf(-1.0f, left->Eval(VV), 1.0f)); }
// ASin:    { return asinf(Clampf(-1.0f, left->Eval(VV), 1.0f)); }
// And:     { unsigned short ll = ToShort(left->Eval(VV)); unsigned short rr = ToShort(right->Eval(VV)); return ToFloat(ll & rr); }
// Clamp:   { return Clampf(0.0f, left->Eval(VV), 1.0f); }
// Div:     { float r = right->Eval(VV); if(r != 0.0f) return left->Eval(VV) / r; else return 0.0f; }
// IFS:     weird
// Ln:      return (lefval == 0.0f) ? 0.0f : logf(fabsf(lefval));
// Mod:     { float r = right->Eval(VV); if(r != 0.0f) return fmodf(left->Eval(VV), r); else return 0.0f; }
// Or:      { unsigned short ll = ToShort(left->Eval(VV)); unsigned short rr = ToShort(right->Eval(VV)); return ToFloat(ll | rr); }
// Round:   { return left->Eval(VV) < 0.5f ? 0.0f : 1.0f; }
// Sqrt:    { return sqrtf(fabsf(left->Eval(VV))); }
// Tan:     { return tanf(fmod(left->Eval(VV), 3.141592653589f)); }
// XOr:     { unsigned short ll = ToShort(left->Eval(VV)); unsigned short rr = ToShort(right->Eval(VV)); return ToFloat(ll ^ rr); }
// Pow:     { see ePow }
// ATan2:

//////////////////////////////////////
// Input Functors

class GetTestExprFromFile
{
public:
    GetTestExprFromFile(VarVals_t& vvals_, std::ifstream& inf_) : inf(inf_), vvals(vvals_) {}

    // Returns NULL when it runs out.
    Expr* operator()()
    {
        std::string Line;
        bool ok = getline(inf, Line) ? true : false;
        if (!ok) return NULL;

        std::string remain;
        Expr* A = ReadExpr(Line, remain, &vvals);

        return A;
    }

private:
    std::ifstream& inf;
    VarVals_t vvals;
};

std::string ExpressionsToTest[] = {
    "* * x 0.87 2.5",
    "* ifs ln sqrt tan / 0.299875 x ln ln sqrt - x y 2",
    "* pow sin x 3"
    "* sin exp - 1 y -1",
    "* sqr y y",
    "* x / 1 y",
    "+ % * ln clamp atan * x 0.164251 % pow x 0.964193 clamp sqrt y pow abs ln + y x 0.27132 ln acos ln - / sin exp exp - sin + y x - tan x x exp sin exp exp - ^ cos x - x 0.650258 y - sin exp exp "
    "sqr x sin exp exp sin sqrt y",
    "+ + + atan * r clamp * 0.759335 r atan * ln * 0.4343 ln ln r ifs acos r + abs + r * x y * 0.4343 ln % x y ifs + y + 0.266561 ~ r + -0.247315 * 0.4343 ln - 0.446822 / x y ifs + y + 0.323163 ~ r "
    "+ sin cos y * 0.4343 ln - 0.501776 / x y",
    "+ + + atan * r clamp * 0.759335 r atan * ln * 0.4343 ln ln r ifs acos r + abs + r * x y * 0.4343 ln % x y ifs + y + 0.323163 ~ r + sin cos y * 0.4343 ln - 0.501776 / x y ifs + y + 0.266561 ~ r "
    "+ -0.247315 * 0.4343 ln - 0.446822 / x y",
    "+ - exp ^ % asin r * 4.06077 y + -0.200677 + y asin asin r / * atan x sin clamp - 0.195316 x ~ r * 0.5 sqrt tan + exp sin exp exp - -0.857132 y exp - sin + x sin exp exp + -0.706627 + x sin x "
    "ln cube atan abs * 0.0364843 ^ x y",
    "+ atan * cube atan cube r ifs acos r + abs + r * x y * 0.4343 ln % x y ifs + y + 0.266561 ~ r + -0.247315 * 0.4343 ln - 0.446822 / x y",
    "+ atan * ln * 0.4343 ln ln x ifs acos r + abs * 2 y * 0.4343 ln % x y ifs + y + 0.266561 ~ r + -0.247315 * 0.4343 ln - 0.446822 / x y",
    "+ cube sqrt ln tan x cube sqrt ln tan x",
    "+ ifs - * y * x 2.14818 + sqr y * 0.4343 ln - 0.462837 + x y * y * x 2.13419 * exp sqr y * sqrt x 0.0353474",
    "+ pow sqr x y sqr x",
    "+ sqr asin sqr tan sqrt / asin / x y sqr asin sqr y + * 2 sqr * 0.4343 ln sqrt asin / x y pow tan sqrt / sqr * 0.4343 ln sqrt asin / x y sqr asin sqr y 4",
    "- * sqrt tan exp - * clamp sin tan - cube acos pow r 0.25 * 0.4343 ln cbrt % y x cos y asin pow 0.330428 * 1.27142 cos y ^ * exp - sin x asin ^ sin y ifs | 0.283943 asin y + sqr y * 0.4343 ln - "
    "0.501776 / x y sin * 0.389718 asin y cos y ifs + exp round r + 0.179928 ln * sqrt sin & 0.71179 y + -0.720325 sin r + sqr y * 0.4343 ln - 0.499015 / x y",
    "- - x x clamp 0.864688",
    "- / sin exp - sin + y x - tan x x exp sin exp - ^ cos x 1 y - x 1",
    "--- clamp ln 0.886185",
    "--- sqrt pow 0 x", // 0
    "--- sqrt pow x x",
    "/ exp x exp sin exp - 1 y",
    "/ ln tan | y y cbrt clamp | 0.835445 0.633637",
    "/ sin exp - sin + y x - tan x x exp sin exp - ^ cos x 1 y",
    "/ sin exp - sin x x exp sin exp - ^ cos x 1 y",
    "/ sin exp y exp sin exp y",
    "/ x y",
    "^ * x y % * 0 % pow x 0.680589 clamp sqrt y pow 0.235888 sin * x 1.58714",
    "^ * x y % * ln clamp atan * x 0.0999542 % pow x 0.680589 clamp sqrt y pow 0.235888 sin * x 1.58714",
    "^ | exp 0.421578 | 0.547172 0.251272 cube sqrt - x 0.699815", // The sqrt has an implicit Abs that threw it off.
    "abs ln sin pow atan2 x r abs ^ 0.262373 0.446345",
    "atan2 - - x x clamp 0.864688 ln cos & 0.436723 x",
    "atan2 0.864688 ln cos & 0.436723 x",
    "cbrt - - 0.754252 y atan sqrt atan y",
    "cbrt 0.162444",
    "cos * * exp asin - 0.181764 y 0.872265 26.5514",
    "exp + + x - x ln sqr * 0.04 ^ x y 0.7", // This one is just very numerically unstable.
    "exp ln tan & % x y - x 0.154261",
    "ifs * 0.4343 ln atan2 * 0.4343 ln * 0.4343 ln / x y 1.55741 * y * x 1.73471",
    "ifs - abs * 0.4343 ln atan * ~ r + acos r + abs * x y * 0.4343 ln & x y * 0.4343 ln - 0.285313 / x y * 0.4343 ln - 0.281407 / x y",
    "ifs --- 0.181886 cbrt 0.162444",
    "ifs ln / x y * * x y 1.8",
    "ln exp 1",
    "ln exp round ~ & ln sqrt 0.56459 sqrt / x y",
    "ln 0.886185",
    "ln tan x",
    "pow / sqr x sqr y 1.5", // Simplifies to cube / ln x y, which circumvents pow rounding off exponent if base is negative, which happens because can't take sqrt of negative number.
    "pow --- pow 0 * x 0.5 ifs -0.181886 0.545634", // 0
    "pow --- pow x * x 0.5 ifs -0.181886 0.545634",
    "pow --- sqrt pow 0 x 1",                           // 0
    "pow --- sqrt pow 0 x ifs --- 0.181886 0.545634",   // 0
    "pow --- sqrt pow 0 x ifs -0.181886 0.545634",      // 0
    "pow --- sqrt pow 0 x ifs -0.181886 cbrt 0.162444", // 0
    "pow --- sqrt pow x x 1",
    "pow --- sqrt pow x x ifs --- 0.181886 0.545634",
    "pow --- sqrt pow y x ifs -0.181886 0.545634",
    "pow --- sqrt pow y x ifs -0.181886 cbrt 0.162444",
    "pow -10.0745392 1.5",
    "pow -10.0745392 clamp / x y",
    "pow 0 acos ifs x y",
    "pow 0 x",
    "pow pow x y x",
    "pow pow x y y",
    "pow sqrt --- clamp ln 0.886185 acos ifs x y",
    "round pow --- sqrt pow y x 1",
    "round pow --- sqrt pow y x ifs --- 0.181886 cbrt 0.162444",
    "round ~ & ln sqrt 0.56459 1",
    "sqrt --- clamp ln 0.886185",
    "sqrt -0.07",
    "sqrt ifs - / x 0.409325 sqr r % % y 0.161345 * y 0.0580627",
    "sqrt ln tan x",
    "sqrt sin sin exp exp - sin exp exp - sin exp exp sin sin exp exp - sin + - y x x - 0.713157 x - + y x x sin exp exp sin sin exp exp - sin + - y x x - 0.713157 x",
    "tan & % x y - x 0.154261",
    "| y y", // => y, which doesn't have the float2int convert.
    "| ~ + 0.670788 0.40931 cube cube atan2 y 0.371026",
    "| ~ + 0.670788 0.40931 y",
};

class GetTestExprFromList
{
public:
    GetTestExprFromList(VarVals_t vvals_) : vvals(vvals_) { index = 0; }

    // Returns NULL when it runs out.
    Expr* operator()()
    {
        if (index >= sizeof(ExpressionsToTest) / sizeof(std::string)) return NULL;

        std::string remain;
        Expr* A = ReadExpr(ExpressionsToTest[index], remain, &vvals);
        index++;

        return A;
    }

    int index;
    VarVals_t vvals;
};

class GetTestExprRandom
{
public:
    GetTestExprRandom(VarVals_t vvals_, int max_exprs_ = 0) : vvals(vvals_)
    {
        index = 0;
        max_exprs = max_exprs_;
    }

    // Returns NULL when it runs out.
    Expr* operator()()
    {
        if (index > max_exprs && max_exprs > 0) return NULL;

        Expr* A = RandExpr(12, &vvals);
        index++;

        return A;
    }

    int index, max_exprs;
    VarVals_t vvals;
};

//////////////////////////////////////
// Test Items (symbolic optimization, numerical optimization, etc.)

class TestItemNull
{
public:
    TestItemNull() {}

    Expr* operator()(Expr* A) { return A->Copy(); }
};

class TestItemOptimizer
{
public:
    TestItemOptimizer(VarVals_t vvals_, int Steps_ = 531, float maxAbsErr_ = 0.01f) : Steps(Steps_), maxAbsErr(maxAbsErr_), MinVV(vvals_), MaxVV(vvals_)
    {
        for (size_t i = 0; i < MinVV.vals.size(); i++) {
            MinVV.vals[i] = 0.0f;
            MaxVV.vals[i] = 1.0f;
        }
    }

    Expr* operator()(Expr* A)
    {
        interval outSpan;
        outSpan.set_infinite();

        Expr* O = Optimize(A, MinVV, MaxVV, Steps, maxAbsErr, outSpan);
        // std::cerr << O->Print(PREFIX) << std::endl << O->Print(PREFIX) << std::endl;

        return O;
    }

private:
    VarVals_t MinVV;
    VarVals_t MaxVV;
    int Steps;
    float maxAbsErr;
};

class TestItemTokenizedEval
{
public:
    TestItemTokenizedEval() {}

    Expr* operator()(Expr* A)
    {
#if 0
        // This will require changing TestItem to evaluate the expression and return the result. Later.
        float VVals[MAX_VARIABLES];
        int HostTokens[MAX_TOKENS];
        int TokenCnt = 1; // Leave slot 0 for the length
        TokenCnt += Tokenize(A, HostTokens + TokenCnt, MAX_TOKENS - TokenCnt);
        TokenCnt += Tokenize(new Const(0), HostTokens + TokenCnt, MAX_TOKENS - TokenCnt);
        TokenCnt += Tokenize(new Const(0), HostTokens + TokenCnt, MAX_TOKENS - TokenCnt);
        HostTokens[0] = TokenCnt;

        float r = sqrtf(x * x + y * y);
        GlobalVV.Values[0] = x;
        GlobalVV.Values[2] = r;
        VVals[0] = x;
        VVals[1] = y;
        VVals[2] = r;

        float rout, gout, bout;
        EvaluateTokens(HostTokens /* + 1 */, TokenCnt, VVals, rout, gout, bout);
#endif
    }
};

//////////////////////////////////////
// Equality Criteria

class RelAbsComparator
{
public:
    RelAbsComparator(bool RelQuant_ = false, bool AbsQuant_ = false, float rel_ = 0.0001f, float abs_ = 0.0001f, bool either_ = false) :
        RelQuant(RelQuant_), AbsQuant(AbsQuant_), RelErr(rel_), AbsErr(abs_), EitherMetric(either_)
    {
    }

    bool operator()(float a, float b)
    {
        if (RelQuant) {
            a = QuantizeRelative(a);
            b = QuantizeRelative(b);
        }

        if (AbsQuant) {
            a = QuantizeAbsolute(a);
            b = QuantizeAbsolute(b);
        }

        float abserr = fabsf(a - b);
        float denom = std::max(fabsf(a), fabsf(b));
        float relerr = fabsf(abserr / denom);

        bool relOk = denom != 0 ? relerr < RelErr : false;
        bool absOk = abserr < AbsErr;

        bool nanMatch = (IsNaN(a) && IsNaN(b)) || (!IsNaN(a) && !IsNaN(b));
        bool finiteMatch = (Finite(a) && Finite(b)) || (!Finite(a) && !Finite(b));

        return nanMatch && finiteMatch && ((relOk && absOk) || (EitherMetric && (relOk || absOk)));
    }

private:
    union ftoi
    {
        int i;
        unsigned int ui;
        float f;
    };

    // Quantize to keep 16 MSBs of the float
    float QuantizeRelative(const float a)
    {
        ftoi fi, fui;
        fi.f = a;

        fui.ui = 0xffff0000 & fi.ui;

        return fui.f;
    };

    // Quantize to the nearest 1/65536
    float QuantizeAbsolute(const float a)
    {
        float Cl = Saturate(a);
        float Sc = Cl * 65535.0f;
        unsigned short l = (unsigned short)(Sc);
        return float(l) / 65535.0f;
    };

    float RelErr, AbsErr;
    bool RelQuant, AbsQuant, EitherMetric;
};

//////////////////////////////////////
// Ways of sampling the variable domain

class SamplerLoopRegular
{
public:
    SamplerLoopRegular(VarVals_t vvals_, RelAbsComparator RAC_, float sampleStep_ = 0.001f) : sampleStep(sampleStep_), Comp(RAC_), vvals(vvals_) {}

    // Compares two expressions by stepping in regular intervals in the domain of the variables and returns the failing percentage
    double operator()(Expr* A, Expr* B)
    {
        int failed = 0, numSamples = 0;
        for (float y = 0; y <= 1; y += sampleStep) {
            vvals.vals[1] = y;
            for (float x = 0; x <= 1; x += sampleStep) {
                vvals.vals[0] = x;
                vvals.vals[2] = sqrtf(vvals.vals[0] * vvals.vals[0] + vvals.vals[1] * vvals.vals[1]);

                float aval = A->Eval(&vvals);
                float bval = B->Eval(&vvals);

                bool equal = Comp(aval, bval);
                numSamples++;

                if (!equal) {
                    std::cerr << "x=" << vvals.vals[0] << " y=" << vvals.vals[1] << " r=" << vvals.vals[2] << " aval=" << aval << " bval=" << bval << std::endl;

                    float av = A->Eval(&vvals);
                    float bv = B->Eval(&vvals);

                    bool equal = Comp(av, bv);

                    failed++;
                }
            }
        }

        if (failed > 0) {
            // std::cerr << failed << " of " << numSamples << " failed.\n";
            return failed / (double)numSamples;
        }

        return failed / (double)numSamples;
    }

private:
    float sampleStep;
    RelAbsComparator Comp;
    VarVals_t vvals;
};

class SamplerLoopRandom
{
public:
    SamplerLoopRandom(VarVals_t vvals_, RelAbsComparator RAC_, int numSamples_ = 100000) : numSamples(numSamples_), Comp(RAC_), vvals(vvals_) {}

    // Compares two expressions by sampling random points in the domain of the variables and returns the failing percentage
    double operator()(Expr* A, Expr* B)
    {
        int failed = 0;
        for (int i = 0; i < numSamples; i++) {
            for (size_t j = 0; j < vvals.vals.size(); j++) vvals.vals[j] = i == 0 ? 0.0f : DRandf();
            vvals.vals[2] = sqrtf(vvals.vals[0] * vvals.vals[0] + vvals.vals[1] * vvals.vals[1]);

            float aval = A->Eval(&vvals);
            float bval = B->Eval(&vvals);

            bool equal = Comp(aval, bval);

            // std::cerr << "?? x=" << vvals.vals[0] << " y=" << vvals.vals[1] << " r=" << vvals.vals[2] << " aval=" << aval << " bval=" << bval << std::endl;
            if (!equal) {
                std::cerr << "x=" << vvals.vals[0] << " y=" << vvals.vals[1] << " r=" << vvals.vals[2] << " aval=" << aval << " bval=" << bval << std::endl;

                float av = A->Eval(&vvals);
                float bv = B->Eval(&vvals);

                bool equal = Comp(av, bv);

                failed++;
            }
        }

        if (failed > 0) {
            // std::cerr << failed << " of " << numSamples << " failed.\n";
            return failed / (double)numSamples;
        }

        return failed / (double)numSamples;
    }

private:
    int numSamples;
    RelAbsComparator Comp;
    VarVals_t vvals;
};

class SamplerLoopInterval
{
public:
    SamplerLoopInterval(VarVals_t vvals_, RelAbsComparator RAC_, int numSamples_ = 100000, int Steps_ = 531) : numSamples(numSamples_), Steps(Steps_), Comp(RAC_), VV(vvals_) {}

    // Compares two expressions by stepping in regular intervals in the domain of the variables and returns the failing percentage
    double operator()(Expr* A, Expr* B)
    {
        int failed = 0;
        for (int i = 0; i < numSamples; i++) {
            if (!compareInterval(A)) failed++;
        }

        // if (failed > 0) {
        std::cerr << failed << " of " << numSamples << " intervals failed.\n";
        //}

        return failed / (double)numSamples;
    }

private:
    bool compareInterval(Expr* A)
    {
        opInfo opI;
        opI.steps = Steps;
        opI.maxAbsErr = 0.0f; // Unused since stopAtMaxErr is false
        opI.stopAtMaxErr = false;
        opI.vn = VV;

        // Test random intervals to make sure sampleIval matches Ival
        for (size_t i = 0; i < VarVals_t::NUM_VARS; i++) {
            opI.spans[i] = interval(randVal(), randVal());
            // std::cerr << i << ": " << tostring(opI.spans[i]) << '\n';
        }
        float inf = std::numeric_limits<float>::infinity();

        //             opI.spans[0] = interval(-0,119.29f);
        //             opI.spans[1] = interval(0,1);

        interval ival = A->Ival(opI);
        // std::cerr << "I: " << tostring(ival) << " = " << A->Print(PREFIX) << std::endl;

        interval sival = A->sIval(opI);
        // std::cerr << "S: " << tostring(sival) << " = " << A->Print(PREFIX) << std::endl;

        bool matches = num_match(ival.lower, sival.lower) && num_match(ival.upper, sival.upper);

        bool fits = matches && !ival.empty() && !sival.empty();
        if (!fits) {
            std::cerr << "FAIL: " << tostring(opI.spans[0]) << '\t' << tostring(opI.spans[1]) << "\tI: " << tostring(ival) << " S: " << tostring(sival) << " = " << A->Print(PREFIX) << '\n';
#if 0
            extern bool DEBUGprint;
            DEBUGprint = true;

            interval ival2 = A->Ival(opI);
            std::cerr << "Iv: " << tostring(ival2) << " = " << A->Print(PREFIX) << std::endl;

            interval sival2 = A->sIval(opI);
            std::cerr << "sI: " << tostring(sival2) << " = " << A->Print(PREFIX) << std::endl;

            bool matches2 = num_match(ival2.lower, sival2.lower) && num_match(ival2.upper, sival2.upper);
            bool fits2 = ival2.contains(sival2) && matches2;
            std::cerr << fits2 << '\n';

            DEBUGprint = false;
#endif
        }

        return fits;
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

    // True if they match
    bool num_match(float a, float b)
    {
        if (IsNaN(a) || IsNaN(b)) return false;

        const float inf = std::numeric_limits<float>::infinity();
        const float big = 64.0f;

        if (a == inf) return b > big;
        if (a == -inf) return b < -big;
        if (b == inf) return a > big;
        if (b == -inf) return a < -big;

        if (fabs(a) < big || fabs(b) < big) // Too big to care about accuracy?
            return Comp(a, b);
        else
            return (a > 0 && b > 0) || (a < 0 && b < 0); // Do their signs at least match?
    }

    int numSamples;
    int Steps;
    RelAbsComparator Comp;
    VarVals_t VV;
};

//////////////////////////////////////
// Result Actions (what to do when an expression matches or doesn't match

class ResultActionLogFile
{
public:
    ResultActionLogFile(bool PrintErrors_, bool PrintAll_, std::ofstream& GoodEqnFile_, std::ofstream& BadEqnFile_, std::ofstream& GoodOptEqnFile_, std::ofstream& BadOptEqnFile_) :
        maxFailPercent(0), m_tested(0), m_failed(0), m_printErrors(PrintErrors_), m_printAll(PrintAll_), m_goodEqnFile(GoodEqnFile_), m_badEqnFile(BadEqnFile_), m_goodOptEqnFile(GoodOptEqnFile_),
        m_badOptEqnFile(BadOptEqnFile_)
    {
    }

    void operator()(Expr* A, Expr* B, double failingPercent)
    {
        bool passed = failingPercent <= maxFailPercent;
        m_tested++;

        if (!passed) {
            m_failed++;
            if (m_printErrors) std::cerr << "Mismatch percentage: " << failingPercent << "\n";
        }

        if ((!passed && m_printErrors) || m_printAll) std::cerr << "A: " << A->Print(INFIX) << "\nB: " << B->Print(INFIX) << '\n';

        if (m_goodEqnFile.is_open() && passed) m_goodEqnFile << A->Print(PREFIX) << std::endl;
        if (m_badEqnFile.is_open() && !passed) m_badEqnFile << A->Print(PREFIX) << std::endl;
        if (m_goodOptEqnFile.is_open() && passed) m_goodOptEqnFile << B->Print(PREFIX) << std::endl;
        if (m_badOptEqnFile.is_open() && !passed) m_badOptEqnFile << B->Print(PREFIX) << std::endl;

        std::cerr << m_failed << " / " << m_tested << " equations failed\n";
    }

    void MaxFailPercent(double val) { maxFailPercent = val; }

private:
    double maxFailPercent;
    int m_tested, m_failed;
    bool m_printErrors, m_printAll;
    std::ofstream& m_goodEqnFile;
    std::ofstream& m_badEqnFile;
    std::ofstream& m_goodOptEqnFile;
    std::ofstream& m_badOptEqnFile;
};

}; // namespace

//////////////////////////////////////
// Main Test Harness

template <typename GTE, typename TI, typename SL, typename RA> void ExprOptTestHarness(GTE GetEx, TI TestItem, SL SamplerLoop, RA ResultAction)
{
    for (Expr* A; A = GetEx();) {
        // std::cerr << "Testing: " << A->Print(PREFIX) << std::endl;

        Expr* B = TestItem(A); // Make something to compare against

        if (B == NULL) {
            std::cerr << "Did not optimize: " << A->Print(PREFIX) << std::endl;
        } else {
            double failing = SamplerLoop(A, B); // Compare them across lots of samples

            ResultAction(A, B, failing); // Do stuff based on the result
        }

        delete A;
        delete B;
    }

    std::cerr << "Done\n";
}

void TestExpressions(const char* exprFName)
{
    std::cerr << "TestExpressions\n";

    ///////////////////////////////////////////////////////////////
    // Set up the kind of test to run

    VarVals_t vvals;
    InitVVals(vvals);

    TestItemOptimizer TestItem(vvals);
    // TestItemNull TestItem;

    // RelAbsComparator ErrorComparator(false, false, 0.4f, 0.1f, true); For interval testing
    RelAbsComparator ErrorComparator(false, false, 0.004f, 0.0001f, true);

    // SamplerLoopRegular SamplerLoop(ErrorComparator, 0.01f);
    SamplerLoopRandom SamplerLoop(vvals, ErrorComparator, 100000);
    // SamplerLoopInterval SamplerLoop(vvals, ErrorComparator, 10000, 191);

    std::ofstream GoodEqnFile("GoodExprs.txt");
    std::ofstream BadEqnFile("BadExprs.txt");
    std::ofstream GoodOptEqnFile("GoodOptExprs.txt");
    std::ofstream BadOptEqnFile("BadOptExprs.txt");

    ResultActionLogFile ResultAction(true, true, GoodEqnFile, BadEqnFile, GoodOptEqnFile, BadOptEqnFile);
    ResultAction.MaxFailPercent(0.001);

    ///////////////////////////////////////////////////////////////
    // Run the test

    if (exprFName) {
        std::ifstream InputEqnFile(exprFName);
        GetTestExprFromFile GetEx(vvals, InputEqnFile);
        std::cerr << "Reading input file: " << exprFName << std::endl;

        ExprOptTestHarness(GetEx, TestItem, SamplerLoop, ResultAction);
    } else {
        // GetTestExprFromList GetEx(vvals);
        GetTestExprRandom GetEx(vvals);

        ExprOptTestHarness(GetEx, TestItem, SamplerLoop, ResultAction);
    }

    exit(0);
}
