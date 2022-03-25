#include "ExprTools.h"

#include "BinaryExprSubclasses.h"
#include "Evaluator.h"
#include "NonaryExprSubclasses.h"
#include "UnaryExprSubclasses.h"

#include <Math/Random.h>
#include <Util/Assert.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <typeinfo>

Expr* ReadExpr(const std::string Str, std::string& remain, const VarVals_t* VV)
{
    size_t ind = int(Str.find_first_not_of(std::string(" \t\n")));
    if (ind == std::string::npos) { throw DMcError("Error parsing Expr: " + Str); }
    size_t ind2 = int(Str.find_first_of(std::string(" \t\n"), ind));
    std::string funame = (ind2 == std::string::npos) ? Str.substr(ind) : Str.substr(ind, ind2 - ind);

    std::string nokey;
    if (ind2 != std::string::npos) nokey = Str.substr(ind2);
    std::string secsub;
    Expr *A = NULL, *B = NULL;

    if (funame == Abs::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Abs(A);
    } else if (funame == And::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new And(A, B);
    } else if (funame == ATan2::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new ATan2(A, B);
    } else if (funame == ACos::name) {
        A = ReadExpr(nokey, remain, VV);
        return new ACos(A);
    } else if (funame == ASin::name) {
        A = ReadExpr(nokey, remain, VV);
        return new ASin(A);
    } else if (funame == ATan::name) {
        A = ReadExpr(nokey, remain, VV);
        return new ATan(A);
    } else if (funame == BitNot::name) {
        A = ReadExpr(nokey, remain, VV);
        return new BitNot(A);
    } else if (funame == Cbrt::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Cbrt(A);
    } else if (funame == Clamp::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Clamp(A);
    } else if (funame == Cos::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Cos(A);
    } else if (funame == Cube::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Cube(A);
    } else if (funame == Div::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new Div(A, B);
    } else if (funame == Exp::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Exp(A);
    } else if (funame == Ln::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Ln(A);
    } else if (funame == Minus::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new Minus(A, B);
    } else if (funame == Mod::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new Mod(A, B);
    } else if (funame == Mult::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new Mult(A, B);
    } else if (funame == Or::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new Or(A, B);
    } else if (funame == Plus::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new Plus(A, B);
    } else if (funame == Pow::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new Pow(A, B);
    } else if (funame == Round::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Round(A);
    } else if (funame == Sin::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Sin(A);
    } else if (funame == Sqr::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Sqr(A);
    } else if (funame == Sqrt::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Sqrt(A);
    } else if (funame == Tan::name) {
        A = ReadExpr(nokey, remain, VV);
        return new Tan(A);
    } else if (funame == UnaryMinus::name) {
        A = ReadExpr(nokey, remain, VV);
        return new UnaryMinus(A);
    } else if (funame == XOr::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new XOr(A, B);
    } else if (funame == IFS::name) {
        A = ReadExpr(nokey, secsub, VV);
        B = ReadExpr(secsub, remain, VV);
        return new IFS(A, B);
    } else {
        for (size_t i = 0; i < VV->names.size(); i++) {
            if (funame == VV->names[i]) {
                // XXX Ought to see if it's a valid identifier and make it a variable.
                remain = nokey;
                return new Var(funame, (int)i);
            }
        }

        // It must be a constant.
        remain = nokey;
        return new Const((float)atof(funame.c_str()));
    }
}

void CopyVValsToArray(const VarVals_t* VV, float* VVals)
{
    ASSERT_D(VV->vals.size() <= MAX_VARIABLES);
    for (size_t i = 0; i < VV->vals.size(); i++) VVals[i] = VV->vals[i];
}

void InitVVals(VarVals_t& VV)
{
    // Set up VV by copying from the static arrays
    const std::string VarNameList[] = {"x", "y", "r"}; // , "t" };
    const float VarValueList[] = {0.0f, 0.0f, 0.0f};   // , 0.5f };

    for (unsigned int i = 0; i < sizeof(VarValueList) / sizeof(float); i++) {
        VV.names.push_back(VarNameList[i]);
        VV.vals.push_back(VarValueList[i]);
    }
}

int Tokenize(const Expr* E, int* TokenStream, int MaxTokens)
{
    int NSize = E->postTokenStream(TokenStream, MaxTokens);
    ASSERT_D(NSize <= MaxTokens);
    return NSize;
}

void EvaluateTokens(int* Tok, float* VVals, float& rout, float& gout, float& bout)
{
    EvaluateTokenized(Tok, VVals[0], VVals[1], VVals[2], 0, rout, gout, bout);
}

float EvaluateHardcoded(const VarVals_t* VV)
{
    float x = VV->vals[0];
    float y = VV->vals[1];
    float r = VV->vals[2];

    float R = eMult(x, y); // Replace with some expression that you want to evaluate on CPU
    return R;
}

bool DEBUGprint = false;

namespace {

class SampleSet
{
public:
    SampleSet() { init(); }

    SampleSet(const interval& ivl_, int span) : ivl(ivl_)
    {
        if (DEBUGprint) std::cerr << "\nSampleSet: ";
        if (ivl.empty()) {
            ivl.extend(0);
            insert(0);
        } else {
            // Do something more complex here with Halton
            init();
            if (DEBUGprint) std::cerr << "\n\n";
            insert_span(ivl, span);
        }
    }

    void init()
    {
        insert(ivl.lower); // Handles endpoints of 0.0 and -0.0.
        insert(ivl.upper);
        if (ivl.lower < 0) insert(-0.0f, true); // Handles midpoints of 0.0 and -0.0.
        if (ivl.upper > 0) insert(0.0f, true);
        insert(std::numeric_limits<float>::min());
        insert(-std::numeric_limits<float>::min());
        insert(std::numeric_limits<float>::infinity());
        insert(-std::numeric_limits<float>::infinity());
        insert(99999.0f);  // Need a large, positive number as exponent to get span of -infinity for negative bases with pow.
        insert(-99999.0f); // Need a large, negative, odd number as exponent to get span of infinity for pow.

        static const float numerators[] = {1.0f, 2.0f, E_E, 3.0f, E_PI, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        for (int n = 0; n < sizeof(numerators) / sizeof(float); n++) { denom(numerators[n]); }

        // std::sort(m_table.begin(), m_table.end());
    }

    void insert_span(interval m, int steps, bool log_scale = false)
    {
        float mn = m.lower_finite();
        float mx = m.upper_finite();
        ASSERT_D(mn <= mx);
        ASSERT_D(!IsNaN(mn) && !IsNaN(mx));

        insert(mn);
        insert(mx);

        float span = mx - mn;
        if (span < 1e-3f) return;

        if (fabs(span) > 1000.0f) {
            if (mn < 0) {
                if (mx > 0) {
                    mx = std::max(fabs(mn), fabs(mx));
                    mn = 0.01f;
                } else {
                    float t = -mn;
                    mn = -mx;
                    mx = t;
                }
            }
            ASSERT_D(mn >= 0 && mx >= 0);

            float f = mn + 0.1f;
            for (int i = 0; i < steps && f < mx; i++) {
                f *= 1.21f + float(i);
                insert(f);
                insert(-f);
            }
        } else {
            // Compute interval across its range
            float fStep = span / float(steps);
            for (float f = mn; f <= mx; f += fStep) { insert(f); }
        }
    }

    std::vector<float> m_table;
    interval ivl;

private:
    void insert(const float v, bool suppress_dupl_check = false)
    {
        if (!ivl.contains(v)) return;

        for (float f : m_table)
            if (f == v && !suppress_dupl_check) return;
        m_table.push_back(v);
        if (DEBUGprint) std::cerr << v << ' ';
    }

    void denom(float n)
    {
        static const float denominators[] = {1.0f, 2.0f, E_E, 3.0f, E_PI, 4.0f, 5.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        for (int d = 0; d < sizeof(denominators) / sizeof(float); d++) {
            float v = n / denominators[d];
            insert(v);
            insert(-v);
        }
    }
};

}; // namespace

interval sampleIval(const Expr* E, const opInfo& opI, const interval& lv, const interval& rv)
{
    ASSERT_D(VarVals_t::NUM_VARS == 3);

    unsigned int hasV = E->HasVars();
    interval iout;
    VarVals_t VV = opI.vn;

    SampleSet xSS((hasV & 1) ? interval(opI.spans[0]) : interval(), opI.steps);
    if (hasV & 2) {
        xSS.insert_span(opI.spans[1], 1);
        xSS.insert_span(-opI.spans[1], 1);
    }
    if (hasV & 4) {
        xSS.insert_span(opI.spans[2], 1);
        xSS.insert_span(-opI.spans[2], 1);
    }
    SampleSet ySS((hasV & 2) ? interval(opI.spans[1]) : interval(), opI.steps);
    if (hasV & 1) {
        ySS.insert_span(opI.spans[0], 1);
        ySS.insert_span(-opI.spans[0], 1);
    }
    if (hasV & 4) {
        ySS.insert_span(opI.spans[2], 1);
        ySS.insert_span(-opI.spans[2], 1);
    }
    SampleSet rSS((hasV & 4) ? interval(opI.spans[2]) : interval(), opI.steps);
    if (hasV & 1) {
        rSS.insert_span(opI.spans[0], 1);
        rSS.insert_span(-opI.spans[0], 1);
    }
    if (hasV & 2) {
        rSS.insert_span(opI.spans[1], 1);
        rSS.insert_span(-opI.spans[1], 1);
    }

    for (float x : xSS.m_table) {
        VV.vals[0] = x;
        for (float y : ySS.m_table) {
            VV.vals[1] = y;
            for (float r : rSS.m_table) {
                VV.vals[2] = r;

                float v = E->Eval(&VV);
                if (IsNaN(v)) {
                    if (DEBUGprint) std::cerr << "f(" << VV.vals[0] << "," << VV.vals[1] << "," << VV.vals[2] << ") = " << v << " => " << tostring(iout) << '\n';
                    continue;
                }

                iout.extend(v);
                if (DEBUGprint) std::cerr << "f(" << VV.vals[0] << "," << VV.vals[1] << "," << VV.vals[2] << ") = " << v << " => " << tostring(iout) << '\n';

                if (opI.stopAtMaxErr && iout.span() >= opI.maxAbsErr) break;
            }
        }
    }

    if (iout.empty()) iout.extend(0);

    return iout;
}

// Optimization:
// External code calls the SymOptimize() entry point, which makes a copy and loops over the optimizer.
// Optimize bottom-up: Calls Opt() on the root. Every Opt() function optimizes its children first.
// This gives optimized subtrees to work with.
// Opt() returns a pointer to the replacement or NULL if no optimization happened.
// Opt() may return "this" if optimization succeeds, or it will create a replacement for itself.
// It may grab its children and attach them to the replacement.
// It may delete old children and attach their replacements.
// Opt() can call GrabR() and GrabL() on a descendant to have that descendant detach its child and return it.
// Only do this when the descendant itself is about to be deleted.
// Since Opt() may call GrabRL() on itself it must immediately return so as not to access the left and right pointers, which are now NULL.

Expr* Optimize(const Expr* E, const VarVals_t& MinVV, const VarVals_t& MaxVV, const int steps, const float maxAbsErr, const interval outSpan)
{
    Expr* F = E->Copy();
    int sizePre = E->size();

    opInfo opI;
    opI.vn = MinVV;
    for (int phase = 0; phase < VarVals_t::NUM_VARS; phase++) opI.spans[phase] = interval(MinVV.vals[phase], MaxVV.vals[phase]);
    opI.steps = steps;
    opI.maxAbsErr = maxAbsErr;

    for (opI.phase = 0; opI.phase < 4; opI.phase++) {
        int j = 0;
        Expr* A = NULL;
        do {
            // std::cerr << "T: " << F->Print(PREFIX) << std::endl;
            A = F->Opt(opI);

            if (A && A != F) {
                A->ivl = F->ivl; // It's returning a new node that hasn't had the interval set
                delete F;
                F = A;
            }
        } while ((A || j < 15) && j++ < 40);
    }

    // Clamp to the color space's interval
    if (F->ivl.upper <= outSpan.lower) F->ivl = interval(outSpan.lower);
    if (F->ivl.lower >= outSpan.upper) F->ivl = interval(outSpan.upper);

    if (F->ivl.lower == F->ivl.upper) {
        delete F;
        F = new Const(F->ivl.lower);
        F->ivl = F->Ival(opI);
    }

    // Not sure why this has to be done, but somehow it gets inaccurate.
    F->ivl = F->Ival(opI);

    int sizePost = F->size();

    if (sizePre < sizePost) {
        // std::cerr << "BAD:\npre [" << sizePre << "]: " << E->Print(PREFIX) << '\n';
        // std::cerr << "pst [" << sizePost << "]: " << F->Print(PREFIX) << '\n';
        delete F;
        F = E->Copy();
    } else if (sizePre > sizePost) {
        // std::cerr << "GOOD:\npre [" << sizePre << "]: " << E->Print(PREFIX) << '\n';
        // std::cerr << "pst [" << sizePost << "]: " << F->Print(PREFIX) << '\n';
    }

    return F;
}

Expr* RandNonaryNode(const VarVals_t* VV)
{
    Expr* E;

    if (chance(1, 2)) {
        size_t a = randn((int)VV->vals.size());
        E = new Var(VV->names[a], (int)a);
    } else {
        Const* C = new Const();
        C->UniformRandom();
        E = C;
    }

    return E;
}

Expr* RandUnaryNode(Expr* A)
{
    Expr* E;

    int ftype = Abs_e + randn(NUM_UNARY_EXPR_OPCODES);
    switch (ftype) {
    case Abs_e: E = new Abs(A); break;
    case ACos_e: E = new ACos(A); break;
    case ASin_e: E = new ASin(A); break;
    case ATan_e: E = new ATan(A); break;
    case BitNot_e: E = new BitNot(A); break;
    case Cbrt_e: E = new Cbrt(A); break;
    case Clamp_e: E = new Clamp(A); break;
    case Cos_e: E = new Cos(A); break;
    case Cube_e: E = new Cube(A); break;
    case Exp_e: E = new Exp(A); break;
    case Ln_e: E = new Ln(A); break;
    case Round_e: E = new Round(A); break;
    case Sin_e: E = new Sin(A); break;
    case Sqr_e: E = new Sqr(A); break;
    case Sqrt_e: E = new Sqrt(A); break;
    case Tan_e: E = new Tan(A); break;
    case UnaryMinus_e: E = new UnaryMinus(A); break;
    default: ASSERT_R("Unknown random value"); E = NULL;
    }

    return E;
}

Expr* RandBinaryNode(Expr* A, Expr* B)
{
    Expr* E;

    int ftype = And_e + randn(NUM_BINARY_EXPR_OPCODES);
    switch (ftype) {
    case And_e: E = new And(A, B); break;
    case ATan2_e: E = new ATan2(A, B); break;
    case Div_e: E = new Div(A, B); break;
    case IFS_e: E = new IFS(A, B); break;
    case Minus_e: E = new Minus(A, B); break;
    case Mod_e: E = new Mod(A, B); break;
    case Mult_e: E = new Mult(A, B); break;
    case Or_e: E = new Or(A, B); break;
    case Plus_e: E = new Plus(A, B); break;
    case Pow_e: E = new Pow(A, B); break;
    case XOr_e: E = new XOr(A, B); break;
    default: ASSERT_R("Unknown random value"); E = NULL;
    }

    return E;
}

Expr* RandExpr(int siz, const VarVals_t* VV)
{
    // if siz<=1 creates a Var or Const.
    // if siz>=2 creates one random symbol and children.

    if (siz <= 1) { return RandNonaryNode(VV); }

    siz--; // To account for this one

    if (chance(1, 2)) {
        Expr* A = RandExpr(siz, VV);
        return RandUnaryNode(A);
    } else {
        Expr* A = RandExpr(siz / 2, VV);
        Expr* B = RandExpr(siz / 2, VV);
        return RandBinaryNode(A, B);
    }
}

Expr* MutateExpr(const Expr* A, const int prob, const int siz, const float ConstPerturb, const VarVals_t* VV)
{
    // std::cerr << "Starting MutateExpr: " << prob << '\n';
    ASSERT_D(A);
    Expr* F = A->Copy();

    Expr* L = F->Mutate(prob, siz, ConstPerturb, VV);
    if (L && L != F) {
        delete F;
        F = L;
    }

    return F;
}

Expr* BlendExprs(const Expr* A, const Expr* B)
{
    Expr* E = NULL;
    Expr* Ac = A->Copy();
    Expr* Bc = B->Copy();
    // std::cerr << Ac->Print(PREFIX) << std::endl << Bc->Print(PREFIX) << std::endl;

    int ftype = And_e + randn(NUM_BINARY_EXPR_OPCODES);
    switch (ftype) {
    case And_e: E = new And(Ac, Bc); break;
    case ATan2_e: E = new ATan2(Ac, Bc); break;
    case Div_e: E = new Div(Ac, Bc); break;
    case IFS_e: E = new IFS(Ac, Bc); break;
    case Minus_e: E = new Minus(Ac, Bc); break;
    case Mod_e: E = new Mod(Ac, Bc); break;
    case Mult_e: E = new Mult(Ac, Bc); break;
    case Or_e: E = new Or(Ac, Bc); break;
    case Plus_e: E = new Plus(Ac, Bc); break;
    case Pow_e: E = new Pow(Ac, Bc); break;
    case XOr_e: E = new XOr(Ac, Bc); break;
    default: ASSERT_R("Unknown random value"); E = NULL;
    }

    // std::cerr << "Blend: " << E->Print(PREFIX) << std::endl;

    return E;
}

Expr* CrossExprs(const Expr* A, const Expr* B)
{
    Expr* C = A->Copy();

    // std::cerr<<"Copy: " << C->Print(PREFIX) << std::endl;

    // This is what will be grafted in.
    int count = 0;
    Expr* NewExpr;
    Expr* const* NewExprP = B->FindRand(count);
    if (NewExprP == NULL)
        NewExpr = B->Copy();
    else
        NewExpr = (*NewExprP)->Copy();

    // std::cerr << "Adding in: " << NewExpr->Print(PREFIX) << std::endl;

    // Find a branch to remove.
    Expr** SnipP = (Expr**)(C->FindRand(count));
    if (SnipP) {
        // std::cerr << "Burning: " << (*SnipP)->Print(PREFIX) << std::endl;
        delete *SnipP;
        *SnipP = NewExpr;
    } else {
        // std::cerr << "Burning the whole thing.\n";
        delete C;
        C = NewExpr;
    }

    return C;
}

Expr* ReplaceVarWithConst(Expr* E, int VarID, float val)
{
    if (typeid(*E) == typeid(Var) && ((const Var*)E)->getVarID() == size_t(VarID)) return new Const(val);

    Expr* L = E->left ? ReplaceVarWithConst(E->left, VarID, val) : NULL;
    if (L && L != E->left) {
        delete E->left;
        E->left = L;
    }
    Expr* R = E->right ? ReplaceVarWithConst(E->right, VarID, val) : NULL;
    if (R && R != E->right) {
        delete E->right;
        E->right = R;
    }

    return (L || R) ? E : NULL;
}

// Replace variable in in-place expression
Expr* ReplaceVars(Expr* E, const VarVals_t& reVV)
{
    if (typeid(*E) == typeid(Var)) {
        Var* V = dynamic_cast<Var*>(E);
        int oVID = (int)V->getVarID();
        return new Var(reVV.names[oVID], reVV.vals[oVID]);
    } else {
        Expr* L = E->left ? ReplaceVars(E->left, reVV) : NULL;
        if (L && L != E->left) {
            delete E->left;
            E->left = L;
        }
        Expr* R = E->right ? ReplaceVars(E->right, reVV) : NULL;
        if (R && R != E->right) {
            delete E->right;
            E->right = R;
        }

        return (L || R) ? E : NULL;
    }
}

Expr* MakeConst(const float v)
{
    return new Const(v);
}

bool IsConst(const Expr* E)
{
    return typeid(*E) == typeid(Const);
}

namespace {
Expr* FixIFS(Expr* E, bool InIFS)
{
    if (typeid(*E) == typeid(IFS)) {
        if (InIFS) {
            Expr* P = new Plus(E->GrabL(), E->GrabR());
            Expr* Q = FixIFS(P, true);
            if (Q && Q != P) {
                delete P;
                P = Q;
            }
            return Q;
        } else {
            Expr* L = E->left ? FixIFS(E->left, true) : NULL;
            if (L && L != E->left) {
                delete E->left;
                E->left = L;
            }
            Expr* R = E->right ? FixIFS(E->right, true) : NULL;
            if (R && R != E->right) {
                delete E->right;
                E->right = R;
            }
            return E;
        }
    } else {
        Expr* L = E->left ? FixIFS(E->left, InIFS) : NULL;
        if (L && L != E->left) {
            delete E->left;
            E->left = L;
        }
        Expr* R = E->right ? FixIFS(E->right, InIFS) : NULL;
        if (R && R != E->right) {
            delete E->right;
            E->right = R;
        }
        return E;
    }
}
}; // namespace

Expr* RemoveNestedIFS(const Expr* E)
{
    Expr* F = E->Copy();

    Expr* A = FixIFS(F, false);
    if (A && A != F) {
        delete F;
        F = A;
    }

    return F;
}

// If the root node is already Mult we could just modify the existing multiplicand
Expr* ScaleBias(Expr* E, const float scale, const float bias)
{
    Expr* F = E->Copy();
    if (scale == 1.0f)
        return new Plus(F, new Const(bias));
    else if (bias == 1.0f)
        return new Mult(new Const(scale), F);
    else
        return new Plus(new Mult(new Const(scale), F), new Const(bias));
}
