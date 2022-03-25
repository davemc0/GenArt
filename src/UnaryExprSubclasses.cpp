#include "UnaryExprSubclasses.h"

#include "BinaryExprSubclasses.h"
#include "ExprImplementations.h"
#include "ExprTools.h"
#include "IntervalImplementations.h"
#include "NonaryExprSubclasses.h"

#include <Math/Random.h>
#include <Util/Assert.h>

int Abs::token = Abs_e;
int ACos::token = ACos_e;
int ASin::token = ASin_e;
int ATan::token = ATan_e;
int BitNot::token = BitNot_e;
int Cbrt::token = Cbrt_e;
int Clamp::token = Clamp_e;
int Cos::token = Cos_e;
int Cube::token = Cube_e;
int Exp::token = Exp_e;
int Ln::token = Ln_e;
int Round::token = Round_e;
int Sin::token = Sin_e;
int Sqr::token = Sqr_e;
int Sqrt::token = Sqrt_e;
int Tan::token = Tan_e;
int UnaryMinus::token = UnaryMinus_e;

std::string ACos::name = "acos";
std::string ASin::name = "asin";
std::string ATan::name = "atan";
std::string Abs::name = "abs";
std::string BitNot::name = "~";
std::string Cbrt::name = "cbrt";
std::string Clamp::name = "clamp";
std::string Cos::name = "cos";
std::string Cube::name = "cube";
std::string Exp::name = "exp";
std::string Ln::name = "ln";
std::string Round::name = "round";
std::string Sin::name = "sin";
std::string Sqr::name = "sqr";
std::string Sqrt::name = "sqrt";
std::string Tan::name = "tan";
std::string UnaryMinus::name = "---";

std::string ACos::fname = "eACos";
std::string ASin::fname = "eASin";
std::string ATan::fname = "eATan";
std::string Abs::fname = "eAbs";
std::string BitNot::fname = "eBitNot";
std::string Cbrt::fname = "eCbrt";
std::string Clamp::fname = "eClamp";
std::string Cos::fname = "eCos";
std::string Cube::fname = "eCube";
std::string Exp::fname = "eExp";
std::string Ln::fname = "eLn";
std::string Round::fname = "eRound";
std::string Sin::fname = "eSin";
std::string Sqr::fname = "eSqr";
std::string Sqrt::fname = "eSqrt";
std::string Tan::fname = "eTan";
std::string UnaryMinus::fname = "eUnaryMinus";

std::string UnaryExpr::Print(int pstyle) const
{
    // For function calls
    if (pstyle & PREFIX)
        return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle);
    else
        return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + "(" + left->Print(pstyle) + ")";
}

int UnaryExpr::preTokenStream(int* TokenStream, const int max_len) const
{
    *(TokenStream) = getToken();
    return 1 + left->preTokenStream(TokenStream + 1, max_len - 1);
}

int UnaryExpr::postTokenStream(int* TokenStream, const int max_len) const
{
    int cnt = left->postTokenStream(TokenStream, max_len - 1);
    ASSERT_R(cnt < max_len);
    TokenStream[cnt] = getToken();
    return cnt + 1;
}

bool UnaryExpr::isequal(const Expr* E) const { return typeid(*E) == typeid(*this) && left->isequal(E->left); }

bool UnaryExpr::isless(const Expr* E) const
{
    if (getToken() == E->getToken())
        return left->isless(E->left);
    else
        return getToken() < E->getToken();
}

Expr* UnaryExpr::OptHelp(const opInfo& opI)
{
    Expr* L = left->Opt(opI);
    if (L && L != left) {
        L->ivl = left->ivl;
        delete left;
        left = L;
    }

    init(left);
    ivl = Ival(opI, left->ivl);
    // std::cerr << "Una " << count << tostring(ivl) << Print(PREFIX) << '\n';
    ASSERT_D(!IsNaN(ivl.lower) && !IsNaN(ivl.upper));
    ASSERT_D(!ivl.empty());

    // If child is Const then I should be Const, too.
    if (typeid(*left) == typeid(Const) && FOG(AA)) {
        float value = Eval();
        return new Const(value);
    }

    // If interval is flat over the range, return a constant with the same value as this expression.
    if (ivl.span() <= opI.maxAbsErr && FOG(AI)) {
        // std::cerr << "\nUConst: " << count << tostring(ivl) << ' ' << Print(PREFIX) << '\n';
        ivl = interval(ivl.lower); // Ivl is a return value, so make it accurate.
        return new Const(ivl.lower);
    }

    // If something changed with a child, this is still valid.
    return L ? this : NULL;
}

void UnaryExpr::init(Expr* E)
{
    left = E;
    count = left->size() + 1;
    hasVars = left->hasVars;
    // Ivl will still be empty until Opt or Ival is called.
}

Expr* UnaryExpr::Mutate(const int prob, const int siz, const float ConstPerturb, const VarVals_t* VV)
{
    ASSERT_D(left != NULL && right == NULL);
    ASSERT_D(prob >= 1);

    // Mutate children and hook them in. Delete old children if necessary.
    // Might waste some work if this one, when modified, doesn't use its children.
    Expr* L = left->Mutate(prob, siz, ConstPerturb, VV);
    if (L && L != left) {
        delete left;
        left = L;
    }

    // Only a one in prob chance of mutating this node
    if (!chance(1, prob)) return this;

    int S = randn(4);
    switch (S) {
    case 0: // Replace this with my child
        return GrabL();
    case 1: // Attach child to a new one
        return RandUnaryNode(GrabL());
    case 2: // Make self a child of a new one
        if (chance(1, 2)) {
            return RandUnaryNode(Copy());
        } else {
            return chance(1, 2) ? RandBinaryNode(Copy(), RandExpr(siz, VV)) : RandBinaryNode(RandExpr(siz, VV), Copy());
        }
    case 3: // Replace with new expression
        return RandExpr(siz, VV);
    }

    ASSERT_R(0);
    return NULL;
}

float Abs::Eval(const VarVals_t* VV /*= NULL*/) const { return eAbs(left->Eval(VV)); }

interval Abs::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iAbs(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Abs::Copy() const
{
    Expr* E = new Abs(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Abs::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // This Abs isn't needed for expressions that are always >= 0.
    if (typeid(*left) == typeid(Abs) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(BitNot) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(Clamp) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(Exp) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(Round) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(Sqr) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(Sqrt) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(UnaryMinus) && FOG(AB)) return new Abs(left->GrabL());
    if (typeid(*left) == typeid(And) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(IFS) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(Or) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(XOr) && FOG(AB)) return GrabL();

    if (left->ivl.lower >= 0.0f && FOG(AB)) return GrabL(); // Don't need Abs if the variable will always be non-negative.

    if (left->ivl.upper <= 0.0f && FOG(AB)) return new UnaryMinus(GrabL());

    return E;
}

float ACos::Eval(const VarVals_t* VV /*= NULL*/) const { return eACos(left->Eval(VV)); }

interval ACos::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iACos(lv.empty() ? left->Ival(opI) : lv);
}

Expr* ACos::Copy() const
{
    Expr* E = new ACos(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* ACos::Opt(const opInfo& opI)
{
    interval lv;
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(Cos) && FOG(AA)) return left->GrabL();

    return E;
}

float ASin::Eval(const VarVals_t* VV /*= NULL*/) const { return eASin(left->Eval(VV)); }

interval ASin::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iASin(lv.empty() ? left->Ival(opI) : lv);
}

Expr* ASin::Copy() const
{
    Expr* E = new ASin(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* ASin::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(Sin) && FOG(AA)) return left->GrabL();

    return E;
}

float ATan::Eval(const VarVals_t* VV /*= NULL*/) const { return eATan(left->Eval(VV)); }

interval ATan::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iATan(lv.empty() ? left->Ival(opI) : lv);
}

Expr* ATan::Copy() const
{
    Expr* E = new ATan(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* ATan::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(Tan) && FOG(AA)) return left->GrabL();

    return E;
}

std::string BitNot::Print(int pstyle) const
{
    // For unary impure operators
    if (pstyle & PREFIX) return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle);

    if (pstyle & (OP_EVAL | OP_EVAL_IMPURE)) {
        return getFuncName() + "(" + left->Print(pstyle) + ")";
    } else {
        bool lp = left->getArity() > 1 && left->getPrecedence() >= getPrecedence();

        return getName() + (lp ? "(" : "") + left->Print(pstyle) + (lp ? ")" : "");
    }
}

float BitNot::Eval(const VarVals_t* VV /*= NULL*/) const { return eBitNot(left->Eval(VV)); }

interval BitNot::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iBitNot(lv.empty() ? left->Ival(opI) : lv);
}

Expr* BitNot::Copy() const
{
    Expr* E = new BitNot(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* BitNot::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // ~~A => A, except that BitNot clamps so keep the clamp.
    if (typeid(*left) == typeid(BitNot) && FOG(AB)) return new Clamp(left->GrabL());

    // ~Clamp(A) => ~A because ~ clamps its input.
    if (typeid(*left) == typeid(Clamp) && FOG(AB)) return new BitNot(left->GrabL());

    // ~(~A xor B) => A xor B
    if (typeid(*left) == typeid(XOr) && typeid(*(left->left)) == typeid(BitNot) && FOG(AB)) return new XOr(left->left->GrabL(), left->GrabR());

    // ~(A xor B) => ~A xor B
    if (typeid(*left) == typeid(XOr) && FOG(NE)) return new XOr(new BitNot(left->GrabL()), left->GrabR());

    return E;
}

float Cbrt::Eval(const VarVals_t* VV /*= NULL*/) const { return eCbrt(left->Eval(VV)); }

interval Cbrt::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iCbrt(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Cbrt::Copy() const
{
    Expr* E = new Cbrt(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Cbrt::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(Cube) && FOG(AB)) return left->GrabL();

    // Cbrt(Round(A)) => Round(A) because Cbrt(0)==0 and Cbrt(1)==1
    if (typeid(*left) == typeid(Round) && FOG(AB)) return GrabL();

    if (FOG(WE)) return new Pow(GrabL(), new Const(1.0f / 3.0f));

    return E;
}

float Clamp::Eval(const VarVals_t* VV /*= NULL*/) const { return eClamp(left->Eval(VV)); }

interval Clamp::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iClamp(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Clamp::Copy() const
{
    Expr* E = new Clamp(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Clamp::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // Clamp isn't needed for expressions that are always on 0..1.
    if (typeid(*left) == typeid(And) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(BitNot) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(Clamp) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(IFS) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(Or) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(Round) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(XOr) && FOG(AB)) return GrabL();

    if (interval(0, 1).contains(left->ivl) && FOG(AI)) {
        return GrabL(); // Don't need to clamp if the variable will always be in range.
    }

    return E;
}

float Cos::Eval(const VarVals_t* VV /*= NULL*/) const { return eCos(left->Eval(VV)); }

interval Cos::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iCos(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Cos::Copy() const
{
    Expr* E = new Cos(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Cos::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(ACos) && FOG(AA)) return left->GrabL();

    // Cos(-A) => Cos(A)
    if (typeid(*left) == typeid(UnaryMinus) && FOG(AB)) return new Cos(left->GrabL());

    return E;
}

float Cube::Eval(const VarVals_t* VV /*= NULL*/) const { return eCube(left->Eval(VV)); }

interval Cube::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iCube(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Cube::Copy() const
{
    Expr* E = new Cube(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Cube::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(Cbrt) && FOG(AA)) return left->GrabL();

    // Cube(Round(A)) => Round(A) because Cube(0)==0 and Cube(1)==1
    if (typeid(*left) == typeid(Round) && FOG(AB)) return GrabL();

    if (FOG(WE)) return new Pow(GrabL(), new Const(3.0f));

    return E;
}

float Exp::Eval(const VarVals_t* VV /*= NULL*/) const { return eExp(left->Eval(VV)); }

interval Exp::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iExp(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Exp::Copy() const
{
    Expr* E = new Exp(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Exp::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // Exp(Ln(X)) = X
    // We really compute ln(abs(x)), so optimized doesn't match.
    if (typeid(*left) == typeid(Ln) && FOG(AA)) return left->GrabL();

    if (FOG(WE)) { return new Pow(new Const(E_E), GrabL()); }

    return E;
}

float Ln::Eval(const VarVals_t* VV /*= NULL*/) const { return eLn(left->Eval(VV)); }

interval Ln::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iLn(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Ln::Copy() const
{
    Expr* E = new Ln(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Ln::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // Ln(Exp(X)) => X
    if (typeid(*left) == typeid(Exp) && FOG(AA)) return left->GrabL();

    // Ln(A^B) => B*Ln(A)
    // Sometimes changes the result, such as ifs ln pow x y r
    // ln cbrt x => ln pow x 0.33333 => * 0.33333 ln x, but this misses clamping x to 0 in cbrt.
    if (typeid(*left) == typeid(Pow) && FOG(NL)) return new Mult(left->GrabR(), new Ln(left->GrabL()));

    if (typeid(*left) == typeid(Abs) && FOG(AB)) // ELn implicitly does Abs.
        return new Ln(left->GrabL());

    return E;
}

float Round::Eval(const VarVals_t* VV /*= NULL*/) const { return eRound(left->Eval(VV)); }

interval Round::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iRound(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Round::Copy() const
{
    Expr* E = new Round(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Round::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(Round) && FOG(AB)) return GrabL();
    if (typeid(*left) == typeid(Clamp) && FOG(AB)) return new Round(left->GrabL());

    return E;
}

float Sin::Eval(const VarVals_t* VV /*= NULL*/) const { return eSin(left->Eval(VV)); }

interval Sin::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iSin(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Sin::Copy() const
{
    Expr* E = new Sin(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Sin::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(ASin)) return left->GrabL();

    if (typeid(*left) == typeid(UnaryMinus) && FOG(NL)) return new UnaryMinus(new Sin(left->GrabL())); // Sin(-A) => -Sin(A)

    return E;
}

float Sqr::Eval(const VarVals_t* VV /*= NULL*/) const { return eSqr(left->Eval(VV)); }

interval Sqr::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iSqr(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Sqr::Copy() const
{
    Expr* E = new Sqr(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Sqr::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // We really compute sqrt(abs(x)), so doesn't match when optimized.
    if (typeid(*left) == typeid(Sqrt) && FOG(AA)) return left->GrabL();

    // Sqr(Round(A)) => Round(A) because Sqr(0)==0 and Sqr(1)==1
    if (typeid(*left) == typeid(Round) && FOG(AB)) return GrabL();

    // Sqr(-A) => Sqr(A)
    if (typeid(*left) == typeid(UnaryMinus) && FOG(AB)) return new Sqr(left->GrabL());

    // Sqr(Abs(A)) => Sqr(A)
    if (typeid(*left) == typeid(Abs) && FOG(AB)) return new Sqr(left->GrabL());

    // Sqr(A*B) => Sqr(A) * Sqr(B)
    if (typeid(*left) == typeid(Mult) && FOG(WE)) return new Mult(new Sqr(left->GrabL()), new Sqr(left->GrabR()));

    if (FOG(WE)) return new Pow(GrabL(), new Const(2.0f));

    return E;
}

float Sqrt::Eval(const VarVals_t* VV /*= NULL*/) const { return eSqrt(left->Eval(VV)); }

interval Sqrt::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iSqrt(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Sqrt::Copy() const
{
    Expr* E = new Sqrt(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Sqrt::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // We really compute sqrt(abs(x)), so optimized doesn't match.
    if (typeid(*left) == typeid(Sqr) && FOG(AA)) return left->GrabL();
    if (typeid(*left) == typeid(Abs) && FOG(AB)) return new Sqrt(left->GrabL());

    // Sqrt(Round(A)) => Round(A) because Sqrt(0)==0 and Sqrt(1)==1
    if (typeid(*left) == typeid(Round) && FOG(AB)) return GrabL();

    if (FOG(WE)) return new Pow(new Abs(GrabL()), new Const(0.5f));

    return E;
}

float Tan::Eval(const VarVals_t* VV /*= NULL*/) const { return eTan(left->Eval(VV)); }

interval Tan::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iTan(lv.empty() ? left->Ival(opI) : lv);
}

Expr* Tan::Copy() const
{
    Expr* E = new Tan(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* Tan::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(ATan) && FOG(AA)) return left->GrabL();

    // tan(atan2(x,y)) => tan(atan(x/y)) => x/y
    if (typeid(*left) == typeid(ATan2) && FOG(AA)) return new Div(left->GrabL(), left->GrabR());

    return E;
}

std::string UnaryMinus::Print(int pstyle) const
{
    // For unary pure operators
    if (pstyle & PREFIX) return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle);

    if (pstyle & (OP_EVAL)) {
        return getFuncName() + "(" + left->Print(pstyle) + ")";
    } else {
        bool lp = left->getArity() > 1 && left->getPrecedence() >= getPrecedence();

        return std::string("-") + (lp ? "(" : "") + left->Print(pstyle) + (lp ? ")" : "");
    }
}

float UnaryMinus::Eval(const VarVals_t* VV /*= NULL*/) const { return eUnaryMinus(left->Eval(VV)); }

interval UnaryMinus::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iUnaryMinus(lv.empty() ? left->Ival(opI) : lv);
}

Expr* UnaryMinus::Copy() const
{
    Expr* E = new UnaryMinus(*this);
    if (left) E->left = left->Copy();
    return E;
}

Expr* UnaryMinus::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(UnaryMinus)) return left->GrabL();

    // -(A+B) => (-A)-B
    if (typeid(*left) == typeid(Plus) && FOG(NE)) return new Minus(new UnaryMinus(left->GrabL()), left->GrabR());
    // -(A-B) => (-A)+B
    if (typeid(*left) == typeid(Minus) && FOG(NE)) return new Plus(new UnaryMinus(left->GrabL()), left->GrabR());
    // -(A/B) => (-A)/B
    if (typeid(*left) == typeid(Div) && FOG(NE)) return new Div(new UnaryMinus(left->GrabL()), left->GrabR());
    // -(A/B) => A/(-B)
    if (typeid(*left) == typeid(Div) && FOG(NE)) return new Div(left->GrabL(), new UnaryMinus(left->GrabR()));
    // -(A*B) => A*(-B)
    if (typeid(*left) == typeid(Mult) && FOG(NE)) return new Mult(left->GrabL(), new UnaryMinus(left->GrabR()));
    // -Sin(A) => Sin(-A)
    if (typeid(*left) == typeid(Sin) && FOG(NE)) return new Sin(new UnaryMinus(left->GrabL()));

    // -(A+B) => (-A)+(-B)
    if (typeid(*left) == typeid(Plus) && FOG(WE)) return new Plus(new UnaryMinus(left->GrabL()), new UnaryMinus(left->GrabR()));

    return E;
}
