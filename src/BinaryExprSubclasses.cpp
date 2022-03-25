#include "BinaryExprSubclasses.h"

#include "ExprImplementations.h"
#include "ExprTools.h"
#include "IntervalImplementations.h"
#include "NonaryExprSubclasses.h"
#include "UnaryExprSubclasses.h"

#include <Math/Random.h>
#include <Util/Assert.h>

int And::token = And_e;
int ATan2::token = ATan2_e;
int Div::token = Div_e;
int IFS::token = IFS_e;
int Minus::token = Minus_e;
int Mod::token = Mod_e;
int Mult::token = Mult_e;
int Or::token = Or_e;
int Plus::token = Plus_e;
int Pow::token = Pow_e;
int XOr::token = XOr_e;

std::string And::name = "&";
std::string ATan2::name = "atan2";
std::string Div::name = "/";
std::string IFS::name = "ifs";
std::string Minus::name = "-";
std::string Mod::name = "%";
std::string Mult::name = "*";
std::string Or::name = "|";
std::string Plus::name = "+";
std::string Pow::name = "pow";
std::string XOr::name = "^";

std::string And::fname = "eAnd";
std::string ATan2::fname = "eATan2";
std::string Div::fname = "eDiv";
std::string IFS::fname = "eIFS";
std::string Minus::fname = "eMinus";
std::string Mod::fname = "eMod";
std::string Mult::fname = "eMult";
std::string Or::fname = "eOr";
std::string Plus::fname = "ePlus";
std::string Pow::fname = "ePow";
std::string XOr::fname = "eXOr";

std::string BinaryExpr::Print(int pstyle) const
{
    // For function calls
    if (pstyle & PREFIX)
        return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle) + right->Print(pstyle);
    else
        return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + "(" + left->Print(pstyle) + ", " + right->Print(pstyle) + ")";
}

int BinaryExpr::preTokenStream(int* TokenStream, const int max_len) const
{
    *(TokenStream) = getToken();
    int WrittenL = left->preTokenStream(TokenStream + 1, max_len - 1);
    int WrittenR = right->preTokenStream(TokenStream + WrittenL + 1, max_len - WrittenL - 1);
    return 1 + WrittenL + WrittenR;
}

int BinaryExpr::postTokenStream(int* TokenStream, const int max_len) const
{
    int cntL = left->postTokenStream(TokenStream, max_len - 1);
    int cntR = right->postTokenStream(TokenStream + cntL, max_len - cntL - 1);
    ASSERT_R(cntL + cntR < max_len);
    TokenStream[cntL + cntR] = getToken();
    return cntL + cntR + 1;
}

bool BinaryExpr::isequal(const Expr* E) const
{
    if (typeid(*this) != typeid(*E)) return false;

    if (left->isequal(E->left) && right->isequal(E->right)) return true;

    if (isSymmetric() && left->isequal(E->right) && right->isequal(E->left)) { return true; }

    return false;
}

bool BinaryExpr::isless(const Expr* E) const
{
    if (getToken() == E->getToken()) {
        if (left->isequal(E->left))
            return right->isless(E->right);
        else
            return left->isless(E->left);
    } else
        return getToken() < E->getToken();
}

Expr* BinaryExpr::OptHelp(const opInfo& opI)
{
    Expr* L = left->Opt(opI);
    if (L && L != left) {
        L->ivl = left->ivl;
        delete left;
        left = L;
    }
    Expr* R = right->Opt(opI);
    if (R && R != right) {
        R->ivl = right->ivl;
        delete right;
        right = R;
    }

    init(left, right);
    ivl = Ival(opI, left->ivl, right->ivl);
    // std::cerr << "Bin " << count << tostring(ivl) << Print(PREFIX) << '\n';
    ASSERT_D(!IsNaN(ivl.lower) && !IsNaN(ivl.upper));
    ASSERT_D(!ivl.empty());

    // If children are all Const then I should be Const, too.
    if (typeid(*left) == typeid(Const) && typeid(*right) == typeid(Const) && FOG(AA)) {
        float value = Eval();
        return new Const(value);
    }

    // If interval is flat over the range, return a constant with the same value as this expression.
    if (ivl.span() <= opI.maxAbsErr && FOG(AI)) {
        // std::cerr << "\nBConst: " << count << tostring(ivl) << ' ' << Print(PREFIX) << '\n';
        ivl = interval(ivl.lower); // ivl is a return value, so make it accurate.
        return new Const(ivl.lower);
    }

    if (isSymmetric()) {
        if (FOG(SE)) {
            // Any data we carry up from the children must be swapped, too.
            std::swap(left, right);
        } else if (FOG(SL)) {
            if (typeid(*left) != typeid(*this) && typeid(*right) != typeid(*this)) {
                if (right->isless(left)) std::swap(left, right);
            } else {
                if (typeid(*left) == typeid(*this)) std::swap(left, right);

                if (typeid(*left) == typeid(*this)) {
                    ASSERT_D(typeid(*left->left) != typeid(*this)); // Children are already in sorted form.
                    ASSERT_D(typeid(*right->left) != typeid(*this));

                    // Move left list to the end of the right list
                    Expr* L = left;
                    Expr* R = right;

                    while (typeid(*R->right) == typeid(*this)) R = R->right;

                    // R is now the last one of this type on the right.
                    std::swap(left, R->right);

                    ASSERT_D(typeid(*left) != typeid(*this)); // Children are already in sorted form.
                }

                ASSERT_D(typeid(*right) == typeid(*this));

                // Now do some sorting
                if (right->left->isless(left)) std::swap(left, right->left);
            }
        }
    }

    // If something changed with a child, this is still valid.
    return (L || R) ? this : NULL;
}

void BinaryExpr::init(Expr* E1, Expr* E2)
{
    left = E1;
    right = E2;
    count = left->size() + right->size() + 1;
    hasVars = left->hasVars | right->hasVars;
}

Expr* BinaryExpr::Mutate(const int prob, const int siz, const float ConstPerturb, const VarVals_t* VV)
{
    ASSERT_D(left != NULL && right != NULL);
    ASSERT_D(prob >= 1);

    // Mutate children and hook them in. Delete old children if necessary.
    // Might waste some work if this one, when modified, doesn't use its children.
    Expr* L = left->Mutate(prob, siz, ConstPerturb, VV);
    if (L && L != left) {
        delete left;
        left = L;
    }

    Expr* R = right->Mutate(prob, siz, ConstPerturb, VV);
    if (R && R != right) {
        delete right;
        right = R;
    }

    // Only a one in prob chance of mutating this node
    if (!chance(1, prob)) return this;

    int S = randn(5);
    switch (S) {
    case 0:                                      // Replace this with one of my children
        return chance(1, 2) ? GrabL() : GrabR(); // Non-returned child gets deleted by recursive call from parent.
    case 1:                                      // Attach children to a new one
        return RandBinaryNode(GrabL(), GrabR());
    case 2: // Make self a child of a new one
        if (chance(1, 2)) {
            return RandUnaryNode(Copy());
        } else {
            // Make self a child of a new one and sibling of a random tree
            return chance(1, 2) ? RandBinaryNode(Copy(), RandExpr(siz, VV)) : RandBinaryNode(RandExpr(siz, VV), Copy());
        }
    case 3: // Swap children
        if (!isSymmetric()) {
            Expr* T = left;
            left = right;
            right = T;
            return this;
        }   // For symmetric functions, fall through.
    case 4: // Replace with new expression
        return RandExpr(siz, VV);
    }

    ASSERT_R(0);
    return NULL;
}

std::string And::Print(int pstyle) const
{
    // For infix impure operators
    if (pstyle & PREFIX) return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle) + right->Print(pstyle);

    if (pstyle & (OP_EVAL | OP_EVAL_IMPURE)) {
        return getFuncName() + "(" + left->Print(pstyle) + ", " + right->Print(pstyle) + ")";
    } else {
        bool lp = left->getArity() > 1 && left->getPrecedence() >= getPrecedence();
        bool rp = right->getArity() > 1 && right->getPrecedence() >= getPrecedence();

        return (lp ? "(" : "") + left->Print(pstyle) + (lp ? ")" : "") + " " + getName() + " " + (rp ? "(" : "") + right->Print(pstyle) + (rp ? ")" : "");
    }
}

float And::Eval(const VarVals_t* VV /*= NULL*/) const
{
    return eAnd(left->Eval(VV), right->Eval(VV));
}

interval And::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iAnd(lv.empty() ? left->Ival(opI) : lv, rv.empty() ? right->Ival(opI) : rv);
}

Expr* And::Copy() const
{
    Expr* E = new And(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* And::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // The less and greaters are because And does an implicit Clamp.
    if (typeid(*left) == typeid(Const)) {
        float lv = left->Eval();
        if (lv <= 0.0f && FOG(AA)) return new Const(0.0f);    // 0&A => 0
        if (lv >= 1.0f && FOG(AA)) return new Clamp(GrabR()); // 1&A => A
    }

    if (typeid(*right) == typeid(Const)) {
        float rv = right->Eval();
        if (rv <= 0.0f && FOG(AA)) return new Const(0.0f);    // A&0 => 0
        if (rv >= 1.0f && FOG(AA)) return new Clamp(GrabL()); // A&1 => A
    }

    if (left->ivl.lower >= 1.0f && FOG(AA)) return new Clamp(GrabR());  // 1&A => A
    if (right->ivl.lower >= 1.0f && FOG(AA)) return new Clamp(GrabL()); // A&1 => A

    // A&A => A
    if (left->isequal(right) && FOG(AA)) return GrabL();

    // A&(A|B) => A
    if (typeid(*right) == typeid(Or) && left->isequal(right->left) && FOG(AA)) return GrabL();

    // ~A&A => 0
    if (typeid(*left) == typeid(BitNot) && left->left->isequal(right) && FOG(AA)) return new Const(0);

    // (B&A)&B => B&A
    // (A&B)&B => A&B
    if (typeid(*left) == typeid(And) && (left->left->isequal(right) || left->right->isequal(right)) && FOG(AA)) return GrabL();
    // B&(B&A) => B&A
    // B&(A&B) => A&B
    if (typeid(*right) == typeid(And) && (right->left->isequal(left) || right->right->isequal(left)) && FOG(AA)) return GrabR();
    // (~A)&(~B) => ~(A|B)
    if (typeid(*left) == typeid(BitNot) && typeid(*right) == typeid(BitNot) && FOG(AA)) return new BitNot(new Or(left->GrabL(), right->GrabL()));

    // (A|B)&(A|C) => A|(B&C)
    if (typeid(*left) == typeid(Or) && typeid(*right) == typeid(Or) && left->left->isequal(right->left) && FOG(AA)) return new Or(left->GrabL(), new And(left->GrabR(), right->GrabR()));

    // A&(~A|B) => A&B
    if (typeid(*right) == typeid(Or) && typeid(*(right->left)) == typeid(BitNot) && right->left->left->isequal(left) && FOG(AA)) return new And(GrabL(), right->GrabR());

    // (A&B)&C => B&(A&C)
    if (typeid(*left) == typeid(And) && FOG(NE)) return new And(left->GrabR(), new And(GrabR(), left->GrabL()));
    // A&(B&C) => (C&A)&B
    if (typeid(*right) == typeid(And) && FOG(NE)) return new And(new And(right->GrabL(), GrabL()), right->GrabR());

    // Clamp(A)&B => A&B because & clamps both its inputs.
    if (typeid(*left) == typeid(Clamp) && FOG(AB)) return new And(left->GrabL(), GrabR());
    if (typeid(*right) == typeid(Clamp) && FOG(AB)) return new And(GrabL(), right->GrabL());

    return E;
}

float ATan2::Eval(const VarVals_t* VV /*= NULL*/) const
{
    return eATan2(left->Eval(VV), right->Eval(VV));
}

interval ATan2::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iATan2(lv.empty() ? left->Ival(opI) : lv, rv.empty() ? right->Ival(opI) : rv);
}

Expr* ATan2::Copy() const
{
    Expr* E = new ATan2(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* ATan2::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // Atan2(x,x) => Atan2(1,1)
    if (left->isequal(right)) {
        if (left->ivl.lower >= 0 && right->ivl.lower >= 0 && FOG(AA)) return new Const(atan2f(1.0f, 1.0f));
        if (left->ivl.upper <= 0 && right->ivl.upper <= 0 && FOG(AA)) return new Const(atan2f(-1.0f, -1.0f));
        // Could return an expression with round that gives the right sign, maybe.
    }

    return E;
}

std::string Div::Print(int pstyle) const
{
    // For infix impure operators
    if (pstyle & PREFIX) return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle) + right->Print(pstyle);

    if (pstyle & (OP_EVAL | OP_EVAL_IMPURE)) {
        return getFuncName() + "(" + left->Print(pstyle) + ", " + right->Print(pstyle) + ")";
    } else {
        bool lp = left->getArity() > 1 && left->getPrecedence() >= getPrecedence();
        bool rp = right->getArity() > 1 && right->getPrecedence() >= getPrecedence();

        return (lp ? "(" : "") + left->Print(pstyle) + (lp ? ")" : "") + " " + getName() + " " + (rp ? "(" : "") + right->Print(pstyle) + (rp ? ")" : "");
    }
}

float Div::Eval(const VarVals_t* VV /*= NULL*/) const
{
    return eDiv(left->Eval(VV), right->Eval(VV));
}

interval Div::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iDiv(lv.empty() ? left->Ival(opI) : lv, rv.empty() ? right->Ival(opI) : rv);
}

Expr* Div::Copy() const
{
    Expr* E = new Div(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* Div::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // (A/B)/C => A/(B*C)
    if (typeid(*left) == typeid(Div) && FOG(AB)) return new Div(left->GrabL(), new Mult(left->GrabR(), GrabR()));
    // A/(B/C) => (A*C)/B
    if (typeid(*right) == typeid(Div) && FOG(AB)) return new Div(new Mult(GrabL(), right->GrabR()), right->GrabL());

    // (A*B)/A => B
    if (typeid(*left) == typeid(Mult) && left->left->isequal(right) && FOG(AA)) return left->GrabR();
    // (A*B)/B => A
    if (typeid(*left) == typeid(Mult) && left->right->isequal(right) && FOG(AA)) return left->GrabL();
    // A/(A*B) => 1/B
    if (typeid(*right) == typeid(Mult) && left->isequal(right->left) && FOG(AA)) return new Div(new Const(1.0f), right->GrabR());
    // A/(B*A) => 1/B
    if (typeid(*right) == typeid(Mult) && left->isequal(right->right) && FOG(AA)) return new Div(new Const(1.0f), right->GrabL());

    // 0/A => 0
    if (typeid(*left) == typeid(Const)) {
        float lv = left->Eval();
        if (lv == 0.0f && FOG(AA)) return new Const(0.0f);
    }

    if (typeid(*right) == typeid(Const)) {
        float rv = right->Eval();
        if (rv == 0.0f && FOG(AA)) return new Const(0.0f); // A/0 => 0  Divide by zero error!
        if (rv == 1.0f && FOG(AB)) return GrabL();         // A/1 => A

        if (FOG(AB)) return new Mult(GrabL(), new Const(1.0f / rv)); // A/k1 => A*(1/k1)
    }

    // A/A => 1
    if (left->isequal(right) && FOG(AA)) return new Const(1.0f);

    // (A^C)/(B^C) => (A/B)^C
    if (typeid(*left) == typeid(Pow) && typeid(*right) == typeid(Pow) && left->right->isequal(right->right) && FOG(AA)) return new Pow(new Div(left->GrabL(), right->GrabL()), left->GrabR());

    // (-A)/B => -(A/B)
    if (typeid(*left) == typeid(UnaryMinus) && FOG(NL)) return new UnaryMinus(new Div(left->GrabL(), GrabR()));
    // A/(-B) => -(A/B)
    if (typeid(*right) == typeid(UnaryMinus) && FOG(NL)) return new UnaryMinus(new Div(GrabL(), right->GrabL()));

    // (C*A)/B => (A/B)*C
    if (typeid(*left) == typeid(Mult) && FOG(NE)) return new Mult(new Div(left->GrabR(), GrabR()), left->GrabL());

    if ((typeid(*left) == typeid(Mult)) && (typeid(*right) == typeid(Mult))) {
        if ((typeid(*left->left) == typeid(Const) && typeid(*right->left) == typeid(Const)) && FOG(AB)) // (k1*A)/(k2*B) => (k1/k2)*(A/B)
            return new Mult(new Div(left->GrabL(), right->GrabL()), new Div(left->GrabR(), right->GrabR()));
        if (FOG(WE)) // (A*B)/(C*D) => (A/C)*(B/D)
            return new Mult(new Div(left->GrabL(), right->GrabL()), new Div(left->GrabR(), right->GrabR()));
    }

    // A/(B^C) => A*(B^-C)
    if (typeid(*right) == typeid(Pow)) {
        if (typeid(*(right->right)) == typeid(Const)) {
            if (FOG(XX /*XE*/)) {
                // std::cerr << "DIVPOWCONST " << Print(PREFIX) << std::endl;
                // Bad because a lot of the constant exponents will turn back into Sqr, etc., without this.
                return new Mult(GrabL(), new Pow(right->GrabL(), new UnaryMinus(right->GrabR())));
            }
        } else if (FOG(WE)) {
            return new Mult(GrabL(), new Pow(right->GrabL(), new UnaryMinus(right->GrabR())));
        }
    }

    // A/B => A*(B^-1)
    if (FOG(XX /*XE*/)) {
        // std::cerr << "INVERT " << Print(PREFIX) << std::endl;
        return new Mult(GrabL(), new Pow(GrabR(), new Const(-1.0f))); // Puts a lot of Pows in the expressions that don't get eliminated.
    }

    return E;
}

float IFS::Eval(const VarVals_t* VV /*= NULL*/) const
{
    static VarVals_t VVDummy(2); // XXX Does this static provide enough optimization to be worth it?
    if (VV == NULL) VV = &VVDummy;
    return IFSVal((VarVals_t*)VV);
}

interval IFS::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    // Can't use the incoming intervals because each iteration changes the variables on which they are computed.
    opInfo opU = opI;

    // Iterate the function system until the entire output interval is outside the radius
    interval x, y;
    int itlo = -1, ithi = -1;
    interval distsqr;
    do {
        opU.spans[0] = left->Ival(opU);
        opU.spans[1] = right->Ival(opU);

        distsqr = opU.spans[0] * opU.spans[0] + opU.spans[1] * opU.spans[1];
        if (itlo == -1 && distsqr.upper >= 4.0f) itlo = ithi + 1;
    } while (++ithi < IFS_MAX_ITER && distsqr.lower < 4.0f);

    if (itlo == -1) itlo = ithi;
    return interval(itlo / float(IFS_MAX_ITER), ithi / float(IFS_MAX_ITER));
}

Expr* IFS::Copy() const
{
    Expr* E = new IFS(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* IFS::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    return E;
}

int IFS::postTokenStream(int* TokenStream, const int max_len) const
{
    int cntL = left->postTokenStream(TokenStream, max_len - 1);
    int cntR = right->postTokenStream(TokenStream + cntL, max_len - cntL - 1);
    ASSERT_R(cntL + cntR < max_len);
    // Tell how big the left and right children are so we can find them in stream
    TokenStream[cntL + cntR] = int(token) | ((cntL + cntR) << 16);
    return cntL + cntR + 1;
}

float IFS::IFSVal(VarVals_t* VV) const
{
    float xsave = VV->vals[0], ysave = VV->vals[1];
    float x, y;
    int it = -1;
    do {
        x = left->Eval(VV);
        y = right->Eval(VV);

        VV->vals[0] = x;
        VV->vals[1] = y;
    } while (++it < IFS_MAX_ITER && x * x + y * y < 4.0f);

    // Restore these to their correct value.
    VV->vals[0] = xsave;
    VV->vals[1] = ysave;

    return it / float(IFS_MAX_ITER);
}

std::string Minus::Print(int pstyle) const
{
    // For infix pure operators
    if (pstyle & PREFIX) return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle) + right->Print(pstyle);

    if (pstyle & OP_EVAL) {
        return BinaryExpr::Print(pstyle);
    } else {
        bool lp = left->getArity() > 1 && left->getPrecedence() >= getPrecedence();
        bool rp = right->getArity() > 1 && right->getPrecedence() >= getPrecedence();

        return (lp ? "(" : "") + left->Print(pstyle) + (lp ? ")" : "") + " " + getName() + " " + (rp ? "(" : "") + right->Print(pstyle) + (rp ? ")" : "");
    }
}

float Minus::Eval(const VarVals_t* VV /*= NULL*/) const
{
    return eMinus(left->Eval(VV), right->Eval(VV));
}

interval Minus::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iMinus(lv.empty() ? left->Ival(opI) : lv, rv.empty() ? right->Ival(opI) : rv);
}

Expr* Minus::Copy() const
{
    Expr* E = new Minus(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* Minus::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // A-(A+B) => -B
    if (typeid(*right) == typeid(Plus) && left->isequal(right->left) && FOG(AA)) return new UnaryMinus(right->GrabR());

    // A-(B+A) => -B
    if (typeid(*right) == typeid(Plus) && left->isequal(right->right) && FOG(AA)) return new UnaryMinus(right->GrabL());

    // (A+B)-A => B
    if (typeid(*left) == typeid(Plus) && left->left->isequal(right) && FOG(AA)) return left->GrabR();

    // (A+B)-B => A
    if (typeid(*left) == typeid(Plus) && left->right->isequal(right) && FOG(AA)) return left->GrabL();

    // (A-B)-C => A-(B+C)
    if (typeid(*left) == typeid(Minus) && FOG(AB)) return new Minus(left->GrabL(), new Plus(left->GrabR(), GrabR()));

    // A-(-B) = A+B
    if (typeid(*right) == typeid(UnaryMinus) && FOG(AB)) return new Plus(GrabL(), right->GrabL());

    // A*B-A*C => A*(B-C)
    if (typeid(*left) == typeid(Mult) && typeid(*right) == typeid(Mult) && left->left->isequal(right->left) && FOG(AA)) return new Mult(left->GrabL(), new Minus(left->GrabR(), right->GrabR()));

    // (ln A) - (ln B) => ln(A/B)
    if (typeid(*left) == typeid(Ln) && typeid(*right) == typeid(Ln) && FOG(AA)) { return new Ln(new Div(left->GrabL(), right->GrabL())); }

    // 0-A => -A
    if (typeid(*left) == typeid(Const)) {
        float lv = left->Eval();
        if (lv == 0.0f && FOG(AB)) return new UnaryMinus(GrabR());
    }

    // A-0 => A
    if (typeid(*right) == typeid(Const)) {
        float rv = right->Eval();
        if (rv == 0.0f && FOG(AB)) return GrabL();
    }

    // A-A => 0
    if (left->isequal(right) && FOG(AA)) return new Const(0.0f);

    // (-A)-B = -(A+B)
    if (typeid(*left) == typeid(UnaryMinus) && FOG(NL)) return new UnaryMinus(new Plus(left->GrabL(), GrabR()));

    // k1-(A*k2) => ((k1/k2)-A)*k2
    if (typeid(*left) == typeid(Const) && typeid(*right) == typeid(Mult) && typeid(*right->right) == typeid(Const) && FOG(NL)) {
        float v = left->Eval() / right->right->Eval();
        return new Mult(new Minus(new Const(v), right->GrabL()), right->GrabR());
    }

    // (A-B) => A+(-B)
    if (FOG(WE)) return new Plus(GrabL(), new UnaryMinus(GrabR()));

    return E;
}

std::string Mod::Print(int pstyle) const
{
    // For infix impure operators
    if (pstyle & PREFIX) return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle) + right->Print(pstyle);

    if (pstyle & (OP_EVAL | OP_EVAL_IMPURE)) {
        return getFuncName() + "(" + left->Print(pstyle) + ", " + right->Print(pstyle) + ")";
    } else {
        bool lp = left->getArity() > 1 && left->getPrecedence() >= getPrecedence();
        bool rp = right->getArity() > 1 && right->getPrecedence() >= getPrecedence();

        return (lp ? "(" : "") + left->Print(pstyle) + (lp ? ")" : "") + " " + getName() + " " + (rp ? "(" : "") + right->Print(pstyle) + (rp ? ")" : "");
    }
}

float Mod::Eval(const VarVals_t* VV /*= NULL*/) const
{
    return eMod(left->Eval(VV), right->Eval(VV));
}

interval Mod::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iMod(lv.empty() ? left->Ival(opI) : lv, rv.empty() ? right->Ival(opI) : rv);
}

Expr* Mod::Copy() const
{
    Expr* E = new Mod(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* Mod::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(Const)) {
        float lv = left->Eval();
        if (lv == 0.0f && FOG(AA)) return new Const(0.0f);
    }

    if (typeid(*right) == typeid(Const)) {
        float rv = right->Eval();
        if (rv == 0.0f && FOG(AA)) return new Const(0.0f); // A%0 => 0  Divide by zero error!
    }

    // A % -B => A%B
    if (typeid(*right) == typeid(Abs) && FOG(AB)) return new Mod(GrabL(), right->GrabL());
    if (typeid(*right) == typeid(UnaryMinus) && FOG(AB)) return new Mod(GrabL(), right->GrabL());

    if (left->isequal(right) && FOG(AA)) return new Const(0.0f);

    // A % B => A if A < B
    interval lva(fabs(left->ivl.lower), fabs(left->ivl.upper));
    interval rva(fabs(right->ivl.lower), fabs(right->ivl.upper));
    if (lva.upper < rva.lower && !right->ivl.contains(0) && FOG(AI)) return GrabL();

    return E;
}

std::string Mult::Print(int pstyle) const
{
    // For infix pure operators
    if (pstyle & PREFIX) return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle) + right->Print(pstyle);

    if (pstyle & OP_EVAL) {
        return BinaryExpr::Print(pstyle);
    } else {
        bool lp = left->getArity() > 1 && left->getPrecedence() >= getPrecedence();
        bool rp = right->getArity() > 1 && right->getPrecedence() >= getPrecedence();

        return (lp ? "(" : "") + left->Print(pstyle) + (lp ? ")" : "") + " " + getName() + " " + (rp ? "(" : "") + right->Print(pstyle) + (rp ? ")" : "");
    }
}

float Mult::Eval(const VarVals_t* VV /*= NULL*/) const
{
    return eMult(left->Eval(VV), right->Eval(VV));
}

interval Mult::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iMult(lv.empty() ? left->Ival(opI) : lv, rv.empty() ? right->Ival(opI) : rv);
}

Expr* Mult::Copy() const
{
    Expr* E = new Mult(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* Mult::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(Const)) {
        float lv = left->Eval();
        if (lv == 0.0f && FOG(AA)) return new Const(0.0f);
        if (lv == 1.0f && FOG(AB)) return GrabR();
        if (lv == -1.0f && FOG(AB)) return new UnaryMinus(GrabR());
    }

    if (typeid(*right) == typeid(Const)) {
        float rv = right->Eval();
        if (rv == 0.0f && FOG(AA)) return new Const(0.0f);
        if (rv == 1.0f && FOG(AB)) return GrabL();
        if (rv == -1.0f && FOG(AB)) return new UnaryMinus(GrabL());
    }

    // (A/B)*B => A
    if (typeid(*left) == typeid(Div) && left->right->isequal(right) && FOG(AA)) return left->GrabL();

    // A*(B/A) => B
    if (typeid(*right) == typeid(Div) && left->isequal(right->right) && FOG(AA)) return right->GrabL();

    // (A^B)*(A^C) = A^(B+C)
    if (typeid(*left) == typeid(Pow) && typeid(*right) == typeid(Pow) && left->left->isequal(right->left) && FOG(AA)) return new Pow(left->GrabL(), new Plus(left->GrabR(), right->GrabR()));

    // (A^B)*A = A^(B+1)
    if (typeid(*left) == typeid(Pow) && left->left->isequal(right) && FOG(AA)) return new Pow(left->GrabL(), new Plus(left->GrabR(), new Const(1))); // Does Sqr(A)*A => Cube(A)

    // (A^C)*(B^C) => (A*B)^C
    if (typeid(*left) == typeid(Pow) && typeid(*right) == typeid(Pow) && left->right->isequal(right->right) && FOG(AA)) return new Pow(new Mult(left->GrabL(), right->GrabL()), left->GrabR());

    // A*A => Sqr(A)
    if (left->isequal(right) && FOG(AA)) return new Sqr(GrabL());

    // Sqr(A)*Sqr(B) => Sqr(A*B)
    if (typeid(*left) == typeid(Sqr) && typeid(*right) == typeid(Sqr) && FOG(AL)) return new Sqr(new Mult(left->GrabL(), right->GrabL()));

    // (A/B)*A => Sqr(A)/B
    if (typeid(*left) == typeid(Div) && left->left->isequal(right) && FOG(AA)) return new Div(new Sqr(GrabR()), left->GrabR());

    // A*(A/B) => Sqr(A)/B
    if (typeid(*right) == typeid(Div) && left->isequal(right->left) && FOG(AA)) return new Div(new Sqr(GrabL()), right->GrabR());

    // (A-k1)*k2 => (A*k2)-(k1*k2)
    if (typeid(*left) == typeid(Minus) && typeid(*right) == typeid(Const) && typeid(*left->right) == typeid(Const) && FOG(NE)) {
        float v = left->right->Eval() * right->Eval();
        return new Minus(new Mult(left->GrabL(), GrabR()), new Const(v));
    }

    // (k1-A)*k2 => (k1*k2)-(A*k2)
    if (typeid(*left) == typeid(Minus) && typeid(*right) == typeid(Const) && typeid(*left->left) == typeid(Const) && FOG(NE)) {
        float v = left->left->Eval() * right->Eval();
        return new Minus(new Const(v), new Mult(left->GrabR(), GrabR()));
    }

    // (A+k1)*k2 => (A*k2)+(k1*k2)
    if (typeid(*left) == typeid(Plus) && typeid(*right) == typeid(Const) && typeid(*left->right) == typeid(Const) && FOG(NE)) {
        float v = left->right->Eval() * right->Eval();
        return new Plus(new Mult(left->GrabL(), GrabR()), new Const(v));
    }

    // (A*B)*C => B*(A*C)
    if (typeid(*left) == typeid(Mult) && FOG(NE)) return new Mult(left->GrabR(), new Mult(GrabR(), left->GrabL()));
    // A*(B*C) => (C*A)*B
    if (typeid(*right) == typeid(Mult) && FOG(NE)) return new Mult(new Mult(right->GrabL(), GrabL()), right->GrabR());

    // (A/C)*(B/D) => (A*B)/(C*D)
    if (typeid(*left) == typeid(Div) && typeid(*right) == typeid(Div) && FOG(AL)) return new Div(new Mult(left->GrabL(), right->GrabL()), new Mult(left->GrabR(), right->GrabR()));

    // (A/B)*C => (C*A)/B
    if (typeid(*left) == typeid(Div) && FOG(NL)) return new Div(new Mult(GrabR(), left->GrabL()), left->GrabR());
    // A*(B/C) => (B*A)/C
    if (typeid(*right) == typeid(Div) && FOG(NL)) return new Div(new Mult(right->GrabL(), GrabL()), right->GrabR());

    // -A*B => -(A*B)
    if (typeid(*left) == typeid(UnaryMinus) && FOG(NL)) return new UnaryMinus(new Mult(GrabR(), left->GrabL()));
    // A*-B => -(A*B)
    if (typeid(*right) == typeid(UnaryMinus) && FOG(NL)) return new UnaryMinus(new Mult(right->GrabL(), GrabL()));

    // A*(B^-C) => A/(B^C)
    if (typeid(*right) == typeid(Pow) && typeid(*right->right) == typeid(UnaryMinus) && FOG(AL)) return new Div(GrabL(), new Pow(right->GrabL(), right->right->GrabL()));

    // B*Ln(A) => Ln(A^B)
    // Messes some up; not sure why yet.
    if (typeid(*right) == typeid(Ln) && FOG(XX /*NE*/)) return new Ln(new Pow(right->GrabL(), GrabL()));

    return E;
}

std::string Or::Print(int pstyle) const
{
    // For infix impure operators
    if (pstyle & PREFIX) return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle) + right->Print(pstyle);

    if (pstyle & (OP_EVAL | OP_EVAL_IMPURE)) {
        return getFuncName() + "(" + left->Print(pstyle) + ", " + right->Print(pstyle) + ")";
    } else {
        bool lp = left->getArity() > 1 && left->getPrecedence() >= getPrecedence();
        bool rp = right->getArity() > 1 && right->getPrecedence() >= getPrecedence();

        return (lp ? "(" : "") + left->Print(pstyle) + (lp ? ")" : "") + " " + getName() + " " + (rp ? "(" : "") + right->Print(pstyle) + (rp ? ")" : "");
    }
}

float Or::Eval(const VarVals_t* VV /*= NULL*/) const
{
    return eOr(left->Eval(VV), right->Eval(VV));
}

interval Or::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iOr(lv.empty() ? left->Ival(opI) : lv, rv.empty() ? right->Ival(opI) : rv);
}

Expr* Or::Copy() const
{
    Expr* E = new Or(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* Or::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // The less and greaters are because Or does an implicit Clamp.
    if (typeid(*left) == typeid(Const)) {
        float lv = left->Eval();
        if (lv <= 0.0f && FOG(AA)) return new Clamp(GrabR());
        if (lv >= 1.0f && FOG(AA)) return new Const(1.0f);
    }

    if (typeid(*right) == typeid(Const)) {
        float rv = right->Eval();
        if (rv <= 0.0f && FOG(AA)) return new Clamp(GrabL());
        if (rv >= 1.0f && FOG(AA)) return new Const(1.0f);
    }

    if (left->ivl.upper <= 0.0f && FOG(AI)) return new Clamp(GrabR());  // 0|A => A
    if (right->ivl.upper <= 0.0f && FOG(AI)) return new Clamp(GrabL()); // A|0 => A

    // A|A = A
    if (left->isequal(right) && FOG(AA)) return GrabL();

    // A|(A&B) => A
    if (typeid(*right) == typeid(And) && left->isequal(right->left) && FOG(AA)) return GrabL();

    // ~A|A => 1
    if (typeid(*left) == typeid(BitNot) && left->left->isequal(right) && FOG(AA)) return new Const(1);

    // (B|A)|B = B|A
    // (A|B)|B = A|B
    if (typeid(*left) == typeid(Or) && (left->left->isequal(right) || left->right->isequal(right)) && FOG(AA)) return GrabL();
    // B|(B|A) = B|A
    // B|(A|B) = A|B
    if (typeid(*right) == typeid(Or) && (right->left->isequal(left) || right->right->isequal(left)) && FOG(AA)) return GrabR();
    // (A|B)|(B|A) = A|B
    if (typeid(*left) == typeid(Or) && typeid(*right) == typeid(Or) && (right->left->isequal(left->right) || right->right->isequal(left->left)) && FOG(AA)) return GrabL();
    // (~A)|(~B) = ~(A&B)
    if (typeid(*left) == typeid(BitNot) && typeid(*right) == typeid(BitNot) && FOG(AB)) return new BitNot(new And(left->GrabL(), right->GrabL()));

    // (A&B)|(A&C) => A&(B|C)
    if (typeid(*left) == typeid(And) && typeid(*right) == typeid(And) && left->left->isequal(right->left) && FOG(AB)) return new And(left->GrabL(), new Or(left->GrabR(), right->GrabR()));

    // A|(~A&B) => A|B
    if (typeid(*right) == typeid(And) && typeid(*(right->left)) == typeid(BitNot) && right->left->left->isequal(left) && FOG(AB)) return new Or(GrabL(), right->GrabR());

    // Clamp(A)|B = A|B because | clamps both its inputs.
    if (typeid(*left) == typeid(Clamp) && FOG(AB)) return new Or(left->GrabL(), GrabR());
    if (typeid(*right) == typeid(Clamp) && FOG(AB)) return new Or(GrabL(), right->GrabL());

    // (A|B)|C => B|(A|C)
    if (typeid(*left) == typeid(Or) && FOG(NE)) return new Or(left->GrabR(), new Or(GrabR(), left->GrabL()));
    // A|(B|C) => (C|A)|B
    if (typeid(*right) == typeid(Or) && FOG(NE)) return new Or(new Or(right->GrabL(), GrabL()), right->GrabR());

    return E;
}

std::string Plus::Print(int pstyle) const
{
    // For infix pure operators
    if (pstyle & PREFIX) return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle) + right->Print(pstyle);

    if (pstyle & OP_EVAL) {
        return BinaryExpr::Print(pstyle);
    } else {
        bool lp = left->getArity() > 1 && left->getPrecedence() >= getPrecedence();
        bool rp = right->getArity() > 1 && right->getPrecedence() >= getPrecedence();

        return (lp ? "(" : "") + left->Print(pstyle) + (lp ? ")" : "") + " " + getName() + " " + (rp ? "(" : "") + right->Print(pstyle) + (rp ? ")" : "");
    }
}

float Plus::Eval(const VarVals_t* VV /*= NULL*/) const
{
    return ePlus(left->Eval(VV), right->Eval(VV));
}

interval Plus::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iPlus(lv.empty() ? left->Ival(opI) : lv, rv.empty() ? right->Ival(opI) : rv);
}

Expr* Plus::Copy() const
{
    Expr* E = new Plus(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* Plus::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // 0+A => A
    if (typeid(*left) == typeid(Const) && FOG(AB)) {
        float lv = left->Eval();
        if (lv == 0.0f) return GrabR();
    }

    // A+0 => A
    if (typeid(*right) == typeid(Const) && FOG(AB)) {
        float rv = right->Eval();
        if (rv == 0.0f) return GrabL();
    }

    // (A+(B-A)) => B
    if (typeid(*right) == typeid(Minus) && left->isequal(right->right) && FOG(AA)) return right->GrabL();

    // ((A-B)+B) => A
    if (typeid(*left) == typeid(Minus) && right->isequal(left->right) && FOG(AA)) return left->GrabR();

    // (-A) + (-B) => -(A+B)
    if (typeid(*left) == typeid(UnaryMinus) && typeid(*right) == typeid(UnaryMinus) && FOG(AB)) return new UnaryMinus(new Plus(left->GrabL(), right->GrabL()));

    // (ln A) + (ln B) => ln(A*B)
    if (typeid(*left) == typeid(Ln) && typeid(*right) == typeid(Ln) && FOG(AA)) return new Ln(new Mult(left->GrabL(), right->GrabL()));

    // (A+A) => A*2
    if (left->isequal(right) && FOG(AA)) return new Mult(GrabL(), new Const(2.0f));

    // ((B-A)+B) => B*2 - A
    if (typeid(*left) == typeid(Minus) && right->isequal(left->left) && FOG(AA)) return new Minus(new Mult(left->GrabL(), new Const(2.0f)), left->GrabR());

    // (A+(A-B)) => A*2-B
    if (typeid(*right) == typeid(Minus) && left->isequal(right->left) && FOG(AA)) return new Minus(new Mult(GrabL(), new Const(2.0f)), right->GrabR());

    // ((A-B)+B) => A
    if (typeid(*left) == typeid(Minus) && right->isequal(left->right) && FOG(AA)) return left->GrabL();

    // (A+(B-A)) => B
    if (typeid(*right) == typeid(Minus) && left->isequal(right->right) && FOG(AA)) return right->GrabL();

    // A*B+A*C => A*(B+C)
    if (typeid(*left) == typeid(Mult) && typeid(*right) == typeid(Mult) && left->left->isequal(right->left) && FOG(AA)) return new Mult(left->GrabL(), new Plus(left->GrabR(), right->GrabR()));

    // (-A)+B => B-A
    if (typeid(*left) == typeid(UnaryMinus) && FOG(AL)) return new Minus(GrabR(), left->GrabL());
    // A+(-B) => A-B
    if (typeid(*right) == typeid(UnaryMinus) && FOG(AL)) return new Minus(GrabL(), right->GrabL());

    // (A+B)+C => (B+C)+A
    if (typeid(*left) == typeid(Plus) && FOG(NE)) return new Plus(left->GrabR(), new Plus(GrabR(), left->GrabL()));
    // A+(B+C) => (C+A)+B
    if (typeid(*right) == typeid(Plus) && FOG(NE)) return new Plus(new Plus(right->GrabR(), GrabL()), right->GrabL());
    // A+(B-C) => (A+B)-C
    if (typeid(*right) == typeid(Minus) && FOG(NE)) return new Minus(new Plus(GrabL(), right->GrabL()), right->GrabR());

    // k1+(A*k2) => ((k1/k2)+A)*k2
    if (typeid(*left) == typeid(Const) && typeid(*right) == typeid(Mult) && typeid(*right->right) == typeid(Const) && FOG(NL)) {
        // std::cerr << Print(PREFIX) << '\n';
        float v = left->Eval() / right->right->Eval();
        return new Mult(new Plus(new Const(v), right->GrabL()), right->GrabR());
    }

    return E;
}

float Pow::Eval(const VarVals_t* VV /*= NULL*/) const
{
    return ePow(left->Eval(VV), right->Eval(VV));
}

interval Pow::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iPow(lv.empty() ? left->Ival(opI) : lv, rv.empty() ? right->Ival(opI) : rv);
}

Expr* Pow::Copy() const
{
    Expr* E = new Pow(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* Pow::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    // (A^B)^C => A^(B*C)
    if (typeid(*left) == typeid(Pow) && FOG(AB)) return new Pow(left->GrabL(), new Mult(left->GrabR(), GrabR()));

    // Convert to one of the built-in power types.
    // NOTE: Can't raise a negative number to a fractional power.
    if (typeid(*right) == typeid(Const) && FOG(AL)) {
        float rv = right->Eval();
        if (rv == -1.0f) return new Div(new Const(1.0f), GrabL());           // A^-1 => 1/A
        if (rv == -2.0f) return new Div(new Const(1.0f), new Sqr(GrabL()));  // A^-2 => 1/Sqr(A)
        if (rv == -3.0f) return new Div(new Const(1.0f), new Cube(GrabL())); // A^-3 => 1/Cube(A)
        if (rv == 0.0f) return new Const(1.0f);                              // A^0 => 1
        if (rv == 1.0f) return GrabL();                                      // A^1 => A
        if (rv == 2.0f) return new Sqr(GrabL());
        if (rv == 3.0f) return new Cube(GrabL());
        if (rv == 0.5f) return new Sqrt(GrabL());
        if (rv == (1.0f / 3.0f)) return new Cbrt(GrabL());
    }

    if (typeid(*left) == typeid(Const)) {
        float lv = left->Eval();
        if (lv == E_E && FOG(AL)) return new Exp(GrabR()); // E^A => Exp(A)
        if (lv == 0.0f && FOG(AA)) return new Const(0.0f); // 0^A => 0
        if (lv == 1.0f && FOG(AA)) return new Const(1.0f); // 1^A => 1

        // k1^(k2+C) => k1^k2 * k1^C ==> k3 * k1^C
        // In practice it's exp(k2+C) => k3 * exp(C)
        if (typeid(*right) == typeid(Plus)) {
            if ((typeid(*(right->left)) == typeid(Const) || typeid(*(right->right)) == typeid(Const)) && FOG(XX /*NE*/)) {
                // std::cerr << "\nPOWDIST: " << Print(PREFIX) << '\n';
                return new Mult(new Pow(new Const(lv), right->GrabL()), new Pow(new Const(lv), right->GrabR()));
            }
        }
    }

    return E;
}

std::string XOr::Print(int pstyle) const
{
    // For infix impure operators
    if (pstyle & PREFIX) return ((pstyle & FUNC_EVAL) ? getFuncName() : getName()) + " " + left->Print(pstyle) + right->Print(pstyle);

    if (pstyle & (OP_EVAL | OP_EVAL_IMPURE)) {
        return getFuncName() + "(" + left->Print(pstyle) + ", " + right->Print(pstyle) + ")";
    } else {
        bool lp = left->getArity() > 1 && left->getPrecedence() >= getPrecedence();
        bool rp = right->getArity() > 1 && right->getPrecedence() >= getPrecedence();

        return (lp ? "(" : "") + left->Print(pstyle) + (lp ? ")" : "") + " " + getName() + " " + (rp ? "(" : "") + right->Print(pstyle) + (rp ? ")" : "");
    }
}

float XOr::Eval(const VarVals_t* VV /*= NULL*/) const
{
    return eXOr(left->Eval(VV), right->Eval(VV));
}

interval XOr::Ival(const opInfo& opI, const interval& lv /* = interval() */, const interval& rv /*= interval() */) const
{
    return iXOr(lv.empty() ? left->Ival(opI) : lv, rv.empty() ? right->Ival(opI) : rv);
}

Expr* XOr::Copy() const
{
    Expr* E = new XOr(*this);
    if (left) E->left = left->Copy();
    if (right) E->right = right->Copy();
    return E;
}

Expr* XOr::Opt(const opInfo& opI)
{
    Expr* E = OptHelp(opI);
    if (E && E != this) return E;

    if (typeid(*left) == typeid(Const)) {
        float lv = left->Eval();
        if (lv <= 0.0f && FOG(AB)) return new Clamp(GrabR());  // 0 xor A => A
        if (lv >= 1.0f && FOG(AB)) return new BitNot(GrabR()); // 1 xor A => ~A
    }

    if (typeid(*right) == typeid(Const)) {
        float rv = right->Eval();
        if (rv <= 0.0f && FOG(AB)) return new Clamp(GrabL());
        if (rv >= 1.0f && FOG(AB)) return new BitNot(GrabL());
    }

    // These two never trigger for some reason.
    if (left->ivl.lower >= 1.0f && FOG(AB)) return new BitNot(GrabR());  // 1 xor A => ~A
    if (right->ivl.lower >= 1.0f && FOG(AB)) return new BitNot(GrabL()); // A xor 1 => ~A

    if (left->ivl.upper <= 0.0f && FOG(AB)) return new Clamp(GrabR());  // 0 xor A => A
    if (right->ivl.upper <= 0.0f && FOG(AB)) return new Clamp(GrabL()); // A xor 0 => A

    // ~A xor ~B => A xor B
    if (typeid(*left) == typeid(BitNot) && typeid(*right) == typeid(BitNot) && FOG(AA)) return new XOr(left->GrabL(), right->GrabL());

    // Clamp(A) xor B = A xor B because xor clamps both its inputs.
    if (typeid(*left) == typeid(Clamp) && FOG(AB)) return new XOr(left->GrabL(), GrabR());
    if (typeid(*right) == typeid(Clamp) && FOG(AB)) return new XOr(GrabL(), right->GrabL());

    // A xor A => 0
    if (left->isequal(right) && FOG(AA)) return new Const(0.0f);

    // (A xor B) xor C => B xor (A xor C)
    if (typeid(*left) == typeid(XOr) && FOG(NE)) return new XOr(left->GrabR(), new XOr(GrabR(), left->GrabL()));

    // A xor (B xor C) => (C xor A) xor B
    if (typeid(*right) == typeid(XOr) && FOG(NE)) return new XOr(new XOr(right->GrabL(), GrabL()), right->GrabR());

    // ~A xor B => ~(A xor B)
    if (typeid(*left) == typeid(BitNot) && FOG(NL)) return new BitNot(new XOr(left->GrabL(), GrabR()));

    return E;
}
