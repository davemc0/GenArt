#include "NonaryExprSubclasses.h"
#include "UnaryExprSubclasses.h"
#include "BinaryExprSubclasses.h"

#include "ExprImplementations.h"

#include "Math/Random.h"
#include "Util/Assert.h"

#include <sstream>
#include <string>

int Const::token = Const_e;
int Var::token = Var_e;

std::string Const::name = "const";
std::string Var::name = "var";

std::string Const::fname = "CONSTISBROKEN";
std::string Var::fname = "VARISBROKEN";

union UFloatInt {
    int i;
    float f;
};

float Const::Eval(const VarVals_t *VV) const
{
    return val;
}

interval Const::Ival(const opInfo &opI, const interval &lv /* = interval() */, const interval &rv /*= interval() */) const
{
    return interval(val);
}

Expr* Const::Copy() const
{
    return new Const(*this);
}

std::string Const::Print(int pstyle) const
{
    std::ostringstream st; st << val;
    return st.str() + ((pstyle & PREFIX) ? " " : "");
}

int Const::preTokenStream(int *TokenStream, const int max_len) const
{
    *(TokenStream++) = token;
    UFloatInt FI;
    FI.f = val;
    *(TokenStream++) = FI.i;
    return 2;
}

int Const::postTokenStream(int *TokenStream, const int max_len) const
{
	ASSERT_R(max_len >= 2);
	*(TokenStream++) = token;
    UFloatInt FI;
    FI.f = val;
    *(TokenStream++) = FI.i;
    return 2;
}

inline Expr* Const::Opt(const opInfo &opI)
{
    ivl = Ival(opI);

    return NULL;
}

Expr* Const::Mutate(const int Prob, const int RandTreeSize, const float ConstPerturb, const VarVals_t *VV)
{
    if (chance(1,Prob)) {
        PerturbConstants(ConstPerturb);
    }
    return NULL;
}

Expr* Const::PerturbConstants(const float rc)
{
    val += NRandf(rc);
    return this;
}

bool Const::isequal(const Expr* E) const
{
    return typeid(*E) == typeid(Const) && Eval() == ((const Const *)E)->Eval();
}

bool Const::isless(const Expr* E) const
{
    if (getToken() == E->getToken())
        return Eval() < ((const Const *)E)->Eval();
    else
        return getToken() < E->getToken();
}

void Const::UniformRandom()
{
    val = DRandf();
}

Const::Const()
{
    val = 0.0f;
    count = 1;
    hasVars = 0u;
}

Const::Const(float v)
{
    val = v;
    count = 1;
    hasVars = 0u;
}

float Var::Eval(const VarVals_t *VV /*= NULL*/) const
{
    ASSERT_D(VV);
    return VV->vals[VarID];
}

interval Var::Ival(const opInfo &opI, const interval &lv /* = interval() */, const interval &rv /*= interval() */) const
{
    return interval(opI.spans[VarID]);
}

Expr* Var::Copy() const
{
    return new Var(*this);
}

std::string Var::Print(int pstyle) const
{
    return VarName + ((pstyle & PREFIX) ? " " : "");
}

int Var::preTokenStream(int *TokenStream, const int max_len) const
{
    *(TokenStream++) = token;
    *(TokenStream++) = (int)VarID;
    return 2;
}

int Var::postTokenStream(int *TokenStream, const int max_len) const
{
	ASSERT_R(max_len >= 2);
	*(TokenStream++) = token;
    *(TokenStream++) = (int)VarID;
    return 2;
}

Expr* Var::Opt(const opInfo &opI)
{
    ivl = Ival(opI);

    return NULL;
}

Expr* Var::Mutate(const int prob, const int siz, const float ConstPerturb, const VarVals_t *VV)
{
    ASSERT_D(VV);
    if (chance(1,prob)) {
        // Replace the variable with a different random variable
        VarID = randn((int)VV->vals.size());
        VarName = VV->names[VarID];
    }
    return NULL;
}

bool Var::isequal(const Expr* E) const
{
    return typeid(*E) == typeid(Var) && getVarID() == ((const Var *)E)->getVarID();
}

bool Var::isless(const Expr* E) const
{
    if (getToken() == E->getToken())
        return getVarID() < ((const Var *)E)->getVarID();
    else
        return getToken() < E->getToken();
}

Var::Var()
{
    count = 0;
}

Var::Var(const std::string VarName_, int VarID_)
{
    VarID = VarID_;
    VarName = VarName_;
    hasVars = (1u << VarID);
    count = 1;
}

std::string Var::GetVarName() const
{
    return VarName;
}

size_t Var::getVarID() const
{
    return VarID;
}
