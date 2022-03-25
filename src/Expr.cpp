#include "ExprTools.h"
#include "NonaryExprSubclasses.h"
#include "MathHelpers.h"

#include <Math/Random.h>
#include <Util/Assert.h>

#include <string>
#include <iostream>
#include <algorithm>

interval Expr::sIval(const opInfo &opI, const interval &lv /* = interval() */, const interval &rv /* = interval() */) const
{
    return sampleIval(this, opI, lv, rv);
}

Expr* Expr::PerturbConstants(const float rc)
{
    if (left)
        left->PerturbConstants(rc);
    if (right)
        right->PerturbConstants(rc);

    return this;
}

Expr::Expr()
{
    left = right = NULL;
    count = 0;
    hasVars = 0;
    // ivl is still empty
}

Expr::~Expr()
{
    if (left) delete left;
    left = NULL;
    if (right) delete right;
    right = NULL;
}

Expr* Expr::GrabL()
{
    Expr* Tmp = left;
    left = NULL;
    count = 0; // Invalidate my records
    return Tmp;
}

Expr* Expr::GrabR()
{
    Expr* Tmp = right;
    right = NULL;
    count = 0; // Invalidate my records
    return Tmp;
}

unsigned int Expr::HasVars() const
{
    ASSERT_R(count > 0);

    return hasVars;
}

int Expr::size() const
{
    ASSERT_R(count > 0);

    return count;
}

Expr* const * Expr::FindRand(int& count) const
{
    int lc = 0, rc = 0;

    Expr* const * lp = left ? left->FindRand(lc) : NULL;
    Expr* const * rp = right ? right->FindRand(rc) : NULL;

    count = lc + rc + 1;

    if (chance(lc, lc + rc))
        if (chance(1,lc))
            return &left;
        else
            return lp;
    else
        if (chance(1,rc))
            return &right;
        else
            return rp;
}
