#pragma once

#include "Interval.h"

#include <Util/Assert.h>

///////////////////////////////////////////////////
// UnaryExpr

interval iAbs(const interval lv);

interval iACos(const interval lv);

interval iASin(const interval lv);

interval iATan(const interval lv);

interval iBitNot(const interval lv);

interval iCbrt(const interval lv);

interval iClamp(const interval lv);

interval iCos(const interval lv);

interval iCube(const interval lv);

interval iExp(const interval lv);

interval iLn(const interval lv);

interval iRound(const interval lv);

interval iSin(const interval lv);

interval iSqr(const interval lv);

interval iSqrt(const interval lv);

interval iTan(const interval lv);

interval iUnaryMinus(const interval lv);

///////////////////////////////////////////////////
// BinaryExpr

interval iAnd(const interval lv, const interval rv);

interval iATan2(const interval lv, const interval rv);

interval iDiv(const interval lv, const interval rv);

// There is no iIFS. Since it has to recurse it needs the Expr tree.

interval iMinus(const interval lv, const interval rv);

interval iMod(const interval lv, const interval rv);

interval iMult(const interval lv, const interval rv);

interval iOr(const interval lv, const interval rv);

interval iPlus(const interval lv, const interval rv);

interval iPow(const interval lv, const interval rv);

interval iXOr(const interval lv, const interval rv);
