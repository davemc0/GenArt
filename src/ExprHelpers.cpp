#include "ExprHelpers.h"

#include <Math/Random.h>
#include <Util/Assert.h>

VarVals_t::VarVals_t(const int n)
{
    names.resize(n);
    vals.resize(n);
}

bool optimizationGuide(const opInfo& opI, opPrio opP)
{ // Phase 0 == A only; 1 == early; 2 == late opt; 3 == late cleanup
    switch(opP) {
    case AA: return true; // 0123
    case AB: return true; // 0123
    case AI: return true; // 0123
    case AL: return opI.phase >= 2 || (opI.phase == 1 && chance(1,4)); // 1p23
    case NL: return opI.phase == 3 || (opI.phase == 2 && chance(1,2)); // 2p3
    case NE: return (opI.phase == 1 || opI.phase == 2) && chance(1,2); // 1p2p
    case WE: return (opI.phase == 1 || opI.phase == 2) && chance(1,3); // 1p2p
    case XE: return opI.phase == 1 && chance(1,3); // 1p
    case SE: return (opI.phase == 1 || opI.phase == 2) && chance(1,2); // 1p2p
    case SL: return opI.phase == 3; // 3
    case XX: return false;
    }

    ASSERT_R(0);
    return false;
}
