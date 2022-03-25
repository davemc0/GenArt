#pragma once

// Doesn't include any of the class-based system.
// Put this in a .h file so that it can be included either by CUDA (Render_kernel.cu) or C++ (ExprTools.cpp).

#include "ExprImplementations.h"

const int MAX_STACK = 17;

#define STK(St) ValStack[St]

const int XI = 0, YI = 1, RI = 2, IFSX = 3, IFSY = 4, REDI = 5, GRNI = 6;

// TokenStream is made of ExprOpcodes_t with inline floats and variable indices.
// TokenStream contains the length and three concatenated streams for R, G, and B. At end of evaluation, these will be on the stack.
DMC_LOC void EvaluateTokenized(const int* DMC_RESTRICT TokenStream, float x, float y, float r, int tid,
    float &rout, float &gout, float &bout)
{
    float ValStack[MAX_STACK];

    STK(XI) = x;
    STK(YI) = y;
    STK(RI) = r;

    float StTop; // Will get stored in STK(IFSY).
    int IFSIter = -1; // < 0 means not inside IFS.  Takes a register, but putting it on the stack is a 10% slowdown.
    int St = IFSX; // Points to the highest used element. Use pre-increment and post-decrement.
    int Tk = 1; // Skip element 0, which is the length
#ifndef __CUDACC__
    static int StHigh = 0;
#endif

    while (Tk < TokenStream[0]) { // TokenStream[0] has the length.
        int tok = TokenStream[Tk++] & 0xff;

        float StM1 = STK(St);
        switch (tok) {
        case Const_e: St++; STK(St) = StTop; StTop = *(const float *)&(TokenStream[Tk++]); break;
        case Var_e: St++; STK(St) = StTop; StTop = STK(TokenStream[Tk++]); break;

        case Abs_e: StTop = eAbs(StTop); break;
        case ACos_e: StTop = eACos(StTop); break;
        case ASin_e: StTop = eASin(StTop); break;
        case ATan_e: StTop = eATan(StTop); break;
        case BitNot_e: StTop = eBitNot(StTop); break;
        case Cbrt_e: StTop = eCbrt(StTop); break;
        case Clamp_e: StTop = eClamp(StTop); break;
        case Cos_e: StTop = eCos(StTop); break;
        case Cube_e: StTop = eCube(StTop); break;
        case Exp_e: StTop = eExp(StTop); break;
        case Ln_e: StTop = eLn(StTop); break;
        case Round_e: StTop = eRound(StTop); break;
        case Sin_e: StTop = eSin(StTop); break;
        case Sqr_e: StTop = eSqr(StTop); break;
        case Sqrt_e: StTop = eSqrt(StTop); break;
        case Tan_e: StTop = eTan(StTop); break;
        case UnaryMinus_e: StTop = eUnaryMinus(StTop); break;

        case And_e: StTop = eAnd(StM1, StTop); break;
        case ATan2_e: StTop = eATan2(StM1, StTop); break;
        case Div_e: StTop = eDiv(StM1, StTop); break;
        case Minus_e: StTop = eMinus(StM1, StTop); break;
        case Mod_e: StTop = eMod(StM1, StTop); break;
        case Mult_e: StTop = eMult(StM1, StTop); break;
        case Or_e: StTop = eOr(StM1, StTop); break;
        case Plus_e: StTop = ePlus(StM1, StTop); break;
        case Pow_e: StTop = ePow(StM1, StTop); break;
        case XOr_e: StTop = eXOr(StM1, StTop); break;
        case IFS_e:
            if (IFSIter < 0) {
                // Starting IFS
                STK(IFSX) = STK(XI);
                STK(IFSY) = STK(YI);
                IFSIter = 0;
            }

            if (IFSIter < IFS_MAX_ITER && ((StM1*StM1 + StTop*StTop) < 4.0f)) {
                // Continuing IFS
                int ChildCnt = (TokenStream[Tk - 1] >> 16); // Encodes how many children the IFS has, and therefore how far back to jump Tk.

                Tk = Tk - 1 - ChildCnt;
                IFSIter++;

                STK(XI) = StM1; // Store the evaluated operands of the IFS as the X and Y global variables. Don't recalculate R.
                STK(YI) = StTop;
                St--; StTop = STK(St); // Prepare for the stack push that happens when we start the next iteration
            }
            else {
                // Stopping IFS
                StTop = IFSIter / float(IFS_MAX_ITER);
                STK(XI) = STK(IFSX); // Restore X and Y to their original values.
                STK(YI) = STK(IFSY);
                IFSIter = -1;
            }
            break;
        }
        // For a binary function, need to pop the stack since it consumed 2 operands and only pushed 1.
        if (tok > UnaryMinus_e)
            St--;

#ifndef __CUDACC__
        if (St > StHigh) { // Debug code to find the necessary stack size
            StHigh = St;
            printf("StHigh=%d\n", StHigh);
        }
#endif
    }

    bout = StTop;
    gout = STK(GRNI);
    rout = STK(REDI);

#ifndef __CUDACC__
    //printf("St=%d StHigh=%d\n", St, StHigh);

    // Make sure we don't overflow the stack.
    if (St >= MAX_STACK) {
        rout = 0.0f;
        gout = 1.0f;
        bout = 0.0f;
    }

    if (St != 6) {
        rout = 1.0f;
        gout = 0.0f;
        bout = 0.0f;
    }
#endif
}
