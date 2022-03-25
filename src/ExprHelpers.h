#pragma once

#include "Interval.h"

#include <vector>

// All the info about the variables
// This is mostly just passed on the stack but is accessed by a Var expression.
struct VarVals_t {
    static const int NUM_VARS = 3;

    std::vector<std::string> names; // XXX Should this be static?
    std::vector<float> vals;

    VarVals_t() {}
    VarVals_t(const int n);
};

struct opInfo {
    // For numerical optimization
    VarVals_t vn; // Used only for variable names and count of variables
    interval spans[VarVals_t::NUM_VARS];
    float maxAbsErr;    // All values on interval must be within this percent difference to optimize
    int steps;          // Number of samples across the interval to compare per variable
    bool stopAtMaxErr;  // True to stop sampling once the interval is larger than MaxErr
    int phase;          // Optimization phase number (0,1,2,3)
    bool swapSymmetric; // True to swap left and right children of symmetric binary operators
};

// When to do a given optimization transformation
enum opPrio {
    AA, // Major op count improvement
    AB, // Minor op count improvement
    AI, // Interval-based op count improvement
    AL, // Minor op count improvement; dual of WE; probabilistic early and for sure late, e.g. pow(x,2) to Sqr(x)
    NL, // Same op count but more canonical form; dual of NE; probabilistic early and for sure late
    NE, // Same op count but hopefully leads to subsequent improvement; dual of NL; probabilistic early and never late
    WE, // Worse op count but hopefully leads to subsequent improvement; dual of AL; probabilistic early and never late, e.g. Sqr(x) to pow(x,2)
    XE, // Much worse op count but hopefully leads to subsequent improvement, e.g. Sqr(x) to pow(x,2)
    SE, // A swap of left and right children of symmetric binary operators
    SL, // Sort left and right children of symmetric binary operators
    XX  // Disable this optimization
};

// How to do infix printing
enum prStyle_t {
    INFIX = 0,          // Infix with name, e.g. ---X
    FUNC_EVAL = 1,      // Functions are printed as e.g. eASin vs. asin
    OP_EVAL = 2,        // Operators are printed as e.g. eUnaryMinus(X) vs. -X
    OP_EVAL_IMPURE = 4, // Operators that exactly match C++ semantics are printed as operators, e.g. -X
    PREFIX = 8          // Print in prefix, rather than infix
};

#define FOG(x) optimizationGuide(opI, x)

// Determine whether to do an optimization based on opInfo and opPrio.
bool optimizationGuide(const opInfo& opI, opPrio opP);
