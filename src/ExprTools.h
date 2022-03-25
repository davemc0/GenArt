#pragma once

// Main entry point for symbolic expression package

#include "Expr.h"

#include <string>

/////////////////////////////////////////////
// Basic

// Read a prefix string Str and return it as an Expr.
// remain always comes in empty and goes out containing leftover symbols
Expr* ReadExpr(const std::string Str, std::string& remain, const VarVals_t* VV);

// Copy the VarVals_t struct into the VVals array
void SetupVVals(const VarVals_t* VV, float* VVals);

// Initialize the VarVals_t with x, y, r
void InitVVals(VarVals_t& VV);

// Convert Expr E to a token stream for evaluation in EvaluateTokens
// Returns the number of tokens added
int Tokenize(const Expr* E, int* Tok, int MaxTokens);

// Evaluate tokenized expression Tok using non-class-based representation
// The number of tokens is stored in Tok[0]
void EvaluateTokens(int* Tok, float* VVals, float& rout, float& gout, float& bout);

// For experimenting with precompiled evaluation, etc.
float EvaluateHardcoded(const VarVals_t* VV);

/////////////////////////////////////////////
// Optimization

// Sample all variables over opI domain and return the result interval.
// If stopAtMaxErr is true and interval is too great, short-circuit. Only called by Expr::sIval.
interval sampleIval(const Expr* E, const opInfo& opI, const interval& lv, const interval& rv);

// Return an optimized copy of Expr E.
// Never modifies E. Never returns NULL.
// Does symbolic, interval, and sampling optimizations.
Expr* Optimize(const Expr* E, const VarVals_t& MinVV, const VarVals_t& MaxVV, const int steps, const float maxAbsErr, const interval outSpan);

/////////////////////////////////////////////
// Evolution

// Create a single random Nonary Expr
Expr* RandNonaryNode(const VarVals_t* VV);

// Create a single random Unary Expr, with the given child
Expr* RandUnaryNode(Expr* A);

// Create a single random Binary Expr, with the given children
Expr* RandBinaryNode(Expr* A, Expr* B);

// Create a tree that is randomly generated of size siz.
Expr* RandExpr(int siz, const VarVals_t* VV);

// Create a tree that is a mutated copy of A.
// Return an optimized copy of this expression.
// Always copies. Never returns NULL.
Expr* MutateExpr(const Expr* A, const int Prob, const int siz, const float ConstPerturb, const VarVals_t* VV);

// Create a tree that is a linear blend of the two, currently just a sum.
Expr* BlendExprs(const Expr* A, const Expr* B);

// Create a tree that is a crossover of the two, with A being dominant.
Expr* CrossExprs(const Expr* A, const Expr* B);

/////////////////////////////////////////////
// Tweaking

// Replace a given variable with Const(val) so we can do better optimization
Expr* ReplaceVarWithConst(Expr* E, int VarID, float val);

// Replace variable in in-place expression
Expr* ReplaceVars(Expr* E, const VarVals_t& reVV);

// Create and return a Const expression. Just makes it so we don't have to publicize ExprSubclasses.h.
Expr* MakeConst(const float v);

// Return true if the expression is a constant.
bool IsConst(const Expr* E);

// Return a new expression that has no nested IFS in it.
// Doesn't modify E.
Expr* RemoveNestedIFS(const Expr* E);

// Create and return a copy of this expression that scales and biases it
Expr* ScaleBias(Expr* E, const float scale, const float bias);
