// Base class for symbolic expression package

// +, -, *, /, %, ^ (pow), unary -
// |, &, ~, xor  (These convert the float to 16-bit fixed point and then back.)
// sin, cos, tan, asin, acos, atan, atan2
// sqrt, cbrt, sqr, cube, ln, exp
// abs, round (-inf->0.49 => 0, 0.5->inf => 1), clamp, ifs
// variable, const

#pragma once

#include "ExprHelpers.h"
#include "Interval.h"

#include <string>

class Expr
{
public:
    Expr*        left;
    Expr*        right;
    interval     ivl; // NOTE: The interval is a function of the current ranges of the variables
    int          count;
    unsigned int hasVars;

    //static std::string name; // Name of the token stored in the file
    //static std::string fname; // Name of the function that evaluates it
    //static int token; // Numeric symbol of the token

    virtual bool isSymmetric() const = 0; // True if this is a symmetric operator
    virtual int getPrecedence() const = 0; // Use numbers from http://en.cppreference.com/w/cpp/language/operator_precedence
    virtual int getArity() const = 0;
    virtual int getToken() const = 0;
    virtual std::string getName() const = 0;
    virtual std::string getFuncName() const = 0;

    // Evaluate this Expr.
    virtual float Eval(const VarVals_t *VV = NULL) const = 0;

    // Evaluate this Expr using interval arithmetic. If intervals are passed in, uses them. Otherwise, calls Ival on children.
    virtual interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const = 0;

    // Evaluate this Expr by sampling it to compute its result interval.
    interval sIval(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;

    // Return a copy of this expression.
    virtual Expr* Copy() const = 0;

    // Return a string representing this Expr
    virtual std::string Print(int pstyle) const = 0;

    // Fill in a prefix tokenized stream made of ExprOpcodes_t with inline floats and variable indices representing this Expr
    // Returns the number of words written
    virtual int preTokenStream(int *TokenStream, const int max_len) const = 0;

    // Fill in a postfix tokenized stream made of ExprOpcodes_t with inline floats and variable indices representing this Expr
    // Returns the number of words written
    virtual int postTokenStream(int *TokenStream, const int max_len) const = 0;

    // Optimize this expression. May detach its children and corrupt this one in the process.
    // External code never calls this. Always call Optimize() instead.
    // Meanings of Expr* return value:
    // this: Something changed below this, but this is still valid.
    // NULL: Nothing has changed either with this or below.
    // else: A replacement for this. Caller needs to delete this old one and attach the new one.
    virtual Expr* Opt(const opInfo &opI) = 0;

    // Return an expression that's a mutation of this one.
    // May detach its children and destroy this one in the process.
    // Caller responsible for deleting this child if something different is returned.
    // Never returns NULL.
    // This function gets overloaded for leaf nodes, so don't worry about them here.
    // External code never calls this. Always call MutateExpr() instead.
    virtual Expr* Mutate(const int prob, const int siz, const float ConstPerturb, const VarVals_t *VV) = 0;

    // Set all constants in this Expr to randomly perturbed values.
    virtual Expr* PerturbConstants(const float rc);

    // True if the given Expr symbolically equals this.
    virtual bool isequal(const Expr* E) const = 0;
    // True if the given Expr is symbolically less than this.
    virtual bool isless(const Expr* E) const = 0;

protected:
    Expr();

public:
    ~Expr();

    // Detach a child and return it.
    Expr* GrabL();

    // Detach a child and return it.
    Expr* GrabR();

    // Bitmask of variables that occur in this expression
    unsigned int HasVars() const;

    // Return the number of nodes in this expression tree
    int size() const;

    // Return a pointer to a uniformly distributed random node in this expression
    // Return either the left child's choice, the right child's choice, or the left or right child itself
    // count is return count of nodes, including root
    Expr* const *FindRand(int& count) const;
};
