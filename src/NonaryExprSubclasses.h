#pragma once

// The subclasses for nonary expressions
// Only include this in internal .cpp files

#include "Expr.h"

class NonaryExpr : public Expr
{
public:
    inline bool isSymmetric() const { return true; }
    inline int getPrecedence() const { return 9; }
    inline int getArity() const { return 0; }
};

class Const : public NonaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }

    float Eval(const VarVals_t* VV = NULL) const;
    interval Ival(const opInfo& opI, const interval& lv = interval(), const interval& rv = interval()) const;

    Expr* Copy() const;

    std::string Print(int pstyle) const;

    int preTokenStream(int* TokenStream, const int max_len) const;
    int postTokenStream(int* TokenStream, const int max_len) const;

    Expr* Opt(const opInfo& opI);

    // If probable, modify the random number
    Expr* Mutate(const int Prob, const int RandTreeSize, const float ConstPerturbStDev, const VarVals_t* VV);

    // Set all constants in this Expr to randomly perturbed values.
    Expr* PerturbConstants(const float ConstPerturbStDev);

    bool isequal(const Expr* E) const;
    bool isless(const Expr* E) const;

    Const();
    Const(float v);

    void UniformRandom(); // Fill this constant with a uniform random variable on 0..1

    unsigned int HasVars() const { return 0u; }

private:
    float val;
};

class Var : public NonaryExpr
{
public:
    static std::string name; // The static name of the function: "Var"
    static std::string fname;
    static int token;

    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }

    float Eval(const VarVals_t* VV = NULL) const;
    interval Ival(const opInfo& opI, const interval& lv = interval(), const interval& rv = interval()) const;

    Expr* Copy() const;

    std::string Print(int pstyle) const;

    int preTokenStream(int* TokenStream, const int max_len) const;
    int postTokenStream(int* TokenStream, const int max_len) const;

    Expr* Opt(const opInfo& opI);

    // If probable, replace variable with a different one
    Expr* Mutate(const int prob, const int siz, const float ConstPerturb, const VarVals_t* VV);

    bool isequal(const Expr* E) const;
    bool isless(const Expr* E) const;

    Var();
    Var(const std::string VarName_, int VarID_);

    size_t getVarID() const;
    std::string GetVarName() const;
    unsigned int HasVars() const; // Bitmask of variables that occur in this expression

private:
    std::string VarName; // Since we don't have VarVals access, keep a local copy of the name
    size_t VarID;
};
