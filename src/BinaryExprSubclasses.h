#pragma once

#include "Expr.h"

class BinaryExpr : public Expr
{
public:
    inline bool isSymmetric() const { return false; }
    inline int getPrecedence() const { return 2; }
    inline int getArity() const { return 2; }

    std::string Print(int pstyle) const; // Plus, Minus, Mult are pure operators.

    int preTokenStream(int *TokenStream, const int max_len) const;
    int postTokenStream(int *TokenStream, const int max_len) const;

protected:
    // Shared optimization steps; called only by Opt function of subclasses.
    // Upon return, all support data of children and this is valid.
    Expr* OptHelp(const opInfo& opI);

    void init(Expr* E1, Expr* E2);

private:
    Expr* Mutate(const int prob, const int siz, const float ConstPerturb, const VarVals_t *VV);

    bool isequal(const Expr* E) const;
    bool isless(const Expr* E) const;
};

class And : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline bool isSymmetric() const { return true; }
    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }
    inline int getPrecedence() const { return 10; }

    std::string Print(int pstyle) const;

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    inline And() { count = 0; }
    inline And(Expr* E1, Expr* E2) { init(E1, E2); }
};

class ATan2 : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    inline ATan2() { count = 0; }
    inline ATan2(Expr* E1, Expr* E2) { init(E1, E2); }
};

class Div : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }
    inline int getPrecedence() const { return 5; }

    std::string Print(int pstyle) const;

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    inline Div() { count = 0; }
    inline Div(Expr* E1, Expr* E2) { init(E1, E2); }
};

class IFS : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    // This one is unique. Can't use the BinaryExpr implementation.
    int postTokenStream(int *TokenStream, const int max_len) const;

    inline IFS() { count = 0; }
    inline IFS(Expr* E1, Expr* E2) { init(E1, E2); }

private:
    float IFSVal(VarVals_t *VV) const;
};

class Minus : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }
    inline int getPrecedence() const { return 6; }

    std::string Print(int pstyle) const;

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    inline Minus() { count = 0; }
    inline Minus(Expr* E1, Expr* E2) { init(E1, E2); }
};

class Mod : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }
    inline int getPrecedence() const { return 5; }

    std::string Print(int pstyle) const;

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    inline Mod() { count = 0; }
    inline Mod(Expr* E1, Expr* E2) { init(E1, E2); }
};

class Mult : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline bool isSymmetric() const { return true; }
    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }
    inline int getPrecedence() const { return 5; }

    std::string Print(int pstyle) const;

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    inline Mult() { count = 0; }
    inline Mult(Expr* E1, Expr* E2) { init(E1, E2); }
};

class Or : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline bool isSymmetric() const { return true; }
    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }
    inline int getPrecedence() const { return 12; }

    std::string Print(int pstyle) const;

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    inline Or() { count = 0; }
    inline Or(Expr* E1, Expr* E2) { init(E1, E2); }
};

class Plus : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline bool isSymmetric() const { return true; }
    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }
    inline int getPrecedence() const { return 6; }

    std::string Print(int pstyle) const;

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    inline Plus() { count = 0; }
    inline Plus(Expr* E1, Expr* E2) { init(E1, E2); }
};

class Pow : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    inline Pow() { count = 0; }
    inline Pow(Expr* E1, Expr* E2) { init(E1, E2); }
};

class XOr : public BinaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline bool isSymmetric() const { return true; }
    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }
    inline int getPrecedence() const { return 11; }

    std::string Print(int pstyle) const;

    float Eval(const VarVals_t *VV = NULL) const;
    interval Ival(const opInfo &opI, const interval &lv = interval(), const interval &rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo &opI);

    inline XOr() { count = 0; }
    inline XOr(Expr* E1, Expr* E2) { init(E1, E2); }
};
