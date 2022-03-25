#pragma once

#include "Expr.h"

class UnaryExpr : public Expr
{
public:
    inline bool isSymmetric() const { return true; }
    inline int getPrecedence() const { return 2; }
    inline int getArity() const { return 1; }

    std::string Print(int pstyle) const;

    int preTokenStream(int* TokenStream, const int max_len) const;
    int postTokenStream(int* TokenStream, const int max_len) const;

protected:
    // Shared optimization steps; called only by Opt function of subclasses.
    // Upon return, all support data of children and this is valid.
    Expr* OptHelp(const opInfo& opI);

    void init(Expr* E);

private:
    Expr* Mutate(const int Prob, const int siz, const float ConstPerturb, const VarVals_t* VV);

    bool isequal(const Expr* E) const;
    bool isless(const Expr* E) const;
};

class Abs : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Abs() { count = 0; }
    inline Abs(Expr* E) { init(E); }
};

class ACos : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline ACos() { count = 0; }
    inline ACos(Expr* E) { init(E); }
};

class ASin : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline ASin() { count = 0; }
    inline ASin(Expr* E) { init(E); }
};

class ATan : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline ATan() { count = 0; }
    inline ATan(Expr* E) { init(E); }
};

class BitNot : public UnaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }
    inline int getPrecedence() const { return 3; }

    std::string Print(int pstyle) const;

    float Eval(const VarVals_t* VV = NULL) const;
    interval Ival(const opInfo& opI, const interval& lv = interval(), const interval& rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo& opI);

    inline BitNot() { count = 0; }
    inline BitNot(Expr* E) { init(E); }
};

class Cbrt : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Cbrt() { count = 0; }
    inline Cbrt(Expr* E) { init(E); }
};

class Clamp : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Clamp() { count = 0; }
    inline Clamp(Expr* E) { init(E); }
};

class Cos : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Cos() { count = 0; }
    inline Cos(Expr* E) { init(E); }
};

class Cube : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Cube() { count = 0; }
    inline Cube(Expr* E) { init(E); }
};

class Exp : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Exp() { count = 0; }
    inline Exp(Expr* E) { init(E); }
};

class Ln : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Ln() { count = 0; }
    inline Ln(Expr* E) { init(E); }
};

class Round : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Round() { count = 0; }
    inline Round(Expr* E) { init(E); }
};

class Sin : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Sin() { count = 0; }
    inline Sin(Expr* E) { init(E); }
};

class Sqr : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Sqr() { count = 0; }
    inline Sqr(Expr* E) { init(E); }
};

class Sqrt : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Sqrt() { count = 0; }
    inline Sqrt(Expr* E) { init(E); }
};

class Tan : public UnaryExpr
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
    Expr* Opt(const opInfo& opI);

    inline Tan() { count = 0; }
    inline Tan(Expr* E) { init(E); }
};

class UnaryMinus : public UnaryExpr
{
public:
    static std::string name;
    static std::string fname;
    static int token;

    inline int getToken() const { return token; }
    inline std::string getName() const { return name; }
    inline std::string getFuncName() const { return fname; }
    inline int getPrecedence() const { return 3; }

    std::string Print(int pstyle) const;

    float Eval(const VarVals_t* VV = NULL) const;
    interval Ival(const opInfo& opI, const interval& lv = interval(), const interval& rv = interval()) const;
    Expr* Copy() const;
    Expr* Opt(const opInfo& opI);

    inline UnaryMinus() { count = 0; }
    inline UnaryMinus(Expr* E) { init(E); }
};
