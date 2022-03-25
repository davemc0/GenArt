#include "ExprTools.h"
#include "MathIndividual.h"
#include "MathStyle.h"
#include "RenderManager.h"

#include <Util/Assert.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

extern MathStyle* MSEng;
extern RenderManager* RMan;

MathIndividual::MathIndividual(Expr* Ri, Expr* Gi, Expr* Bi, ColorMap<f3Pixel> *CMap_, ColorSpace_t ColorSpace_,
                               float Score_, int IDNum_, int Generation_, int ParentA_, int ParentB_, float XMin_, float YMin_, float BoxWid_)
                               : Individual(Score_, IDNum_, Generation_, ParentA_, ParentB_, XMin_, YMin_, BoxWid_), ColorSpace(ColorSpace_)
{
    init(Ri, Gi, Bi, CMap_);
}

MathIndividual::MathIndividual(const std::string &Rs, const std::string &Gs, const std::string &Bs, ColorMap<f3Pixel> *CMap_, ColorSpace_t ColorSpace_,
                               float Score_, int IDNum_, int Generation_, int ParentA_, int ParentB_, float XMin_, float YMin_, float BoxWid_)
                               : Individual(Score_, IDNum_, Generation_, ParentA_, ParentB_, XMin_, YMin_, BoxWid_), ColorSpace(ColorSpace_)
{
    ASSERT_R(MSEng);

    std::string remainder;
    Expr* Ri = ReadExpr(Rs, remainder, MSEng->VarVals());
    Expr* Gi = ReadExpr(Gs, remainder, MSEng->VarVals());
    Expr* Bi = ReadExpr(Bs, remainder, MSEng->VarVals());

    init(Ri, Gi, Bi, CMap_);
}

MathIndividual::MathIndividual(const MathIndividual &In) : Individual(In), CMap(In.CMap)
{
    R = In.R->Copy();
    G = In.G->Copy();
    B = In.B->Copy();

    ColorSpace = In.ColorSpace;
}

MathIndividual::~MathIndividual()
{
    delete R;
    delete G;
    delete B;

#if 0
    // Turn these on just for debugging.
    R = G = B = NULL;
#endif
}

void MathIndividual::init(Expr* Ri, Expr* Gi, Expr* Bi, ColorMap<f3Pixel> *CMap_)
{
    // Get ready for numeric optimization
    ASSERT_R(MSEng);
    const int sampSteps = MSEng->NumOptSteps();
    const float maxError = MSEng->NumOptMaxError();

    VarVals_t MaxVV = *MSEng->VarVals(), MinVV = *MSEng->VarVals();

    // Use the Individual's viewport as the interval
    ASSERT_D(MSEng->VarVals()->vals.size() == 3);
    MinVV.vals[0] = Xmin;
    MaxVV.vals[0] = Xmin + BoxWid;
    MinVV.vals[1] = Ymin;
    MaxVV.vals[1] = Ymin + BoxWid * RMan->thHgt / RMan->thWid;

    interval rivl; // Find the min and max r values of the four corners of the viewport
    rivl.extend(sqrtf(MinVV.vals[0]*MinVV.vals[0] + MinVV.vals[1]*MinVV.vals[1]));
    rivl.extend(sqrtf(MinVV.vals[0]*MinVV.vals[0] + MaxVV.vals[1]*MaxVV.vals[1]));
    rivl.extend(sqrtf(MaxVV.vals[0]*MaxVV.vals[0] + MinVV.vals[1]*MinVV.vals[1]));
    rivl.extend(sqrtf(MaxVV.vals[0]*MaxVV.vals[0] + MaxVV.vals[1]*MaxVV.vals[1]));
    // The zero-crossings
    rivl.extend(sqrtf(MinVV.vals[0]*MinVV.vals[0]));
    rivl.extend(sqrtf(MinVV.vals[1]*MinVV.vals[1]));
    rivl.extend(sqrtf(MaxVV.vals[0]*MaxVV.vals[0]));
    rivl.extend(sqrtf(MaxVV.vals[1]*MaxVV.vals[1]));

    if (MinVV.vals[0] < 0 && MaxVV.vals[0] > 0 && MinVV.vals[1] < 0 && MaxVV.vals[1] > 0)
        rivl.extend(0); // Origin inside viewport

    MinVV.vals[2] = rivl.lower;
    MaxVV.vals[2] = rivl.upper;

    MSEng->setTotalSizeBeforeOpt(MSEng->getTotalSizeBeforeOpt() + Ri->size() + Gi->size() + Bi->size());

    interval spans[3];
    getColorSpaceIntervals(ColorSpace, spans);

    R = OptimizeChannel(Ri, MinVV, MaxVV, sampSteps, maxError, spans[0]);
    G = OptimizeChannel(Gi, MinVV, MaxVV, sampSteps, maxError, spans[1]);
    B = OptimizeChannel(Bi, MinVV, MaxVV, sampSteps, maxError, spans[2]);

    MSEng->setTotalSizeAfterOpt(MSEng->getTotalSizeAfterOpt() + R->size() + G->size() + B->size());

    if (CMap_ != NULL && CMap_->size() > 0) {
        if (CMap_->size() < CMAP_SIZE) {
            CMap = ColorMap<f3Pixel>(*CMap_, CMAP_SIZE); // Upsample to desired size
        }
        else {
            CMap = *CMap_;
        }
    }

    if (CMap.size() < CMAP_SIZE) {
        CMap.C.resize(CMAP_SIZE);
        FillColorMapRandom(CMap);
    }

    // Don't create ColorMaps with invalid colors
    for (size_t j = 0; j < CMap.size(); j++) {
        if (!isFinite(CMap[j]))
            CMap[j] = f3Pixel(0.0f, 0.0f, 0.0f);
    }
}

Expr* MathIndividual::OptimizeChannel(Expr* A0, VarVals_t MinVV, VarVals_t MaxVV, const int sampSteps, const float maxAbsErr, const interval outSpan)
{
    Expr* A1 = RemoveNestedIFS(A0);
    delete A0;

    if (MSEng->getOptimize())
        A0 = Optimize(A1, MinVV, MaxVV, sampSteps, maxAbsErr, outSpan);
    else
        A0 = A1->Copy(); // Disable all optimization
    delete A1;

    return A0;
}

// Make a total random ColorMap
void MathIndividual::ColorMapRandomize()
{
    FillColorMapRandom(CMap);
    ImClear();
}

// Mutate the ColorMap in some way
void MathIndividual::ColorMapMutate(float v)
{
	CMap = *MutateColorMap(CMap, v);
    ImClear();
}

// Reorder the ColorMap in one of the possible ways
void MathIndividual::ColorMapReorder(ColorMapOrderings_t CMapReordering)
{
    ReorderColorMap(CMap, CMapReordering);
    ImClear();
}

void MathIndividual::ScaleBiasChannel(int c, const float scale, const float bias)
{
    if (c == 0) {
        ASSERT_R(R);
        Expr* T = ScaleBias(R, scale, bias);
        delete R;
        R = T;
    }
    if (c == 1) {
        ASSERT_R(G);
        Expr* T = ScaleBias(G, scale, bias);
        delete G;
        G = T;
    }
    if (c == 2) {
        ASSERT_R(B);
        Expr* T = ScaleBias(B, scale, bias);
        delete B;
        B = T;
    }

    ImClear();
}

void MathIndividual::ShuffleVars()
{
    VarVals_t reVV;
    const std::string VarNameList[] = { "y", "x", "r" };
    const float VarValueList[] = { 1.0f, 0.0f, 2.0f };

    for (unsigned int i = 0; i < sizeof(VarValueList) / sizeof(float); i++) {
        reVV.names.push_back(VarNameList[i]);
        reVV.vals.push_back(VarValueList[i]);
    }

    if (R) {
        Expr* T = ReplaceVars(R, reVV);
        delete R;
        R = T;
    }
    if (G) {
        Expr* T = ReplaceVars(G, reVV);
        delete G;
        G = T;
    }
    if (B) {
        Expr* T = ReplaceVars(B, reVV);
        delete B;
        B = T;
    }
}

// Tweak all the constants in the expressions by amount rc
void MathIndividual::RandConst(const float rc)
{
    R->PerturbConstants(rc);
    G->PerturbConstants(rc);
    B->PerturbConstants(rc);

    ImClear();
}

int MathIndividual::GetSpace() const
{
    return ColorSpace;
}

void MathIndividual::SetSpace(ColorSpace_t ColorSpace_)
{
    ColorSpace = ColorSpace_;
    ImClear();
}

void MathIndividual::SetColorMap(ColorMap<f3Pixel> &CMap_)
{
	CMap = CMap_;
	ImClear();
}

// Copy the specified channel into the other channels
void MathIndividual::ReplicateChannel(int c)
{
    ASSERT_R(R);
    ASSERT_R(G);
    ASSERT_R(B);

    if (c == 0) {
        if (G) delete G;
        G = R->Copy();
        if (B) delete B;
        B = R->Copy();
    }
    if (c == 1) {
        if (R) delete R;
        R = G->Copy();
        if (B) delete B;
        B = G->Copy();
    }
    if (c == 2) {
        if (R) delete R;
        R = B->Copy();
        if (G) delete G;
        G = B->Copy();
    }

    ImClear();
}

void MathIndividual::getColorSpaceIntervals(ColorSpace_t space, interval cspaceInterval[]) const
{
    switch(space) {
    case SPACE_RGB:
        cspaceInterval[0] = cspaceInterval[1] = cspaceInterval[2] = interval(0,1);
        break;
    case SPACE_TONEMAP_RGB:
        cspaceInterval[0] = cspaceInterval[1] = cspaceInterval[2] = interval(0,interval::infinity());
        break;
    case SPACE_YCRCB:
        cspaceInterval[0] = cspaceInterval[1] = cspaceInterval[2] = interval(-1,1); // Channels are interdependent; could specify tighter bounds.
        break;
    case SPACE_TONEMAP_HSV:
        cspaceInterval[0] = cspaceInterval[1] = cspaceInterval[2] = interval(-interval::infinity(),interval::infinity()); // Channels are interdependent; could specify tighter bounds.
        break;
    case SPACE_COLMAP:
        cspaceInterval[0] = cspaceInterval[2] = interval(0,0);
        cspaceInterval[1] = interval(0,1);
        break;
    case SPACE_TONEMAP_COLMAP:
        cspaceInterval[0] = cspaceInterval[2] = interval(0,0);
        cspaceInterval[1] = interval(0,interval::infinity());
        break;
    default:
        ASSERT_D(0);
        break;
    }
}

void MathIndividual::SetColorMapEntry(const float v, const f3Pixel& dropperColor)
{
    std::cerr << "Painting ColorMap value " << v << " to " << dropperColor << '\n';
    CMap.SetSpan(v, v, dropperColor, dropperColor);
    ImClear();
}

f3Pixel MathIndividual::EvalSample(const float x, const float y, float& r)
{
    r = sqrtf(x*x + y*y);
    VarVals_t VV = *MSEng->VarVals();
    VV.vals[0] = x;
    VV.vals[1] = y;
    VV.vals[2] = r;
    float rr = R->Eval(&VV);
    float gg = G->Eval(&VV);
    float bb = B->Eval(&VV);

    return f3Pixel(rr, gg, bb);
}

std::string MathIndividual::stringSave()
{
    std::ostringstream outStream;

    outStream << Individual::stringSave() << " " << GetSpace() << std::endl;
    if (GetSpace() == SPACE_COLMAP || GetSpace() == SPACE_TONEMAP_COLMAP) {
        outStream << G->Print(PREFIX) << std::endl;
        size_t Sz = CMap.size();
        outStream << Sz << std::endl;
        for (size_t j = 0; j < Sz; j++) {
            if (isFinite(CMap[j]))
                outStream << CMap[j];
            else
                outStream << f3Pixel(0.0f, 0.0f, 0.0f);
        }
        outStream << std::endl;
    }
    else {
        outStream << R->Print(PREFIX) << std::endl;
        outStream << G->Print(PREFIX) << std::endl;
        outStream << B->Print(PREFIX) << std::endl;
    }

    return outStream.str();
}

std::string MathIndividual::stringDisplay(const float x /*= -1000.0f*/, const float y /*= -1000.0f*/)
{
    bool inWindow = x != -1000.0f && y != -1000.0f;

    std::ostringstream out;
    FloatFmt(out, 3);

    out << Individual::stringDisplay();
    out << std::endl;
    float r;
    f3Pixel s = EvalSample(x, y, r);

	bool c = ColorSpace != SPACE_COLMAP && ColorSpace != SPACE_TONEMAP_COLMAP;

    if (inWindow) {
		out << ColorSpaceNames[ColorSpace] << "(x=" << x << ", y=" << y << ", r=" << r << ") = ";
		if (c) out << s.r() << ",";
		out << s.g();
		if (c) out << "," << s.b();
		out << std::endl << " " << std::endl;
    }
    else {
        out << " " << std::endl; // Evaluatable
        out << "R = " << R->Print(FUNC_EVAL|OP_EVAL_IMPURE) << std::endl;
        out << "G = " << G->Print(FUNC_EVAL|OP_EVAL_IMPURE) << std::endl;
        out << "B = " << B->Print(FUNC_EVAL|OP_EVAL_IMPURE) << std::endl;
        out << " " << std::endl;
    }

    if (c) out << "R" << tostring(R->ivl) << " = " << R->Print(PREFIX) << std::endl;
    out << "G" << tostring(G->ivl) << " = " << G->Print(PREFIX) << std::endl;
    if (c) out << "B" << tostring(B->ivl) << " = " << B->Print(PREFIX) << std::endl;
    out << " " << std::endl;

    if (c) out << "R = " << R->Print(INFIX) << std::endl; // Readable
    out << "G = " << G->Print(INFIX) << std::endl;
    if (c) out << "B = " << B->Print(INFIX) << std::endl;
    out << " " << std::endl;

    if (!c || !inWindow) {
        out << CMap.size() << std::endl;
        for (size_t i = 0; i < CMap.size(); i++) {
            out << CMap[i] << std::endl;
        }
    }
    out << std::endl;
    return out.str();
}

bool MathIndividual::equal(const Individual &p_) const
{
    const MathIndividual& p = static_cast<const MathIndividual&>(p_);

    return static_cast<const Individual&>(*this).Individual::equal(p_) &&
        ColorSpace == p.ColorSpace && G->isequal(p.G) &&
        (((ColorSpace == SPACE_COLMAP || ColorSpace == SPACE_TONEMAP_COLMAP) && Equal(CMap, p.CMap)) ||
        ((ColorSpace != SPACE_COLMAP && ColorSpace != SPACE_TONEMAP_COLMAP) && R->isequal(p.R) && B->isequal(p.B)));
}

bool MathIndividual::less(const Individual &p_) const
{
    const MathIndividual& p = static_cast<const MathIndividual&>(p_);

    if (ColorSpace < p.ColorSpace) {
        return true;
    }
    else if (ColorSpace > p.ColorSpace) {
        return false;
    }
    else {
        if (ColorSpace == SPACE_COLMAP || ColorSpace == SPACE_TONEMAP_COLMAP) {
            if (CMap.size() < p.CMap.size()) {
                return true;
            }
            else if (CMap.size() > p.CMap.size()) {
                return false;
            }
            else {
                return IDNum < p_.IDNum;
            }
        }
        else {
            return IDNum < p_.IDNum;
        }
    }
}

MathIndividual::shp fromstream(std::ifstream& InF, bool onlyColorMaps, bool& finished)
{
    finished = false;
    std::string inl;
    if (!getline(InF, inl)) {
        finished = true;
        return NULL;
    }

    float Score = 0.0f;
    int ColorSpace = -1;
    int IDNum = -1, Generation = -1, ParentA = -1, ParentB = -1;
    float XMin, YMin, BoxWid;

    std::istringstream IS(inl);

#if 1
    IS >> Score >> IDNum >> Generation >> ParentA >> ParentB >> XMin >> YMin >> BoxWid >> ColorSpace;
#else
    // For reading super old files that were used for Bear Lake walls
    float tmp;
    IS >> Score >> tmp >> IDNum >> Generation >> ParentA >> ParentB >> BoxWid;
    XMin = 0;
    YMin = 0;
    ColorSpace = SPACE_COLMAP;
#endif

    Score = floorf(0.5f + float(Score) * 100.0f) * 0.01f;

    std::string Ch0, Ch1, Ch2;
    getline(InF, Ch0);

    ColorMap<f3Pixel>* CMap = NULL;

    if (ColorSpace == SPACE_COLMAP || ColorSpace == SPACE_TONEMAP_COLMAP) {
        std::cerr << 'C';
        int CMapSize;
        InF >> CMapSize;
        getline(InF, inl);
        CMap = new ColorMap<f3Pixel>(CMapSize);
        for (int k = 0; k < CMapSize; k++) InF >> (*CMap)[k];

        getline(InF, inl);

        if (onlyColorMaps) {
            Ch0 = std::string("/ + * 4 - * y 4 % * y 4 1 - * x 4 % * x 4 1 16");
        }
        Ch1 = Ch0;
        Ch0 = "0";
        Ch2 = "0";
    }
    else {
        std::cerr << '3';
        getline(InF, Ch1);
        getline(InF, Ch2);

        if (onlyColorMaps) {
            return NULL;
        }
    }

    MathIndividual::shp ind(new MathIndividual(Ch0, Ch1, Ch2, CMap, ColorSpace_t(ColorSpace),
        Score, IDNum, Generation, ParentA, ParentB, XMin, YMin, BoxWid));

    delete CMap;

    return ind;
}
