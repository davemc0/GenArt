#include "MathStyle.h"

#include "Counters.h"
#include "ExprTools.h"
#include "Image/ColorMap.h"
#include "Image/ImageAlgorithms.h"
#include "Image/Quant.h"
#include "Math/MiscMath.h"
#include "Math/Random.h"
#include "MathIndividual.h"
#include "Population.h"

#include <algorithm>
#include <fstream>

extern Counters* C;
extern Population* Pop;

const int ISIZE = 12;             // Make random trees about this big
const int MUTSIZE = 6;            // Make random subtrees during mutation about this big
const int MUTPROB = 5;            // Mutate about 1/MUTPROB of the nodes in an Expr each time
const float CONST_PERTURB = 0.2f; // Add NRand(CONST_PERTURB) to constants

inline float L2Norm(const f3Pixel& p)
{
    return sqrtf(dmcm::Sqr(p[0]) + dmcm::Sqr(p[1]) + dmcm::Sqr(p[2]));
}

MathStyle::MathStyle()
{
    m_numOptSteps = 40;
    m_numOptMaxError = 0.005f; // Really need to measure the effect on the final image, not on the subexpression's interval.
    m_TotalSizeBeforeOpt = 0;
    m_TotalSizeAfterOpt = 0;
    m_allowOptimize = true;
    m_onlyColorMaps = false;

    m_VarVals = new VarVals_t;
    InitVVals(*m_VarVals);
}

MathStyle::~MathStyle()
{
    delete m_VarVals;
}

void MathStyle::LoadPopulation(const std::string & inFName)
{
    std::cerr << "Loading " << inFName << '\n';

    std::ifstream InF(inFName);
    ASSERT_RM(InF.is_open(), "Couldn't open file " + std::string(inFName) + " to read population.");

    while (true) {
        bool finished = false;
        Individual::shp newInd = fromstream(InF, m_onlyColorMaps, finished);
        if (finished) break;
        if (newInd == NULL) continue;

        // Make sure it doesn't match an existing individual
        // Do not optimize this. It's only 1% of the load time.
        bool skip = false;
        for (auto ind = Pop->beginZoo(); ind != Pop->endZoo(); ind++) {
            if ((*ind)->equal(*newInd)) {
                std::cerr << "Equal ones: " << (*ind)->stringDisplay() << '\n' << newInd->stringDisplay() << '\n';
                skip = true;
                break;
            } else if ((*ind)->IDNum == newInd->IDNum) {
                newInd->assignIDNum(); // Don't let any two have the same ID num.
            }
        }

        if (!skip) Pop->insertZoo(newInd);
    }

    InF.close();
    std::cerr << "\nDone loading: " << Pop->sizeZoo() << " total individuals.\n";

    std::cerr << "TotalSizeBeforeOpt=" << getTotalSizeBeforeOpt() << " TotalSizeAfterOpt=" << getTotalSizeAfterOpt() << '\n';
}

std::string MathStyle::getPopFileSuffix()
{
    return ".gnx";
}

MathIndividual::shp MathStyle::ImageColorMapToIndiv(const char* imgFName, int colorMapSize)
{
    ColorMap<f3Pixel> CMap = GetColorMapFromImage(imgFName, colorMapSize);

    std::string G("+ x 0.01");
    std::string R("0"), B("0");

    MathIndividual::shp ind(new MathIndividual(R, G, B, &CMap, SPACE_COLMAP, 1.1f, -1, 1, -1, -1, -1, -1, 2));

    return ind;
}

ColorSpace_t MathStyle::chooseRandomColorSpace()
{
    return ColorSpace_t(randn(NUM_COLORSPACES));
}

Expr* MathStyle::BreedChannel(const MathIndividual::shp Aa, const MathIndividual::shp Bb, int chan, ColorSpace_t ColorSpace, const VarVals_t* VV)
{
    C->inc("Chan");
    if ((ColorSpace == SPACE_COLMAP || ColorSpace == SPACE_TONEMAP_COLMAP) && chan != 1) {
        // Don't make a channel that won't get seen and scored
        C->inc("Chan.Const");
        return MakeConst(0);
    }

    // Make a list of all valid Exprs to choose from
    std::vector<Expr*> chanChoices;
    if (Aa != NULL) {
        chanChoices.push_back(Aa->G);
        C->inc("Chan.AG");
        if ((Aa->ColorSpace != SPACE_COLMAP && Aa->ColorSpace != SPACE_TONEMAP_COLMAP)) {
            chanChoices.push_back(Aa->R);
            C->inc("Chan.AR");
            chanChoices.push_back(Aa->B);
            C->inc("Chan.AB");
        }
    }

    if (Bb != NULL) {
        chanChoices.push_back(Bb->G);
        C->inc("Chan.BG");
        if ((Bb->ColorSpace != SPACE_COLMAP && Bb->ColorSpace != SPACE_TONEMAP_COLMAP)) {
            chanChoices.push_back(Bb->R);
            C->inc("Chan.BR");
            chanChoices.push_back(Bb->B);
            C->inc("Chan.BB");
        }
    }

    // Zero input channels
    if (chanChoices.size() == 0 || chance(0.25f * m_variability)) {
        C->inc("Chan.RandExpr");
        if (chanChoices.size() > 0) C->inc("Chan.RandExpr.One");
        return RandExpr(ISIZE, VV);
    }

    // One input channel
    if (chanChoices.size() == 1 || chance(1.0f - m_variability)) {
        C->inc("Chan.One");
        if (chanChoices.size() > 1) C->inc("Chan.One.Two");
        Expr* M = chanChoices[randn((int)chanChoices.size())];

        float c = 2.0f * (DRandf() + 1.0f - m_variability);
        switch (int(c)) {
        case 0: C->inc("Chan.One.MutateExpr"); return MutateExpr(M, MUTPROB, MUTSIZE, CONST_PERTURB, VV);
        case 1: C->inc("Chan.One.PerturbConstants"); return M->Copy()->PerturbConstants(CONST_PERTURB);
        default: C->inc("Chan.One.Copy"); return M->Copy();
        }
    }

    // Two input channels
    Expr* M0 = chanChoices[randn((int)chanChoices.size())];
    Expr* M1;
    do {
        M1 = chanChoices[randn((int)chanChoices.size())];
    } while (M1 == M0);

    if (chance(1, 2)) {
        C->inc("Chan.Two.CrossExprs");
        return CrossExprs(M0, M1);
    } else {
        C->inc("Chan.Two.BlendExprs");
        return BlendExprs(M0, M1);
    }
}

ColorMap<f3Pixel>* MathStyle::BreedColorMap(MathIndividual::shp Aa, MathIndividual::shp Bb, ColorSpace_t ColorSpace)
{
    ColorMap<f3Pixel>* CMap = NULL;
    C->inc("CMap");
    if (ColorSpace == SPACE_COLMAP || ColorSpace == SPACE_TONEMAP_COLMAP) {
        C->inc("CMap.CMap");

        // Put viable parent CMaps in a list
        std::vector<ColorMap<f3Pixel>*> cmapChoices;
        if (Aa != NULL && (Aa->ColorSpace == SPACE_COLMAP || Aa->ColorSpace == SPACE_TONEMAP_COLMAP)) {
            cmapChoices.push_back(&(Aa->CMap));
            C->inc("CMap.A");
        }
        if (Bb != NULL && (Bb->ColorSpace == SPACE_COLMAP || Bb->ColorSpace == SPACE_TONEMAP_COLMAP)) {
            cmapChoices.push_back(&(Bb->CMap));
            C->inc("CMap.B");
        }

        // Zero inputs
        if (cmapChoices.size() == 0 || chance(0.25f * m_variability)) {
            C->inc("CMap.Random");
            if (cmapChoices.size() > 0) C->inc("CMap.Random.One");

            CMap = new ColorMap<f3Pixel>(CMAP_SIZE);
            FillColorMapRandom(*CMap);

            return CMap;
        }

        // One input
        if (cmapChoices.size() == 1 || chance(0.25f * m_variability)) {
            C->inc("CMap.One");
            if (cmapChoices.size() > 1) C->inc("CMap.One.Two");
            ColorMap<f3Pixel>* M = cmapChoices[randn((int)cmapChoices.size())];
            CMap = MutateColorMap(*M, m_variability);
            C->inc("CMap.One.MutateColorMap");

            if (chance(m_variability)) {
                ReorderColorMap(*CMap, ColorMapOrderings_t(randn(NUM_REORDERS)));
                C->inc("CMap.One.ReorderColorMap");
            }

            return CMap;
        }

        // Two inputs
        C->inc("CMap.Two");
        CMap = CrossColorMap(*cmapChoices[0], *cmapChoices[1]);

        ASSERT_R(CMap->size() > 0);
    }

    return CMap;
}

Individual::shp MathStyle::BreedIndividual(int IDNum, int Generation, Individual::shp A_, Individual::shp B_, float XMin, float YMin, float BoxWid)
{
    ASSERT_R(A_ != B_ || A_ == NULL);
    MathIndividual::shp Aa(std::dynamic_pointer_cast<MathIndividual>(A_));
    MathIndividual::shp Bb(std::dynamic_pointer_cast<MathIndividual>(B_));

    ColorSpace_t ColorSpace = chooseRandomColorSpace();
    if (Bb && chance(1.0f - m_variability)) ColorSpace = Bb->ColorSpace;
    if (Aa && chance(1.0f - m_variability)) ColorSpace = Aa->ColorSpace;

    Expr* R = BreedChannel(Aa, Bb, 0, ColorSpace, VarVals());
    Expr* G = BreedChannel(Aa, Bb, 1, ColorSpace, VarVals());
    Expr* B = BreedChannel(Aa, Bb, 2, ColorSpace, VarVals());

    ColorMap<f3Pixel>* CMap = BreedColorMap(Aa, Bb, ColorSpace);

    MathIndividual::shp newInd(new MathIndividual(R, G, B, CMap, ColorSpace, DRandf(0.0f, 0.01f), IDNum, Generation, Aa ? Aa->IDNum : -1, Bb ? Bb->IDNum : -1, XMin, YMin, BoxWid));

    delete CMap;

    if (IsConst(newInd->R) && IsConst(newInd->G) && IsConst(newInd->B)) {
        // Made a boring, flat one, so kill it and recursively make another one
        return BreedIndividual(IDNum, Generation, A_, B_, XMin, YMin, BoxWid);
    }

    return newInd;
}

void RandomizeColorMapOrder(ColorMap<f3Pixel>& CMap)
{
    size_t Siz = CMap.size();
    for (size_t i = 0; i < Siz; i++) {
        size_t k = size_t(LRand(0, int(Siz)));
        f3Pixel Tmp = CMap[k];
        CMap[k] = CMap[i];
        CMap[i] = Tmp;
    }
}

float ColorMapOrderScore(ColorMap<f3Pixel>& CMap)
{
    float T = 0.0f;
    for (size_t i = 0; i < CMap.size() - 1; i++) { T += L2Norm(CMap[i] - CMap[i + 1]); }

    return T;
}

// Compare two f3Pixels by luminance
bool luminance_less(const f3Pixel& a, const f3Pixel& b)
{
    return a.luminance() < b.luminance();
}

void FillColorMapRandom(ColorMap<f3Pixel>& CMap)
{
    for (size_t i = 0; i < CMap.size(); i++) CMap[i] = f3Pixel(DRandf(), DRandf(), DRandf());
}

void ReorderColorMap(ColorMap<f3Pixel>& CMap, ColorMapOrderings_t CMapReorderStyle)
{
    if (CMapReorderStyle == REORDER_RANDOM) {
        RandomizeColorMapOrder(CMap);
        return;
    }

    if (CMapReorderStyle == REORDER_LUM) {
        sort(CMap.C.begin(), CMap.C.end(), luminance_less);
        return;
    }

    ColorMap<f3Pixel> BestCMap = CMap;
    float BestScore = ColorMapOrderScore(BestCMap);

    size_t Siz = CMap.size();

    if (Siz > 2) {
        // Find the shortest (or longest) color space path through the ColorMap
        // Randomize the order, then do permutations to shorten the order
        for (int j = 0; j < 100; j++) {
            RandomizeColorMapOrder(CMap);
            for (size_t i = 0; i < Siz * Siz * 20; i++) {
                int k = LRand(0, int(Siz - 2));
                float D0 = L2Norm(CMap[k] - CMap[k + 1]);
                float D1 = L2Norm(CMap[k + 1] - CMap[k + 2]);
                float D2 = L2Norm(CMap[k] - CMap[k + 2]);

                if ((CMapReorderStyle == REORDER_SHORT && (D0 + D1 > D1 + D2)) || (CMapReorderStyle == REORDER_LONG && (D0 + D1 < D1 + D2))) {
                    f3Pixel Tmp = CMap[k + 1];
                    CMap[k + 1] = CMap[k + 2];
                    CMap[k + 2] = Tmp;
                }
            }

            float Score = ColorMapOrderScore(CMap);
            if ((CMapReorderStyle == REORDER_SHORT && (Score < BestScore)) || (CMapReorderStyle == REORDER_LONG && (Score > BestScore))) {
                BestScore = Score;
                BestCMap = CMap;
            }
            // std::cerr << Score << std::endl;
        }
    }

    CMap = BestCMap;
    // std::cerr << BestScore << " is best.\n";
}

ColorMap<f3Pixel> GetColorMapFromImage(const char* imgFName, int colorMapSize)
{
    ASSERT_R(imgFName != NULL);
    std::cerr << "Making ColorMap from image " << imgFName << std::endl;
    ColorMap<f3Pixel> CMap(colorMapSize);

    uc3Image SampImg;
    SampImg.Load(imgFName);

    ASSERT_RM(SampImg.w() > 0, "Image didn't load right.");

    f3Image Tmp1Img = SampImg;
    while (Tmp1Img.size() > (400 * 400)) {
        f3Image TmpImg;
        Downsample2x2(TmpImg, Tmp1Img);
        Tmp1Img=TmpImg;
        // Tmp1Img.swap(TmpImg);
    }
    SampImg = Tmp1Img;

    QuantParams QP;
    QP.maxColorPalette = CMap.size();
    QP.maxIters = QP.maxItersFast = 64;
    Quantizer<uc3Pixel, unsigned char> Qnt(SampImg.pp(), SampImg.size(), false, QP);

    // Extract the image's ColorMap
    for (size_t i = 0; i < CMap.size(); i++) {
        CMap[i] = Qnt.GetColorMap()[i]; // Converts ColorMap from f3Pixel to uc3Pixel.
        std::cerr << CMap[i] << std::endl;
    }

    // ReorderColorMap(CMap, REORDER_SHORT);

    // This code is just for testing.
    Qnt.GetQuantizedTrueColorImage(SampImg.pp());
    SampImg.Save("testing.jpg");

    return CMap;
}

// Make a probability density function of the given number of elements and total weight
namespace {
float* makePDF(int Sz, float scale)
{
    float* p = new float[Sz];
    float sum = 0;
    // Sprinkle probability mass
    for (int i = 0; i < Sz; i++) {
        p[i] = DRand();
        sum += p[i];
    }

    // Normalize
    for (int i = 0; i < Sz; i++) { p[i] = scale * p[i] / sum; }

    return p;
}
}; // namespace

// Modify all color values randomly such that the total change is proportional to variability. Returns a copy.
ColorMap<f3Pixel>* MutateColorMap(const ColorMap<f3Pixel>& Aa, float variability)
{
    size_t Sz = Aa.size();

    ColorMap<f3Pixel>* M = new ColorMap<f3Pixel>(Sz);

    float* p = makePDF((int)Sz, variability);

    // Tweak each ColorMap entry by the alotted amount of the total variability
    for (size_t i = 0; i < Sz; i++)
        (*M)[i] = f3Pixel(dmcm::Saturate(Aa[i].r() + DRandf(-p[i], p[i])), dmcm::Saturate(Aa[i].g() + DRandf(-p[i], p[i])), dmcm::Saturate(Aa[i].b() + DRandf(-p[i], p[i])));

    delete[] p;

    return M;
}

// Randomly select colormap entries from the two parents
ColorMap<f3Pixel>* CrossColorMap(const ColorMap<f3Pixel>& Aa, const ColorMap<f3Pixel>& Bb)
{
    size_t Sz = std::min(Aa.size(), Bb.size());

    ColorMap<f3Pixel>* M = new ColorMap<f3Pixel>(Sz);

    for (size_t i = 0; i < Sz; i++) (*M)[i] = chance(0.5f) ? Aa[i] : Bb[i];

    return M;
}
