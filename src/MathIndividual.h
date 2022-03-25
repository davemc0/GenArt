#pragma once

#include "Expr.h"
#include "Individual.h"
#include "MathStyleDefines.h"

#include <Image/ColorMap.h>
#include <memory>
#include <string>

class MathIndividual : public Individual {
public:
    typedef std::shared_ptr<MathIndividual> shp;

    ColorMap<f3Pixel> CMap;

    Expr* R;
    Expr* G;
    Expr* B;

    ColorSpace_t ColorSpace; // The enumerated color space to use

    // Fills this MathIndividual with the given expressions, which it copies and optimizes.
    // Takes ownership of original expressions and deletes them. Does NOT own CMap_.
    MathIndividual(Expr* Ri, Expr* Gi, Expr* Bi, ColorMap<f3Pixel>* CMap_, ColorSpace_t ColorSpace_, float Score_, int IDNum_, int Generation_, int ParentA_,
                   int ParentB_, float XMin_, float YMin_, float BoxWid_);
    MathIndividual(const std::string& Rs, const std::string& Gs, const std::string& Bs, ColorMap<f3Pixel>* CMap_, ColorSpace_t ColorSpace_, float Score_,
                   int IDNum_, int Generation_, int ParentA_, int ParentB_, float XMin_, float YMin_, float BoxWid_);

    MathIndividual(const MathIndividual& In);

    ~MathIndividual();

    int GetSpace() const;
    void SetSpace(ColorSpace_t ColorSpace_);
    void SetColorMap(ColorMap<f3Pixel>& CMap_);
    void ReplicateChannel(int c);                                      // Copy the specified channel into the other channels
    void ScaleBiasChannel(int c, const float scale, const float bias); // Copy the specified channel into the other channels
    void ShuffleVars();                                                // Reorder the variables to rotate or mirror the image
    void RandConst(const float rc);                                    // Tweak all the constants in the expressions by amount rc
    void ColorMapRandomize();                                          // Make a totally random ColorMap
    void ColorMapMutate(float v);                                      // Modify all color values randomly, proportional to variability
    void ColorMapReorder(ColorMapOrderings_t CMapReordering);          // Reorder the ColorMap in one of the possible ways
    void SetColorMapEntry(const float v, const f3Pixel& dropperColor);
    f3Pixel EvalSample(const float x, const float y, float& r); // Compute the value of the three Exprs (but not color space conversion)

    virtual std::string stringSave();
    virtual std::string stringDisplay(const float x = -1000.0f, const float y = -1000.0f);

    virtual bool equal(const Individual& p) const; // Really comparing two MathIndividuals, but signature must match base class
    virtual bool less(const Individual& p) const;  // Really comparing two MathIndividuals, but signature must match base class

private:
    void init(Expr* Ri, Expr* Gi, Expr* Bi, ColorMap<f3Pixel>* CMap_);

    void getColorSpaceIntervals(ColorSpace_t space, interval cspaceInterval[]) const; // Return the intervals of valid values for the given color space.

    Expr* OptimizeChannel(Expr* A0, VarVals_t MinVV, VarVals_t MaxVV, const int sampSteps, const float maxAbsErr, const interval outSpan);
};

// Load one MathIndividual from the file
// Returns NULL if the file ends
MathIndividual::shp fromstream(std::ifstream& InF, bool onlyColorMaps, bool& finished);
