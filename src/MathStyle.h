#pragma once

#include "Expr.h"
#include "MathIndividual.h"
#include "MathStyleDefines.h"
#include "Style.h"

#include <Image/ColorMap.h>
#include <Image/tImage.h>

class MathStyle : public Style {
public:
    MathStyle();
    ~MathStyle();

    virtual Individual::shp BreedIndividual(int IDNum, int Generation, Individual::shp Aa, Individual::shp Bb, float XMin, float YMin, float BoxWid);

    virtual void LoadPopulation(const std::string& inFName);

    virtual std::string Style::getPopFileSuffix();

    //////////////////////////
    // Non-inherited methods

    // Make an MathIndividual whose ColorMap is derived from the given image
    MathIndividual::shp ImageColorMapToIndiv(const char* imgFName, int colorMapSize);

    int NumOptSteps() const { return m_numOptSteps; }
    void NumOptSteps(int val) { m_numOptSteps = val; }
    float NumOptMaxError() const { return m_numOptMaxError; }
    void NumOptMaxError(float val) { m_numOptMaxError = val; }
    bool OnlyColorMaps() const { return m_onlyColorMaps; }
    void OnlyColorMaps(bool val) { m_onlyColorMaps = val; }

    int getTotalSizeBeforeOpt() const { return m_TotalSizeBeforeOpt; }
    void setTotalSizeBeforeOpt(int val) { m_TotalSizeBeforeOpt = val; }
    int getTotalSizeAfterOpt() const { return m_TotalSizeAfterOpt; }
    void setTotalSizeAfterOpt(int val) { m_TotalSizeAfterOpt = val; }
    bool getOptimize() const { return m_allowOptimize; }
    void setOptimize(bool val) { m_allowOptimize = val; }

    VarVals_t* VarVals() const { return m_VarVals; }

private:
    int m_TotalSizeBeforeOpt; // For tracking how good optimization is upon loading
    int m_TotalSizeAfterOpt;  // For tracking how good optimization is upon loading
    int m_numOptSteps;        // How many steps to take when numeric optimizing
    float m_numOptMaxError;   // Threshold for considering a function constant in numeric optimization, on scale of 0==dark and 1==light
    bool m_allowOptimize;     // Disable expression optimization for debugging
    bool m_onlyColorMaps;     // Only load the ColorMaps of individuals, not the whole things

    VarVals_t* m_VarVals;

    Expr* BreedChannel(const MathIndividual::shp Aa, const MathIndividual::shp Bb, int chan, ColorSpace_t ColorSpace, const VarVals_t* VV);

    ColorMap<f3Pixel>* BreedColorMap(MathIndividual::shp Aa, MathIndividual::shp Bb, ColorSpace_t ColorSpace);
    ///////////////////////////////////////////////////////////////////////
    // ColorMap creation and mutation

    ColorSpace_t chooseRandomColorSpace();
};

// Fill the ColorMap with random gradient segments
void FillColorMapRandom(ColorMap<f3Pixel>& CMap);

void RandomizeColorMapOrder(ColorMap<f3Pixel>& CMap);

float ColorMapOrderScore(ColorMap<f3Pixel>& CMap);

// Change the order of the colors in the ColorMap
void ReorderColorMap(ColorMap<f3Pixel>& CMap, ColorMapOrderings_t CMapReorderStyle);

// Splice a sequential span of pixels from one and the rest from the other
ColorMap<f3Pixel>* CrossColorMap(const ColorMap<f3Pixel>& Aa, const ColorMap<f3Pixel>& Bb);

// Choose a random span within the ColorMap Aa and replace it with a smooth gradient
// Either endpoint may be the color that was already there or a new color
ColorMap<f3Pixel>* MutateColorMap(const ColorMap<f3Pixel>& Aa, float variability);

// Load and analyze the image, extracting its ColorMap
ColorMap<f3Pixel> GetColorMapFromImage(const char* InFName, int colorMapSize);
