#pragma once

// A Style has all global information about this render style, including how to represent it, evolve it, and render it.

#include "Individual.h"

#include <string>

// Describes the quality level for rendering the image
struct Quality_t
{
    float MinSamples;

    Quality_t(const float Min = 0) {
        MinSamples = Min;
    }
    bool operator==(const Quality_t &A) const {
        return MinSamples == A.MinSamples;
    }
    bool operator!=(const Quality_t &A) const { return !(*this == A); }
};

class Style
{
public:
    Style();
    ~Style() {}

    virtual Individual::shp BreedIndividual(int IDNum, int Generation, Individual::shp Aa, Individual::shp Bb, float XMin, float YMin, float BoxWid) = 0;

    virtual void LoadPopulation(std::string& inFName) = 0;

    virtual std::string getPopFileSuffix() = 0;

	float getVariability() { return m_variability; }
	void setVariability(float val) { m_variability = val < 0.0f ? 0.0f : val > 1.0f ? 1.0f : val; }

protected:
	float m_variability; // How different children should be from parents (0.0=similar, 1.0=different)
};

inline std::ostream &operator<<(std::ostream &oStr, const Quality_t &Q)
{
    oStr << Q.MinSamples;
    return oStr;
}
