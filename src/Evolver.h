#pragma once

#include "Individual.h"
#include "Population.h"

class AutoScorer;

class Evolver
{
public:
    Evolver(AutoScorer* Scorer_);
    ~Evolver();

    // Create and return a new random individual, with parents from the span
    Individual::shp Generate(const Population::iterator beginParent, const Population::iterator endParent, bool autoEvolve);

    // Evolve one generation
    void EvolveGeneration();

    int IndivsCreated() const { return m_indivsCreated; }
    void IndivsCreated(int val) { m_indivsCreated = val; }
    AutoScorer* Scorer() const { return m_Scorer; }
    void Scorer(AutoScorer* val) { m_Scorer = val; }

    void ResetScoreStats() { m_maxScore = 0; }

private:
    Individual::shp ChooseParent(const Population::iterator beginParent, const Population::iterator endParent, bool autoEvolve);

    float RandCoord();
    void computeScoreStats(const Population::iterator beginParent, const Population::iterator endParent);

    AutoScorer* m_Scorer;
    int m_indivsCreated;  // How many individuals have been evolved
    size_t m_maxZooSize;  // How many individuals to allow in the zoo
    int m_childrenPerGen; // How many individuals to move from children to zoo each generation
    float m_totalScore, m_maxScore, m_minScore;
};
