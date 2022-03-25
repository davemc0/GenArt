#pragma once

#include "Individual.h"

#include <Image/tImage.h>

class AutoScorer
{
public:
    // Compute the score and store it in the individual
    virtual void ComputeScore(Individual* Ind) = 0;
};

class ColorfulnessAutoScorer : public AutoScorer
{
public:
    void ComputeScore(Individual* Ind);
};

class RandomAutoScorer : public AutoScorer
{
public:
    void ComputeScore(Individual* Ind);
};

class ImageSimilarityAutoScorer : public AutoScorer
{
public:
    ImageSimilarityAutoScorer(uc4Image* Img);
    ~ImageSimilarityAutoScorer();

    void ComputeScore(Individual* Ind);

private:
    void init(Individual* Ind);

    uc4Image* TargetImg; // The image to evolve to be like
};
