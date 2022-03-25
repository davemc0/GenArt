#pragma once

#include "Interop.h"

#include <Image/tImage.h>
#include <memory>
#include <string>

class AutoScorer;

// Represent an individual in the population
class Individual
{
public:
    typedef std::shared_ptr<Individual> shp;

    float Score;              // How much I like this individual
    float Xmin, Ymin, BoxWid; // The square of the infinite plane to map to the image
    int Generation;           // What generation this guy was made in (max parent gen. + 1)
    int IDNum;                // Unique ID of this individual
    int ParentA, ParentB;     // The unique IDs of the parents; ParentB is -1 if none.

    Individual(float Score_, int IDNum_, int Generation_, int ParentA_, int ParentB_, float XMin_, float YMin_, float BoxWid_);

    // The copy constructor copies everything but the image, which it leaves empty.
    Individual(const Individual& In);

    ~Individual();

    uc4Image* ThumbIm();
    uc4Image* FinalIm();
    uc4DImage* ThumbImD();
    uc4DImage* FinalImD();

    void RenderLaunch(bool isFinal, AutoScorer* Scorer = NULL); // Using the parameters from RenderManager, launch a render for either Final or Thumb

    void ImClear(); // Delete all images associated with this individual and reset the flags
    void ImSave();  // Save the Final image; blocks until finished

    inline float GetScore() const { return Score; }
    inline void SetScore(float score) { Score = score; }
    inline void IncScore(float amt = 0.1f) { Score += amt; }

    inline int GetIDNum() const { return IDNum; }
    inline int GetGeneration() const { return Generation; }
    inline int GetParentA() const { return ParentA; }
    inline int GetParentB() const { return ParentB; }
    inline void requestSave() { m_requestedSave = true; }

    void Zoom(float z);           // Zoom in or out on the image
    void Pan(float xm, float ym); // Pan the image to new corner location
    void CenterX();               // Center the image in X
    void CenterY();               // Center the image in Y

    virtual std::string stringSave();
    virtual std::string stringDisplay(const float x = -1.0f, const float y = -1.0f);

    virtual bool equal(const Individual& p) const;
    virtual bool less(const Individual& p) const;

    void assignIDNum();

private:
    uc4Image* m_thumbIm;
    uc4Image* m_finalIm;

    bool m_requestedSave; // True if someone requested that the Final image be saved from within RenderLaunch

    virtual void init(float Score_, int IDNum_, int Generation_, int ParentA_, int ParentB_, float XMin_, float YMin_, float BoxWid_);
};
