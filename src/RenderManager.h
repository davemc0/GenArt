#pragma once

#include "Individual.h"
#include "MathStyleDefines.h"
#include "Population.h"
#include "Style.h"

#include <deque>

// For 4K, good values are 364 and 1288.
const int THUMB_WID_HGT = 160; // Width and height of thumbnail images
const int FINAL_WID_HGT = 512; // Default width and height of final images

class RenderManager
{
public:
    int thWid, thHgt;       // Size of thumbnail images
    Quality_t thumbQuality; // Rendering quality of thumbnail images

    int finalWid, finalHgt; // Size of large images to be rendered and saved, specified on cmd line
    Quality_t finalQuality; // Rendering quality of final images

    std::string imageSaveFormat; // File format of images to be saved

    RenderManager();

    ~RenderManager() {}

    // Push a copy of the individual onto FinalRenderQueue to be rendered at the given resolution and quality
    void PushToFinalRenderQueue(Individual::shp ind);

    // Push a copy of every individual, starting with index ind, onto FinalRenderQueue to be rendered at final resolution and quality
    void PushAllToFinalRenderQueue(const size_t first);

    // Push frames of an animation onto FinalRenderQueue to be rendered at the given resolution and quality
    void PushAnimation(Individual::shp ind);

    // Perform one piece of rendering work
    // This will be called repeatedly by the rendering thread to search for work.
    // Returns true if it did work.
    bool DoVisibleThumbRenderWork(Population::iterator beginVZoo, Population::iterator endVZoo, AutoScorer* Scorer);
    bool DoHiddenRenderWork(AutoScorer* Scorer);
    bool DoFinalRenderQueueWork(bool renderSaveAll);

    std::deque<Individual::shp>& getFinalRenderQueue() { return FinalRenderQueue; }

    bool isRenderQueueEmpty();

    void setQuality(Quality_t& targetQ, const Quality_t& Q);
    void inputQuality(Quality_t& ThumbQuality);

private:
    std::deque<Individual::shp> FinalRenderQueue; // List of individuals to render at final quality
};
