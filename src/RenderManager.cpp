#include "RenderManager.h"

#include "MathStyleCUDARender.h"
#include "Population.h"
#include "UI.h"

#include <Util/Timer.h>
#include <deque>
#include <iostream>

extern Population* Pop;
extern RenderManager* RMan;

RenderManager::RenderManager() : thumbQuality(16), finalQuality(16)
{
    thWid = THUMB_WID_HGT;
    thHgt = THUMB_WID_HGT;
    finalWid = FINAL_WID_HGT;
    finalHgt = FINAL_WID_HGT;
    imageSaveFormat = "png";
}

void RenderManager::PushToFinalRenderQueue(Individual::shp ind)
{
    // std::cerr << "Queuing " << ind->GetIDNum() << " for rendering at " << finalWid << "x" << finalHgt << " Quality=" << finalQuality << '\n';

    if (FinalRenderQueue.size() && FinalRenderQueue[0] == ind) return;
    FinalRenderQueue.push_front(ind);
}

void RenderManager::PushAllToFinalRenderQueue(const size_t first)
{
    std::cerr << "\nRendering and saving all images in Zoo starting with number " << first << ".\n";
    for (auto ind = Pop->rbeginZoo(); ind != Pop->rendZoo() - first; ++ind) { // Enqueue backwards since it renders in reverse order
        (*ind)->requestSave();
        PushToFinalRenderQueue(*ind);
    }
}

void RenderManager::PushAnimation(Individual::shp ind)
{
    // Make a copy so its ID number can change on each frame
    Individual::shp indCopy(new Individual(*ind));

    float Scale = 0.99f;

    // Zoom in on the origin
    for (int i = 0; i < 2000; i++) {
        indCopy->IDNum = i;
        indCopy->Xmin *= Scale;
        indCopy->Ymin *= Scale;
        indCopy->BoxWid *= Scale;

        indCopy->requestSave();
        PushToFinalRenderQueue(indCopy);
    }
}

bool RenderManager::DoVisibleThumbRenderWork(Population::iterator beginVZoo, Population::iterator endVZoo, AutoScorer* Scorer)
{
    const int INDIVS_TO_ENQ = 5; // XXX I forgot what this is based on.
    int enqueued = 0;

    // Render the visible part of the Zoo
    for (auto ind = beginVZoo; ind != endVZoo; ind++) {
        if ((*ind)->ThumbImD()->needsLaunch()) {
            (*ind)->RenderLaunch(false, Scorer);
            if (enqueued++ >= INDIVS_TO_ENQ) return true;
        }
    }

    // Render the parents
    for (auto ind = Pop->beginParents(); ind != Pop->endParents(); ind++) {
        if ((*ind)->ThumbImD()->needsLaunch()) {
            (*ind)->RenderLaunch(false, Scorer);
            if (enqueued++ >= INDIVS_TO_ENQ) return true;
        }
    }

    // Render the children
    for (auto ind = Pop->beginChildren(); ind != Pop->endChildren(); ind++) {
        if ((*ind)->ThumbImD()->needsLaunch()) {
            (*ind)->RenderLaunch(false, Scorer);
            if (enqueued++ >= INDIVS_TO_ENQ) return true;
        }
    }

    // We never get here if we did any rendering.

    return enqueued > 0;
}

bool RenderManager::DoHiddenRenderWork(AutoScorer* Scorer)
{
    // Render the non-visible part of the Zoo
    for (auto ind = Pop->beginZoo(); ind != Pop->endZoo(); ind++) {
        if ((*ind)->ThumbImD()->needsLaunch()) {
            (*ind)->RenderLaunch(false, Scorer);
            return true;
        }
    }

    return false;
}

bool RenderManager::DoFinalRenderQueueWork(bool renderSaveAll)
{
    bool rendered = false;
    float rtime = 0;
    for (auto ind = FinalRenderQueue.begin(); ind != FinalRenderQueue.end(); ind++) {
        if ((*ind)->FinalImD()->needsLaunch()) {
            (*ind)->RenderLaunch(true); // Launches a render, and if m_requestedSave is true, blocks on, and saves
            rendered = true;
            if (renderSaveAll) {
                float t = (*ind)->FinalImD()->renderTime();
                std::cerr << "time = " << t << " msec\n";
                rtime += t;
                (*ind)->ImClear(); // To save memory
            } else
                break;
        }
    }

    if (renderSaveAll) std::cerr << "Total render time = " << rtime << " msec\n";

    return rendered;
}

bool RenderManager::isRenderQueueEmpty()
{
    bool empt = FinalRenderQueue.empty();
    return empt;
}

void RenderManager::setQuality(Quality_t& targetQ, const Quality_t& Q)
{
    targetQ = Q;
    Pop->ClearImages();
}

void RenderManager::inputQuality(Quality_t& targetQ)
{
    Quality_t Q;
    std::cerr << "Old Quality = " << Q << std::endl;
    std::cerr << "MinSamples >";
    std::cin >> Q.MinSamples;
    std::cerr << "New Quality = " << Q << std::endl;

    setQuality(targetQ, Q);
}
