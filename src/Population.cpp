#include "Individual.h"
#include "Population.h"
#include "Style.h"

#include "Math/Random.h"

#include <fstream>
#include <sstream>

Population::Population()
{
    m_FNameBase = std::string("Ex") + static_cast<std::ostringstream*>(&(std::ostringstream() << LRand(1000, 10000)))->str();
}

void Population::MoveParentToZoo(size_t i)
{
    Zoo.push_back(Parents[i]); Parents.erase(beginParents() + i);
}

void Population::MoveChildToZoo(size_t i)
{
    Zoo.push_back(Children[i]); Children.erase(beginChildren() + i);
}

void Population::MoveChildToParent(size_t i)
{
    Parents.push_back(Children[i]); Children.erase(beginChildren() + i);
}

void Population::MoveZooToParent(size_t i)
{
    Parents.push_back(Zoo[i]); Zoo.erase(beginZoo() + i);
}

void Population::ClearImages()
{
    for (auto ind = beginParents(); ind != endParents(); ind++)
        (*ind)->ImClear();
    for (auto ind = beginChildren(); ind != endChildren(); ind++)
        (*ind)->ImClear();
    for (auto ind = beginZoo(); ind != endZoo(); ind++)
        (*ind)->ImClear();
}

void Population::DeleteSpanOfIndividuals(size_t index, size_t count, bool AmParent, bool AmChild)
{
    ASSERT_D(!(AmParent && AmChild));
    if (count == 0) count = 0xfffffff;
    size_t Last = index + count;

    if (AmParent) {
        if (Last > sizeParents())
            Last = sizeParents();
        if (index < Last && index < sizeParents())
            Parents.erase(beginParents() + index, beginParents() + Last);
    }
    else if (AmChild) {
        if (Last > sizeChildren())
            Last = sizeChildren();
        if (index < Last && index < sizeChildren())
            Children.erase(beginChildren() + index, beginChildren() + Last);
    }
    else {
        if (Last > sizeZoo())
            Last = sizeZoo();
        if (index < Last && index < sizeZoo())
            Zoo.erase(beginZoo() + index, beginZoo() + Last);
    }
}

void Population::SavePopulation(std::string& outFNameBase, Style* style)
{
    try {
        std::string outFName = outFNameBase + style->getPopFileSuffix();
        CopyFile(outFName, outFNameBase + "_bkp" + style->getPopFileSuffix());

        std::ofstream outStream(outFName);
        if (!outStream.is_open())
            throw DMcError("SavePopulation: Couldn't open file " + outFName);

        std::cerr << "Saving Parents and Zoo to " << outFName << '\n';

        for (auto ind = beginParents(); ind != endParents(); ind++)
            outStream << (*ind)->stringSave();
        for (auto ind = beginZoo(); ind != endZoo(); ind++)
            outStream << (*ind)->stringSave();

        outStream.close();

        return;
    }
    catch (DMcError& e) {
        std::cerr << "FILE SAVE FAILED: " << e.Er << '\n';
    }
    catch (...) {
        std::cerr << "FILE SAVE FAILED.\n";
    }

    std::string outname = "C:/Users/davemc/Failsafe";
    std::cerr << "Saving backup to " << outname << style->getPopFileSuffix() << '\n';
    SavePopulation(outname, style);
}

namespace {

    bool IndivCompare_IDNum(const Individual::shp Aa, const Individual::shp Bb)
    {
        return Aa->IDNum < Bb->IDNum;
    }

    bool IndivCompare_Score(const Individual::shp Aa, const Individual::shp Bb)
    {
        if (Aa->Score < Bb->Score) {
            return false;
        }
        else if (Aa->Score > Bb->Score) {
            return true;
        }
        else {
            return Aa->IDNum < Bb->IDNum;
        }
    }

    bool IndivCompare_ColorSpace(const Individual::shp Aa, const Individual::shp Bb)
    {
        return Aa->less(*Bb);
    }

    bool IndivCompare_RenderTime(const Individual::shp Aa, const Individual::shp Bb)
    {
        if (Aa->ThumbImD()->renderTime() > Bb->ThumbImD()->renderTime()) {
            return true;
        }
        else if (Aa->ThumbImD()->renderTime() < Bb->ThumbImD()->renderTime()) {
            return false;
        }
        else {
            return Aa->IDNum < Bb->IDNum;
        }
    }
};

void Population::Sort(const SortCriterion SortBy, bool Z/*=true*/, bool P/*=true*/, bool C/*=false*/)
{
    switch (SortBy) {
    case SORTBY_SCORE:
        if (Z) sort(beginZoo(), endZoo(), IndivCompare_Score);
        if (P) sort(beginParents(), endParents(), IndivCompare_Score);
        if (C) sort(beginChildren(), endChildren(), IndivCompare_Score);
        break;
    case SORTBY_IDNUM:
        if (Z) sort(beginZoo(), endZoo(), IndivCompare_IDNum);
        if (P) sort(beginParents(), endParents(), IndivCompare_IDNum);
        if (C) sort(beginChildren(), endChildren(), IndivCompare_IDNum);
        break;
    case SORTBY_COLORSPACE:
        if (Z) sort(beginZoo(), endZoo(), IndivCompare_ColorSpace);
        if (P) sort(beginParents(), endParents(), IndivCompare_ColorSpace);
        if (C) sort(beginChildren(), endChildren(), IndivCompare_ColorSpace);
        break;
    case SORTBY_RENDERTIME:
        if (Z) sort(beginZoo(), endZoo(), IndivCompare_RenderTime);
        if (P) sort(beginParents(), endParents(), IndivCompare_RenderTime);
        if (C) sort(beginChildren(), endChildren(), IndivCompare_RenderTime);
        break;
    default:
        ASSERT_R(0);
        break;
    }
}

//boost::shared_mutex & Population::M(char* S)
//{
//    // std::cerr << "M(" << S << ")\n";
//    return m_mutex;
//}

void Population::M(char* S)
{
    // std::cerr << "M(" << S << ")\n";
}