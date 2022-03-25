#include "Individual.h"

#include "AutoScorer.h"
#include "CUDARender.h"
#include "RenderManager.h"

#include <Util/Assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>

extern RenderManager* RMan;
extern CUDARender* Rend;

// An IndivID is based on the time in seconds of the first call to this function plus a serial number s_offset.
// This is not thread safe because of the static variables. Make sure you use an exclusive lock.
// WARNING: This can give a duplicate ID on a subsequent app run if the first run generated more than one image per second.
int CreateIndivID()
{
    static int s_offset = 0;

    if (s_offset == 0) {
        time_t epochsecs = time(NULL);
        time_t base = time_t(1221400000);
        time_t nowsecs = epochsecs - base;
        s_offset = int(nowsecs);
    }

    return s_offset++;
}

Individual::Individual(float Score_, int IDNum_, int Generation_, int ParentA_, int ParentB_, float XMin_, float YMin_, float BoxWid_)
{
    init(Score_, IDNum_, Generation_, ParentA_, ParentB_, XMin_, YMin_, BoxWid_);
}

Individual::Individual(const Individual& In)
{
    ASSERT_R(0);
}

Individual::~Individual()
{
    delete m_thumbIm;
    delete m_finalIm;

#if 0
    // Turn these on just for debugging.
    m_thumbIm = NULL;
    m_finalIm = NULL;
    Score = Xmin = Ymin = BoxWid = -999.999f;
#endif
}

void Individual::init(float Score_, int IDNum_, int Generation_, int ParentA_, int ParentB_, float XMin_, float YMin_, float BoxWid_)
{
    Score = Score_;
    IDNum = IDNum_;
    Generation = Generation_;
    ParentA = ParentA_;
    ParentB = ParentB_;
    Xmin = XMin_;
    Ymin = YMin_;
    BoxWid = BoxWid_;

    if (Generation < 1) Generation = 1;

    if (IDNum < 1) IDNum = CreateIndivID();

    m_thumbIm = NULL; // Clear these pointers since they have garbage.
    m_finalIm = NULL;
    ImClear();
    m_requestedSave = false;
}

void Individual::RenderLaunch(bool isFinal, AutoScorer* Scorer /*= NULL*/)
{
    if (isFinal) {
        FinalImD()->setLaunched(true);
        Rend->Render(this, FinalImD(), RMan->finalWid, RMan->finalHgt, RMan->finalQuality);
        if (m_requestedSave) {
            ImSave(); // Blocks until render finished, then saves
            m_requestedSave = false;
        }
    } else {
        ThumbImD()->setLaunched(true);
        Rend->Render(this, ThumbImD(), RMan->thWid, RMan->thHgt, RMan->thumbQuality);
        if (Scorer != NULL) { Scorer->ComputeScore(this); }
    }
}

void Individual::ImClear()
{
    if (m_thumbIm) delete m_thumbIm;
    m_thumbIm = new uc4DImage;

    if (m_finalIm) delete m_finalIm;
    m_finalIm = new uc4DImage;
}

void Individual::ImSave()
{
    FinalImD()->renderBlock();

    char imFName[1000];
    sprintf(imFName, "G%09d.%s", GetIDNum(), RMan->imageSaveFormat.c_str());
    std::cerr << "Saving image to " << imFName << std::endl;
    FinalImD()->pp(); // Copy the data to the host
    FinalIm()->Save(imFName);
}

// Zoom in or out on the image
void Individual::Zoom(float z)
{
    float Xctr = Xmin + BoxWid * 0.5f;
    float Yctr = Ymin + BoxWid * 0.5f;
    BoxWid *= z;
    Xmin = Xctr - BoxWid * 0.5f;
    Ymin = Yctr - BoxWid * 0.5f;
    ImClear();
}

// Pan the image to new corner location
void Individual::Pan(float xm, float ym)
{
    Xmin = xm;
    Ymin = ym;
    ImClear();
}

void Individual::CenterX()
{
    Xmin = -BoxWid * 0.5f;
    ImClear();
}

void Individual::CenterY()
{
    Ymin = -(BoxWid * ThumbIm()->h() / ThumbIm()->w()) * 0.5f;
    ImClear();
}

uc4Image* Individual::ThumbIm()
{
    return m_thumbIm;
}

uc4Image* Individual::FinalIm()
{
    return m_finalIm;
}

uc4DImage* Individual::ThumbImD()
{
    uc4DImage* thumbImD = dynamic_cast<uc4DImage*>(m_thumbIm);
    if (thumbImD == NULL) { ASSERT_R("Wasn't a uc4DImage"); }
    return thumbImD;
}

uc4DImage* Individual::FinalImD()
{
    uc4DImage* finalImD = dynamic_cast<uc4DImage*>(m_finalIm);
    if (finalImD == NULL) { ASSERT_R("Wasn't a uc4DImage"); }
    return finalImD;
}

std::string Individual::stringSave()
{
    std::ostringstream outStream;

    outStream << GetScore() << " " << GetIDNum() << " " << GetGeneration() << " " << GetParentA() << " " << GetParentB() << " " << Xmin << " " << Ymin << " " << BoxWid;

    return outStream.str();
}

std::string Individual::stringDisplay(const float x /*= -1.0f*/, const float y /*= -1.0f*/)
{
    std::ostringstream out;
    FloatFmt(out, 2);
    out << "IDNum=" << IDNum << " Score=" << Score << " Corner=" << Xmin << "," << Ymin << " BoxWid=" << BoxWid << " Times=" << ThumbImD()->renderTime() << ", " << FinalImD()->renderTime()
        << " Gen=" << Generation << " Parents=" << ParentA << "," << ParentB;
    return out.str();
}

bool Individual::equal(const Individual& p) const
{
    return Xmin == p.Xmin && Ymin == p.Ymin && BoxWid == p.BoxWid;
}

bool Individual::less(const Individual& p) const
{
    ASSERT_R(0);
}

void Individual::assignIDNum()
{
    IDNum = CreateIndivID();
}
