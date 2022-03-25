#pragma once

#include "Image/tImage.h"

class uc4DImage : public uc4Image
{
public:
    uc4DImage();

    ~uc4DImage();

    // Overrides of uc4Image members
    void SetSize(const int wid_ = 0, const int hgt_ = 0, const bool doFill = false, const bool doDel = true);

    // Copies image from device to host and returns a const pointer to this pixel on host.
    const uc4Pixel* pp(const int i=0);

    // Returns a pointer to this pixel.
    const uc4Pixel* pp(const int x, const int y)
    {
        ASSERT_D(x>=0 && x<w());
        ASSERT_D(y>=0 && y<h());
        return pp(ind(x,y));
    }

    unsigned int Vbo() const { return m_vbo; }
    size_t Pitch() const { return m_pitch; }
    void* map();
    void unmap();
    void renderTimerStart();
    void renderTimerStop();
    bool renderTimerFinished() const;
    void renderBlock(); // Blocks until rendering is finished
    float renderTime(); // Returns render time, if image is finished, or -1 if not

    bool imDone() const; // True if rendering is finished
    bool needsLaunch() const;
    void setLaunched(bool val);

private:
    struct cudaGraphicsResource* m_vboCUDA;
    struct CUevent_st* m_start;
    struct CUevent_st* m_stop;
    size_t m_pitch;
    unsigned int m_vbo;
    float m_renderTime;
    bool m_spoofed_uc4Image;
    bool m_launched; // True when a render has been launched
};

extern void UIYield();
extern void UILock(int win);
extern void UIUnlock(int win);
