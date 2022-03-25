#pragma once

#include "Style.h"

class Individual;
class uc4DImage;

// Contains all the data that should persist across CUDA render tasks
class CUDARender {
public:
    CUDARender(const int deviceId_);

    virtual void Render(Individual* ind_, uc4DImage* Im, const int w_, const int h_, const Quality_t Q_) = 0;

protected:
    int m_deviceId;
};
