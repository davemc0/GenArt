#pragma once

#include "MathIndividual.h"
#include "Population.h"

#include <deque>

class UI {
public:
    UI(int& argc, char** argv);

    ~UI() {}

    void GetOpenGLVersion(int argc, char** argv);

    void MainLoop();

    ////////////////////////////////////////////////////////
    // GUI Callback internals

    void Display();
    void Idle();
    void MenuStatus(int bob, int x, int y);
    void ClickDragMotion(int x, int y);
    void PassiveMotion(int x, int y);
    void Mouse(int button, int state, int x, int y);
    void MouseWheel(int wheel, int direction, int x, int y);
    void Reshape(int w, int h);

    void IndividualOps(int c, int x, int y); // Operations that can be done on an individual

    void MathStyleOps(int c, MathIndividual::shp ind); // Operations specific to the Math Style

    void GeneralOps(int c, int x, int y); // Operations not specific to one individual

private:
    static const int MAX_CHILD_ROWS = 3;      // Max rows for child window
    static const int PARENT_ROWS = 1;         // How many rows for parent window
    static const int SAVE_EVERY_N_SPACE = 10; // After pressing SPACE this many times, save the population

    std::string m_winTitle;
    f3Pixel m_dropperColor;              // Color taken from a pixel of one individual to assign to another
    ColorMap<f3Pixel> m_dropperColorMap; // ColorMap taken from one individual to assign to another
    Individual::shp m_curIndiv;          // Pointer to the individual being hovered on
    Individual::shp m_finalIndiv;        // Pointer to the individual in the big, final image currently displayed
    SortCriterion m_popSortBy;
    int m_curChan;              // Which channel is being operated on (r=0 g=1 b=2)
    int m_childRows;            // How many rows for child window
    int m_lastX, m_lastY;       // Position of mouse pointer at last event
    int m_orderingColorMap;     // Cycles through the ways of reordering a ColorMap
    int m_saveCounter;          // How many times SPACE has been pressed since last time we saved the population
    int m_winImgsX, m_winImgsY; // Number of images wide and high the main window is
    int m_winWid, m_winHgt;     // Dimensions of in-use part of main window
    int m_finalX, m_finalY;     // Position of big, final image
    int m_zooRowOffset;         // Number of zoo rows off-screen above the upper corner of the zoo window
    int m_indivX, m_indivY;     // Mouse coords in pixel space
    float m_findivX, m_findivY; // Mouse coords in image space
    bool m_amPanning;
    bool m_autoEvolve;
    bool m_winIsUp;
    bool m_finalRenderOnHover;

    ////////////////////////////////////////////////////////
    // Helper functions

    void StartUI(int& argc, char** argv);

    void DrawImage(const uc4DImage* Im);

    void DrawLine(int x0, int y0, int x1, int y1, int h, int v);

    void DrawTextRow(const std::string& text, int x, int y, void* font);

    void DrawCMap(ColorMap<f3Pixel>* CMap, const int x0, const int y0, const int x1, const int y1);

    bool VerifyChildren(size_t numWanted); // Create all missing children; doesn't render the images

    void DeleteSpanOfChildren(bool from_start); // When you press space

    void OpenGLVersion();

    void PostRedisplay(); // Only works to call this from the UI thread.

    std::string MakeText();

    void SetTitle();

    void DisplayText(std::string str);

    bool AmZoo(const int x, const int y);

    bool AmParent(const int x, const int y);

    bool AmChild(const int x, const int y);

    bool AmFinal(const int x, const int y);

    int ZOO_Y(const int ImgsY);

    size_t SetCurIndiv(const int x, const int y); // Set m_curIndiv based on event
};
