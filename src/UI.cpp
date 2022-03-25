// UI.cpp - The user interface

#include "UI.h"

#include "CUDAHelpers.h"
#include "Counters.h"
#include "Evolver.h"
#include "MathStyle.h"
#include "RenderManager.h"

#include "GL/glew.h"

// This needs to come after GLEW
#include "GL/freeglut.h"

#include <sstream>

extern Counters* C;
extern Evolver* Evo;
extern RenderManager* RMan;
extern Population* Pop;
extern UI* GUI;
extern MathStyle* MSEng;
extern Style* SEng;

const int KEY_OFFSET = 1000;

////////////////////////////////////////////////////////
// GLUT Callbacks

void cbglutDisplay(void)
{
    GUI->Display();
}

void cbglutIdle()
{
    GUI->Idle();
}

void cbglutKeyPress(unsigned char key, int x, int y)
{
    GUI->GeneralOps(static_cast<int>(key), x, y);
}

void cbglutMenuStatus(int bob, int x, int y)
{
    GUI->MenuStatus(bob, x, y);
}

void cbglutMotion(int x, int y)
{
    GUI->ClickDragMotion(x, y);
}

void cbglutPassiveMotion(int x, int y)
{
    GUI->PassiveMotion(x, y);
}

void cbglutMouse(int button, int state, int x, int y)
{
    GUI->Mouse(button, state, x, y);
}

void cbglutMouseWheel(int wheel, int direction, int x, int y)
{
    GUI->MouseWheel(wheel, direction, x, y);
}

void cbglutReshape(int w, int h)
{
    GUI->Reshape(w, h);
}

void cbglutSpecialKeyPress(int key, int x, int y)
{
    GUI->GeneralOps(key + KEY_OFFSET, x, y);
}

void cbglutMenuPress(int c)
{
    GUI->GeneralOps(c, -1, -1);
}

////////////////////////////////////////////////////////
// Public Members

UI::UI(int& argc, char** argv)
{
    m_winTitle = "GenArt";
    m_dropperColor = f3Pixel(0.2f, 0.16f, 0.18f);
    m_curIndiv = NULL;
    m_finalIndiv = NULL;
    m_popSortBy = SORTBY__MAX;
    m_lastX = 0;
    m_lastY = 0;
    m_orderingColorMap = 0;
    m_curChan = 1;
    m_saveCounter = 0;
    m_winImgsX = 9;
    m_winImgsY = 7;
    m_winWid = -1;
    m_winHgt = -1;
    m_finalX = -1;
    m_finalY = -1;
    m_zooRowOffset = 0;
    m_amPanning = false;
    m_autoEvolve = false;
    m_winIsUp = false;
    m_finalRenderOnHover = true;

    StartUI(argc, argv);
}

void UI::GetOpenGLVersion(int argc, char** argv)
{
    OpenGLVersion();
}

void UI::StartUI(int& argc, char** argv)
{
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(1, 1);
    glutInitWindowPosition(0, 0);
    glutCreateWindow(m_winTitle.c_str());

    glutDisplayFunc(cbglutDisplay);
    glutIdleFunc(cbglutIdle);
    glutKeyboardFunc(cbglutKeyPress);
    glutMotionFunc(cbglutMotion);
    glutMouseFunc(cbglutMouse);
    glutMouseWheelFunc(cbglutMouseWheel);
    glutPassiveMotionFunc(cbglutPassiveMotion);
    glutReshapeFunc(cbglutReshape);
    glutSpecialFunc(cbglutSpecialKeyPress);

    glutMenuStatusFunc(cbglutMenuStatus);
    int menuID = glutCreateMenu(cbglutMenuPress);
    glutSetMenuFont(menuID, GLUT_BITMAP_HELVETICA_18);

    glutAddMenuEntry("____INDIVIDUAL____", 765);
    glutAddMenuEntry("z: Zoom in", 'z');
    glutAddMenuEntry("Z: Zoom out", 'Z');
    glutAddMenuEntry("H: Center horizontally", 'H');
    glutAddMenuEntry("V: Center vertically", 'V');
    glutAddMenuEntry("p: Print Individual", 'p');
    glutAddMenuEntry("s: Save Image", 's');
    glutAddMenuEntry("TAB: Eyedropper pixel color and ColorMap", '\t');
    glutAddMenuEntry("w: Do final render", 'w');
    glutAddMenuEntry("D: Delete Following Individuals", 'D');
    glutAddMenuEntry("0 or DEL or d: Delete Individual", '0');
    glutAddMenuEntry("1: Set Score...", '1');
    glutAddMenuEntry("9: Set Score", '9');
    glutAddMenuEntry("RIGHT: Animate Individual", GLUT_KEY_RIGHT + KEY_OFFSET);
    glutAddMenuEntry("LEFT: Animate Individual", GLUT_KEY_LEFT + KEY_OFFSET);

    glutAddMenuEntry("__MATH INDIVIDUAL__", 765);
    glutAddMenuEntry("r: Choose Red Channel", 'r');
    glutAddMenuEntry("g: Choose Green Channel", 'g');
    glutAddMenuEntry("b: Choose Blue Channel", 'b');
    glutAddMenuEntry("[: Scale Down Channel", '[');
    glutAddMenuEntry("]: Scale Up Channel", ']');
    glutAddMenuEntry("R: Replicate Channel", 'R');
    glutAddMenuEntry("C: Randomize constants", 'C');
    glutAddMenuEntry("u: Set pixel from eyedropper", 'u');
    glutAddMenuEntry("U: Set ColorMap from eyedropper", 'U');
    glutAddMenuEntry("c: Cycle color space", 'c');
    glutAddMenuEntry("x: Randomize ColorMap 50%", 'x');
    glutAddMenuEntry("X: Random ColorMap", 'X');
    glutAddMenuEntry("v: Reorder ColorMap", 'v');

    glutAddMenuEntry("___________________", 765);
    glutAddMenuEntry("e: Save the World", 'e');
    glutAddMenuEntry("t: Sort by Score", 't');
    glutAddMenuEntry("<: Decrease variability 5%", '<');
    glutAddMenuEntry(">: Decrease variability 5%", '>');
    glutAddMenuEntry("a: Auto Evolve on/off", 'a');
    glutAddMenuEntry("I: Apply eyedropper ColorMap to all", 'i');
    glutAddMenuEntry("I: Apply eyedropper ColorMap to all", 'I');
    glutAddMenuEntry("F: Rescore All to 0.02", 'F');
    glutAddMenuEntry("f: Rescore All by Rank", 'f');
    glutAddMenuEntry("W: Toggle Final render on hover", 'W');
    glutAddMenuEntry("J: Enter final quality", 'J');
    glutAddMenuEntry("T: Enter thumb quality", 'T');
    glutAddMenuEntry("SPACE: Make New Generation", ' ');
    glutAddMenuEntry("DOWN: Scroll Down", GLUT_KEY_DOWN + KEY_OFFSET);
    glutAddMenuEntry("UP: Scroll Up", GLUT_KEY_UP + KEY_OFFSET);
    glutAddMenuEntry("PG DOWN: Scroll Down Fast", GLUT_KEY_PAGE_DOWN + KEY_OFFSET);
    glutAddMenuEntry("PG UP: Scroll Up Fast", GLUT_KEY_PAGE_UP + KEY_OFFSET);
    glutAddMenuEntry("ESC or q: Exit program", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    ////////////////////////////////////////////////////////////
    // OpenGL calls may not be above here

    GLenum ret = glewInit();
    if (ret != GLEW_OK) throw DMcError(reinterpret_cast<const char*>(glewGetErrorString(ret)));

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPixelZoom(1, -1);

    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
    glLineWidth(1);
}

void UI::MainLoop()
{
    SetTitle();
    // glutReshapeWindow(RMan->thWid * m_winImgsX + RMan->finalWid, std::max(RMan->thHgt * m_winImgsY, RMan->finalHgt));
    // glutReshapeWindow(3840, 2066); // For 4K
    glutReshapeWindow(1920, 1017); // For 4K

    glutMainLoop();
}

////////////////////////////////////////////////////////
// Private members

void UI::Display()
{
    // std::cerr << "Display\n";

    int childrenToRequest = m_autoEvolve ? 500 : (m_childRows * m_winImgsX);
    if (VerifyChildren(childrenToRequest)) {
        PostRedisplay(); // Redisplay if evolved new children.
        SetTitle();
    }

    m_winIsUp = true;
    bool needRedisplay = false;

    glViewport(0, m_winHgt - m_winHgt, m_winWid, m_winHgt); // If we have vertical offset bugs, change the first m_winHgt to the full window height.
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glOrtho(0, m_winWid, m_winHgt, 0, -1, 1);

    glClearColor(m_dropperColor.r(), m_dropperColor.g(), m_dropperColor.b(), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Render the Zoo
    for (size_t j = m_zooRowOffset * m_winImgsX; (int)j < (m_zooRowOffset + ZOO_Y(m_winImgsY)) * m_winImgsX && j < Pop->sizeZoo(); j++) {
        if (Pop->getZoo(j)->ThumbImD()->imDone()) {
            int ix = ((int)j % m_winImgsX) * RMan->thWid;
            int iy = (((int)j / m_winImgsX) - m_zooRowOffset + m_childRows + PARENT_ROWS) * RMan->thHgt;
            glRasterPos3i(ix, iy, -1);
            DrawImage(Pop->getZoo(j)->ThumbImD());
        } else {
            needRedisplay = true;
        }
    }

    // Render the parents
    for (size_t j = 0; (int)j < PARENT_ROWS * m_winImgsX && j < Pop->sizeParents(); j++) {
        if (Pop->getParents(j)->ThumbImD()->imDone()) {
            int ix = ((int)j % m_winImgsX) * RMan->thWid;
            int iy = (((int)j / m_winImgsX) + m_childRows) * RMan->thHgt;
            glRasterPos3i(ix, iy, -1);
            DrawImage(Pop->getParents(j)->ThumbImD());
        } else {
            needRedisplay = true;
        }
    }

    // Render the children
    for (size_t j = 0; (int)j < m_childRows * m_winImgsX && j < Pop->sizeChildren(); j++) {
        if (Pop->getChildren(j)->ThumbImD()->imDone()) {
            int ix = ((int)j % m_winImgsX) * RMan->thWid;
            int iy = ((int)j / m_winImgsX) * RMan->thHgt;
            glRasterPos3i(ix, iy, -1);
            DrawImage(Pop->getChildren(j)->ThumbImD());
        } else {
            needRedisplay = true;
        }
    }

    {
        // Render a final image

        // Choose the most recently added finished final image to display
        auto& FRQ = RMan->getFinalRenderQueue();

        auto ind = FRQ.begin();
        for (; ind != FRQ.end(); ind++) {
            if ((*ind)->FinalImD()->imDone()) {
                FRQ.erase(ind + 1, FRQ.end()); // Don't bother rendering older ones than this

                glRasterPos3i(m_finalX, m_finalY, -1);
                m_finalIndiv = *ind;
                DrawImage(m_finalIndiv->FinalImD());
                break;
            }
        }

        if (ind != FRQ.begin()) needRedisplay = true; // If there is one that's not done, rerender later
    }

    if (needRedisplay) PostRedisplay(); // It can't draw them all yet, so need to redraw later.

    MathIndividual::shp mind(std::dynamic_pointer_cast<MathIndividual>(m_curIndiv));
    const int RIBBON_HEIGHT = 20;
    if (mind && (mind->ColorSpace == SPACE_COLMAP || mind->ColorSpace == SPACE_TONEMAP_COLMAP)) DrawCMap(&(mind->CMap), m_finalX, m_finalY - RIBBON_HEIGHT, m_finalX + RMan->finalWid, m_finalY);

    // Display text
    DisplayText(MakeText());

    // Draw some lines
    DrawLine(0, m_childRows * RMan->thHgt + 1, m_finalX, m_childRows * RMan->thHgt, 1, 0);
    DrawLine(0, (m_childRows + PARENT_ROWS) * RMan->thHgt, m_finalX, (m_childRows + PARENT_ROWS) * RMan->thHgt, 1, 0);
    DrawLine(m_finalX, 0, m_finalX, m_winImgsY * RMan->thHgt, 0, 1);
    DrawLine(m_finalX, m_finalY, m_finalX + RMan->finalWid, m_finalY, 1, 0);

    // GL_ASSERT();

    glutSwapBuffers();
}

void UI::Idle()
{
    // getCUDAMemInfo(); // XXX This is a hack.

    size_t zooStartind = m_zooRowOffset * m_winImgsX;
    size_t zooEndind = std::min(static_cast<size_t>((m_zooRowOffset + ZOO_Y(m_winImgsY)) * m_winImgsX), Pop->sizeZoo());
    bool DidWork = RMan->DoVisibleThumbRenderWork(Pop->beginZoo() + zooStartind, Pop->beginZoo() + zooEndind, Evo->Scorer());
    if (DidWork) {
        PostRedisplay();
        // std::cerr << "t";
        return;
    }

    DidWork = RMan->DoFinalRenderQueueWork(false);
    if (DidWork) {
        PostRedisplay();
        // std::cerr << "f";
        return;
    }

    DidWork = RMan->DoHiddenRenderWork(Evo->Scorer());
    if (DidWork) {
        // std::cerr << "h";
        return;
    }

    if (m_autoEvolve) {
        Evo->EvolveGeneration(); // Choose best old children and move to zoo
        DeleteSpanOfChildren(false);
        PostRedisplay();
        return;
    }

    Sleep(1);
}

void UI::MenuStatus(int bob, int x, int y)
{
    // std::cerr << "Menu status = " << bob << " " << x << " " << y << std::endl;
    m_lastX = x;
    m_lastY = y;
}

void UI::PassiveMotion(int x, int y)
{
    SetCurIndiv(x, y);

    if (m_curIndiv == NULL) return; // Not hovering on an individual

    if (m_finalRenderOnHover || m_curIndiv->FinalImD()->imDone()) RMan->PushToFinalRenderQueue(m_curIndiv); // Even if it's already rendered, list it here for display
}

void UI::ClickDragMotion(int x, int y)
{
    // Interactive panning
    const int PAN_THRESH = 3;
    if (m_curIndiv != NULL && (abs(x - m_lastX) > PAN_THRESH || abs(y - m_lastY) > PAN_THRESH)) m_amPanning = true;

    if (m_amPanning) {
        ASSERT_R(m_curIndiv != NULL);
        float pixsz = m_curIndiv->BoxWid / float(AmFinal(x, y) ? RMan->finalWid : RMan->thWid);
        float fdx = (x - m_lastX) * pixsz;
        float fdy = (y - m_lastY) * pixsz;
        m_curIndiv->Pan(m_curIndiv->Xmin - fdx, m_curIndiv->Ymin - fdy);

        m_lastX = x;
        m_lastY = y;
        PostRedisplay();
    }
}

void UI::Mouse(int button, int state, int x, int y)
{
    // std::cerr << "button = " << button << " state = " << state << std::endl;
    size_t idx = SetCurIndiv(x, y);

    if (state == 1) { // Button release
        if (m_amPanning) {
            m_amPanning = false;
        } else if (m_curIndiv != NULL && button == 0) { // Left
            if (idx == -1) return;
            if (AmParent(x, y)) {
                Pop->MoveParentToZoo(idx);
            } else if (AmChild(x, y)) {
                if (m_curIndiv->GetScore() < 0.1f) m_curIndiv->SetScore(0.1f);
                Pop->MoveChildToParent(idx);
            } else {
                Pop->MoveZooToParent(idx);
            }
        }
    }
}

void UI::MouseWheel(int wheel, int direction, int x, int y)
{
    // std::cerr << "wheel=" << wheel << " dir=" << direction << " x=" << x << " y=" << y << std::endl;
    if (wheel == 0)
        if (direction == 1)
            GeneralOps(GLUT_KEY_UP + KEY_OFFSET, x, y);
        else if (direction == -1)
            GeneralOps(GLUT_KEY_DOWN + KEY_OFFSET, x, y);
}

void UI::Reshape(int w, int h)
{
    if (w == 0 || h == 0) return;

    std::cerr << w << "x" << h << '\n';

    // Compute number of thumbnails that fit now
    m_winImgsX = std::max(3, (w - RMan->finalWid) / RMan->thWid);
    m_winImgsY = h / RMan->thHgt;
    m_childRows = std::max(1, std::min((m_winImgsY - PARENT_ROWS) / 2, MAX_CHILD_ROWS));

    m_winWid = w;
    m_winHgt = h;

    m_finalX = RMan->thWid * m_winImgsX; // Corner of the big, final image
    m_finalY = std::max(m_winHgt - RMan->finalHgt, 0);
}

void UI::MathStyleOps(int c, MathIndividual::shp ind)
{
    unsigned int mod = glutGetModifiers();

    switch (c) {
    case 'r': m_curChan = 0; break;
    case 'g': m_curChan = 1; break;
    case 'b': m_curChan = 2; break;
    case 'R': ind->ReplicateChannel(m_curChan); break;
    case '[': ind->ScaleBiasChannel(m_curChan, SEng->getVariability(), 0.0f); break;
    case ']': ind->ScaleBiasChannel(m_curChan, 1.0f + SEng->getVariability(), 0.0f); break;
    case '{': ind->ScaleBiasChannel(m_curChan, 1.0f, -SEng->getVariability()); break;
    case '}': ind->ScaleBiasChannel(m_curChan, 1.0f, SEng->getVariability()); break;
    case 'C': ind->RandConst(SEng->getVariability() * 0.2f); break;
    case 'u': {
        float r;
        f3Pixel cmapVal = ind->EvalSample(m_findivX, m_findivY, r);
        ind->SetColorMapEntry(cmapVal.g(), m_dropperColor);
        break;
    }
    case 'U':
        ind->SetColorMap(m_dropperColorMap);
        ind->SetSpace(SPACE_COLMAP);
        break;
    case 'S':
        std::cerr << "Shuffling variables\n";
        ind->ShuffleVars();
        break;
    case 'c': ind->SetSpace(ColorSpace_t((ind->GetSpace() + 1) % NUM_COLORSPACES)); break;
    case 'X': ind->ColorMapRandomize(); break;
    case 'x': ind->ColorMapMutate(SEng->getVariability()); break;
    case 'v':
        m_orderingColorMap = (m_orderingColorMap + 1) % NUM_REORDERS;
        std::cerr << "Reordering ColorMap as " << ColorMapOrderings[m_orderingColorMap] << std::endl;
        ind->ColorMapReorder(ColorMapOrderings_t(m_orderingColorMap));
        break;
    case '\t': // Does both Individual and MathIndividual stuff.
        m_dropperColorMap = ind->CMap;
        break;
    }
}

void UI::IndividualOps(int c, int x, int y)
{
    size_t idx = SetCurIndiv(x, y);

    if (m_curIndiv == NULL) return;

    MathIndividual::shp mind(std::dynamic_pointer_cast<MathIndividual>(m_curIndiv));
    if (mind) MathStyleOps(c, mind);

    switch (c) {
    case 'z': m_curIndiv->Zoom(1.0f / 1.414f); break;
    case 'Z': m_curIndiv->Zoom(1.414f); break;
    case 'H': m_curIndiv->CenterX(); break;
    case 'V': m_curIndiv->CenterY(); break;
    case 'p': std::cout << m_curIndiv->stringDisplay() << std::endl; break;
    case 's':
        if (m_curIndiv->FinalImD()->imDone()) {
            m_curIndiv->ImSave();
        } else {
            m_curIndiv->requestSave();
            RMan->PushToFinalRenderQueue(m_curIndiv);
        }
        break;
    case '\t': {
        uc4Pixel px = *((AmFinal(x, y) ? m_curIndiv->FinalImD() : m_curIndiv->ThumbImD())->pp(m_indivX, m_indivY));
        m_dropperColor = static_cast<f3Pixel>(px);
    } break;
    case 'w': RMan->PushToFinalRenderQueue(m_curIndiv); break;
    case 'D':
        Pop->DeleteSpanOfIndividuals(idx, 0, AmParent(x, y), AmChild(x, y));
        // std::cerr << "Deleted everything from the selected individual on.\n";
        break;
    case '0':
    case 'd':
    case 127: // Delete
        Pop->DeleteSpanOfIndividuals(idx, 1, AmParent(x, y), AmChild(x, y));
        break;
    }

    int a = c - '0';
    if (a >= 0 && a <= 9) {
        m_curIndiv->SetScore(a * 0.1);
        if (AmChild(x, y)) { Pop->MoveChildToZoo(idx); }
        // std::cerr << "Score for image " << m_curIndiv << " is " << m_curIndiv->GetScore() << std::endl;
    }
}

void UI::GeneralOps(int c, int x, int y)
{
    if (x < 0 || y < 0) {
        x = m_lastX;
        y = m_lastY;
    }

    // std::cerr << "Key=" << char(c) << std::endl;

    // These are not operations on a specific individual.
    switch (c) {
    case 'e': Pop->SavePopulation(Pop->getFNameBase(), SEng); break;
    case 't':
        m_popSortBy = static_cast<SortCriterion>((m_popSortBy + 1) % SORTBY__MAX);
        Pop->Sort(m_popSortBy, true, true, true);
        break;
    case '<':
        SEng->setVariability(SEng->getVariability() - 0.05f);
        std::cerr << "Variability = " << SEng->getVariability() << '\n';
        break;
    case '>':
        SEng->setVariability(SEng->getVariability() + 0.05f);
        std::cerr << "Variability = " << SEng->getVariability() << '\n';
        break;
    case 'a': m_autoEvolve = !m_autoEvolve; break;
    case 'f': {
        // Reset scores to be on 0.1 .. 0.6.
        Pop->Sort(m_popSortBy);
        int sz = (int)Pop->sizeZoo(), i = 0;
        for (auto ind = Pop->beginZoo(); ind != Pop->endZoo(); ind++, i++) (*ind)->SetScore(0.1f + (sz - i - 1) / float(2 * sz));
    } break;
    case 'F': {
        // Reset scores to tiny values
        Pop->Sort(m_popSortBy);
        for (auto ind = Pop->beginZoo(); ind != Pop->endZoo(); ind++) (*ind)->SetScore(0.02f);
        for (auto ind = Pop->beginParents(); ind != Pop->endParents(); ind++) (*ind)->SetScore(0.02f);
    } break;
    case 'W': m_finalRenderOnHover = !m_finalRenderOnHover; break;
    case 'J': RMan->inputQuality(RMan->finalQuality); break;
    case 'T': RMan->inputQuality(RMan->thumbQuality); break;
    case ' ': DeleteSpanOfChildren(true); break;
    case 'I':
    case 'i':
        for (auto in = Pop->beginZoo(); in != Pop->endZoo(); in++) {
            std::dynamic_pointer_cast<MathIndividual>(*in)->SetColorMap(m_dropperColorMap);
            if (c == 'I') std::dynamic_pointer_cast<MathIndividual>(*in)->SetSpace(SPACE_COLMAP);
        }
        for (auto in = Pop->beginParents(); in != Pop->endParents(); in++) {
            std::dynamic_pointer_cast<MathIndividual>(*in)->SetColorMap(m_dropperColorMap);
            if (c == 'I') std::dynamic_pointer_cast<MathIndividual>(*in)->SetSpace(SPACE_COLMAP);
        }
        break;
    case GLUT_KEY_DOWN + KEY_OFFSET: {
        int zooSize = static_cast<int>(Pop->sizeZoo());
        int zooRows = static_cast<int>(ceilf(zooSize / static_cast<float>(m_winImgsX)));
        if (m_zooRowOffset < zooRows - ZOO_Y(m_winImgsY)) m_zooRowOffset++; // It's ZOO_Y(m_winImgsY) because we know there are at least that many rows that can't be offscreen zoo.
    } break;
    case GLUT_KEY_UP + KEY_OFFSET:
        if (m_zooRowOffset >= 1) m_zooRowOffset--;
        break;
    case GLUT_KEY_PAGE_DOWN + KEY_OFFSET: {
        int zooSize = static_cast<int>(Pop->sizeZoo());
        int zooRows = static_cast<int>(ceilf(zooSize / static_cast<float>(m_winImgsX)));
        m_zooRowOffset = std::min(m_zooRowOffset + ZOO_Y(m_winImgsY), zooRows - ZOO_Y(m_winImgsY));
    } break;
    case GLUT_KEY_PAGE_UP + KEY_OFFSET: m_zooRowOffset = std::max(m_zooRowOffset - ZOO_Y(m_winImgsY), 0); break;
    case 27:
    case 'q':
        C->print();
        exit(0);
        break;
    default: IndividualOps(c, x, y); break;
    }

    PassiveMotion(x, y); // Since the individuals may have moved under the mouse
    SetTitle();
}

////////////////////////////////////////////////////////
// Helper functions

void UI::DrawImage(const uc4DImage* Im)
{
    // std::cerr << "DrawImage VBO: " << Im->Vbo() << '\n';
    ASSERT_R(Im->Vbo() > 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Im->Vbo());
    glDrawPixels(Im->w(), Im->h(), GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void UI::DrawLine(int x0, int y0, int x1, int y1, int h, int v)
{
    glBegin(GL_LINES);
    glColor4f(0.7f, 0.7f, 0.7f, 1.0f);
    glVertex2i(x0 + v, y0 + h);
    glVertex2i(x1 + v, y1 + h);
    glColor4f(0.9f, 0.9f, 0.9f, 1.0f);
    glVertex2i(x0, y0);
    glVertex2i(x1, y1);
    glColor4f(0.2f, 0.2f, 0.2f, 1.0f);
    glVertex2i(x0 - v, y0 - h);
    glVertex2i(x1 - v, y1 - h);
    glEnd();
}

void UI::DrawTextRow(const std::string& text, int x, int y, void* font /*= GLUT_BITMAP_8_BY_13*/)
{
    // Save state
    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT);

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    // drop shadow
    glColor3f(0, 0, 0);
    // Shift shadow one pixel to the lower right.
    glRasterPos3i(x + 1, y + 1, -1);

    for (std::string::const_iterator it = text.begin(); it != text.end(); ++it) glutBitmapCharacter(font, *it);

    // main text
    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos3i(x, y, -1);

    for (std::string::const_iterator it = text.begin(); it != text.end(); ++it) glutBitmapCharacter(font, *it);

    // Restore state
    glPopAttrib();
}

void UI::DrawCMap(ColorMap<f3Pixel>* CMap, const int x0, const int y0, const int x1, const int y1)
{
    float step = (x1 - x0) / ((float)CMap->size() - 1.0f);
    float x = x0;

    glBegin(GL_QUAD_STRIP);
    glColor3f((*CMap)[0].r(), (*CMap)[0].g(), (*CMap)[0].b());
    glVertex2f(x, y0);
    glVertex2f(x, y1);
    for (int i = 1; i < CMap->size(); i++) {
        x += step;
        glColor3f((*CMap)[i].r(), (*CMap)[i].g(), (*CMap)[i].b());
        glVertex2f(x, y0);
        glVertex2f(x, y1);
    }
    glEnd();
}

bool UI::VerifyChildren(size_t numWanted)
{
    Evo->ResetScoreStats();

    bool madeSome = false;
    while (Pop->sizeChildren() < numWanted) {
        madeSome = true;
        // Autoevolving uses Zoo as parents, rather than Parents.
        // Pop->insertChildren(Evo->Generate(m_autoEvolve ? Pop->beginZoo() : Pop->beginParents(), m_autoEvolve ? Pop->endZoo() : Pop->endParents(), m_autoEvolve)); // Generate a new child.
        Pop->insertChildren(Evo->Generate(Pop->beginParents(), Pop->endParents(), m_autoEvolve)); // Generate a new child.
    }
    return madeSome;
}

void UI::DeleteSpanOfChildren(bool was_space)
{
    if (Pop->sizeChildren() > (m_childRows - 1) * m_winImgsX) {
        if (was_space)
            Pop->DeleteSpanOfIndividuals(0, Pop->sizeChildren() - (m_childRows - 1) * m_winImgsX, false, true); // Delete the top row of children
        else
            Pop->DeleteSpanOfIndividuals(0, Pop->sizeChildren(), false, true); // Delete all children
    }

    int save_thresh = was_space ? SAVE_EVERY_N_SPACE : (SAVE_EVERY_N_SPACE * SAVE_EVERY_N_SPACE * SAVE_EVERY_N_SPACE);
    if (++m_saveCounter >= save_thresh) {
        Pop->SavePopulation(Pop->getFNameBase(), SEng);
        m_saveCounter = 0;
    }
}

void UI::OpenGLVersion()
{
    std::cerr << glGetString(GL_EXTENSIONS) << std::endl;
    std::cerr << glGetString(GL_VENDOR) << std::endl;
    std::cerr << glGetString(GL_RENDERER) << std::endl;
    std::cerr << glGetString(GL_VERSION) << std::endl;
    // std::cerr << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
}

void UI::PostRedisplay()
{
    // std::cerr << "PostRedisplay\n";
    if (m_winIsUp) { glutPostRedisplay(); }
}

std::string UI::MakeText()
{
    std::string str;
    if (m_curIndiv) {
        MathIndividual::shp mind = std::dynamic_pointer_cast<MathIndividual>(m_curIndiv);
        str = mind->stringDisplay(m_findivX, m_findivY);
    }

    return str;
}

void UI::DisplayText(std::string str)
{
    // Break the text into lines and display them
    int xbase = m_finalX + 10;
    int ybase = 4;

    std::string s;
    for (const char* c = str.c_str(); *c; c++) {
        if (*c != '\n')
            s += *c;
        else {
            DrawTextRow(s, xbase, ybase += 24, GLUT_BITMAP_TIMES_ROMAN_24);
            s.clear();
        }
    }
}

bool UI::AmZoo(const int x, const int y)
{
    int row = y / RMan->thHgt;
    return (row >= m_childRows + PARENT_ROWS) && x < m_finalX;
}

bool UI::AmParent(const int x, const int y)
{
    int row = y / RMan->thHgt;
    return (row >= m_childRows && row < m_childRows + PARENT_ROWS) && x < m_finalX;
}

bool UI::AmChild(const int x, const int y)
{
    int row = y / RMan->thHgt;
    return (row < m_childRows) && x < m_finalX;
}

bool UI::AmFinal(const int x, const int y)
{
    return x >= m_finalX && y >= m_finalY && x <= m_finalX + RMan->finalWid && y <= m_finalY + RMan->finalHgt;
}

int UI::ZOO_Y(const int ImgsY)
{
    return ImgsY - PARENT_ROWS - m_childRows;
}

size_t UI::SetCurIndiv(const int x, const int y)
{
    size_t idx = -1;
    m_lastX = x;
    m_lastY = y;

    if (AmFinal(x, y)) {
        m_curIndiv = m_finalIndiv;

        if (m_curIndiv != NULL) {
            m_indivX = m_lastX - m_finalX;
            m_indivY = m_lastY - m_finalY;
            float pixsz = m_curIndiv->BoxWid / float(RMan->finalWid);
            m_findivX = m_curIndiv->Xmin + m_indivX * pixsz;
            m_findivY = m_curIndiv->Ymin + m_indivY * pixsz;
        }
    } else {
        if (AmZoo(x, y)) {
            size_t i = (m_zooRowOffset + (y / RMan->thHgt) - (m_childRows + PARENT_ROWS)) * m_winImgsX + (x / RMan->thWid);
            idx = (i < 0 || i >= Pop->sizeZoo()) ? -1 : i;
            m_curIndiv = (idx != -1) ? Pop->getZoo(i) : NULL;
        } else if (AmParent(x, y)) {
            size_t i = (y / RMan->thHgt - m_childRows) * m_winImgsX + (x / RMan->thWid);
            idx = (i < 0 || i >= Pop->sizeParents()) ? -1 : i;
            m_curIndiv = (idx != -1) ? Pop->getParents(i) : NULL;
        } else if (AmChild(x, y)) {
            size_t i = (y / RMan->thHgt) * m_winImgsX + (x / RMan->thWid);
            idx = (i < 0 || i >= Pop->sizeChildren()) ? -1 : i;
            m_curIndiv = (idx != -1) ? Pop->getChildren(i) : NULL;
        } else
            m_curIndiv = NULL;

        if (m_curIndiv != NULL) {
            int xb = (m_lastX / RMan->thWid) * RMan->thWid;
            int yb = (m_lastY / RMan->thHgt) * RMan->thHgt;
            m_indivX = m_lastX - xb;
            m_indivY = m_lastY - yb;
            float pixsz = m_curIndiv->BoxWid / float(RMan->thWid);
            m_findivX = m_curIndiv->Xmin + m_indivX * pixsz;
            m_findivY = m_curIndiv->Ymin + m_indivY * pixsz;
        }
    }

    PostRedisplay();

    return idx;
}

void UI::SetTitle()
{
    const char* SortByNames[] = {"Score", "IDNumber", "ColorSpace", "RenderTime", "None"};
    const char* ChannelNames[] = {"Red", "Green", "Blue"};

    int zooOfs = m_zooRowOffset * m_winImgsX;
    std::ostringstream ost;
    ost << m_winTitle << " Sort by: " << SortByNames[m_popSortBy] << "   " << (m_finalRenderOnHover ? " Final Rendering    " : "") << (m_autoEvolve ? " autoEvolve    " : "") << ChannelNames[m_curChan]
        << "    "
        << "Variability: " << SEng->getVariability() << "    Row: " << zooOfs << " Total Created: " << Evo->IndivsCreated();
    glutSetWindowTitle(ost.str().c_str());
}
