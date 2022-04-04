// This is the genetic art main program.
// Copyright David K. McAllister, 1998 - 2008, 2022.

// TODO:
// Bug: Pressing space doesn't redo the final render when hovering on a child
// Bug: Window layout is wonky with non-square sizes
// Color: Fix number of colors in a ColorMap to a constant (8?), but be able to parse and remap any size for manual tweaking
// Color: Improve color levels // What did I mean by this?
// Color: Remove invisible ColorMap entries by scaling the Exprs. This prevents junk DNA and increases resolution of specifying colors
// Evo: Evolve an image so its histogram matches a ColorMap
// Evo: Fix Evolution
// Evo: Get a database of good color palettes from the internet
// Evo: When breeding a color mapped child of a non-color mapped parent, use palette generator to compute parent's color map
// Evo: When making new random indiv, copy a colormap from a random Zoo individual
// Expr: ?: operator is ternary. Try (A>0 ? A : B)
// Expr: Output an expression as a .dot file to visualize
// Expr: min and max operators
// Expr: tex operator. How to handle multiple output channels?
// Opt: Automatic way to discover that a subexpression is useless, not just const: Abs([1,2]) => [1,2] ; Clamp([0,1]) => [0,1].
// Opt: Exprs aren't getting put in canonical order if the count would increase. Need a mode to do just safe opts.
// Opt: Halton sequence for NumOpt
// Opt: Replace subexpression with linear gradient, not just const, that approximates a more complex expression.
// Opt: Test subexpression replacement on whole equation, not just subexpression.
// Opt: Use sIval for IFS when appropriate.
// UI: A button to aggressively round off constants in this individual // Can't remember why. - Artistic intent?
// UI: Eye dropper to sample a color and click to paint it in a color map
// UI: Hover and interact with big image
// UI: Save backup files to temp dir
// UI: Sometimes clicking on an image to move it doesn't work. It's shortly after pressing space, but the children are done rendering.
//
// REJECTED:
// Expr: Improved Perlin Noise operator. No; too complex on GPU
// Expr: New variable: theta. No; don't want to add an atan2 per pixel.
// Opt: tan acos A => / sqrt - 1 sqr A A etc. It's not simpler, and there are two A subexprs.
// Opt: Sometimes do the next opt pass based on the current, sometimes based on the best, sometimes original Expr. No; idea is that currrent expr is
// sufficiently mutable.

#include "AutoScorer.h"
#include "Counters.h"
#include "Evolver.h"
#include "Math/Random.h"
#include "MathStyle.h"
#include "MathStyleCUDARender.h"
#include "Population.h"
#include "RenderManager.h"
#include "Style.h"
#include "Test.h"
#include "UI.h"

#include <Util/Timer.h>
#include <iostream>

Counters* C = NULL;
Evolver* Evo = NULL;
MathStyle* MSEng = NULL;
CUDARender* Rend = NULL;
MathStyleCUDARender* MSRend = NULL;
Population* Pop = NULL;
RenderManager* RMan = NULL;
Style* SEng = NULL;
UI* GUI = NULL;

static void Usage(const char* message = NULL, const bool Exit = true)
{
    if (message) std::cerr << "\nERROR: " << message << std::endl;

    std::cerr << "Program options:\n";
    std::cerr << "-colormapimg <im.jpg> <N> Make an individual whose ColorMap is derived from the given image and is N colors\n";
    std::cerr << "-colormapsonly            Load only ColorMaps from the NEXT file\n";
    std::cerr << "-rendersaveall <N>        Render and save all individuals starting with the Nth, then exit\n";
    std::cerr << "-rendersave <N>           Render and save Nth individual or individual with IDNum N, then exit\n";
    std::cerr << "-fmt <jpg|png|gif|tif>    The output format of images to save (currently " << RMan->imageSaveFormat << ")\n";
    std::cerr << "-gpuinfo                  Print info about OpenGL version and CUDA device\n";
    std::cerr << "-thqual <min_samp>        Rendering quality of thumbnails, etc. (currently " << RMan->thumbQuality << ")\n";
    std::cerr << "-qual <min_samp>          Rendering quality of output images (currently " << RMan->finalQuality << ")\n";
    std::cerr << "-thsize <wid> <hgt|asp>   Width and height of thumbnail images (" << RMan->thWid << "x" << RMan->thHgt << ")\n";
    std::cerr << "-size <wid> <hgt|asp>     Width and height of output images (" << RMan->finalWid << "x" << RMan->finalHgt << ")\n";
    std::cerr << "-evolveimg <im.jpg>       Specify an image to evolve to be like\n";
    std::cerr << "-noopt                    Don't optimize expressions\n";
    std::cerr << "-test                     Run weird tests\n";
    std::cerr << "-testex fname.txt         Run expression tests with input list of expressions\n";

    std::cerr << "\nAll GLUT arguments must come after all app. arguments.\n\n";

    if (Exit) exit(1);
}

static void Args(int& argc, char** argv)
{
    bool onlyColorMaps = false;
    int CUDADevice = 0;

    C = new Counters();
    // Evo = new Evolver(new ColorfulnessAutoScorer());
    // Evo = new Evolver(new ImageSimilarityAutoScorer(nullptr));
    // Evo = new Evolver(new RandomAutoScorer());
    Evo = new Evolver(NULL);
    Pop = new Population();
    RMan = new RenderManager();
    MSEng = new MathStyle(); // When more styles exist, fix this.
    SEng = MSEng;
    GUI = new UI(argc, argv);

    for (int i = 1; i < argc; i++) {
        std::string starg(argv[i]);

        if (starg == "-h" || starg == "-help") {
            Usage();
        } else if (starg == "-rendersave") {
            size_t idx = atoi(argv[i + 1]);
            if (idx > 10000) {
                int IDNum = (int)idx; // They specified an IDNum, so search for the index
                for (idx = 0; idx < Pop->sizeZoo(); idx++) {
                    if ((Pop->beginZoo() + idx)->get()->GetIDNum() == IDNum) break;
                }
            }
            if (idx < 0 || idx > Pop->sizeZoo()) Usage("Must specify a population file before -rendersaveimg and the index must be valid.");
            std::cerr << "Rendering at index " << idx << '\n';
            Pop->getZoo(idx)->requestSave();
            RMan->PushToFinalRenderQueue(Pop->getZoo(idx));
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-rendersaveall") {
            size_t idx = atoi(argv[i + 1]);
            if (idx < 0 || idx > Pop->sizeZoo()) Usage("Must specify a population file before -rendersaveall and the index must be valid.");
            RMan->PushAllToFinalRenderQueue(idx);
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-anim") {
            size_t idx = atoi(argv[i + 1]);
            if (idx < 0 || idx > Pop->sizeZoo()) Usage("Must specify a population file before -anim and the index must be valid.");
            RMan->PushAnimation(Pop->getZoo(idx));
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-evolveimg") {
            if (Evo) delete Evo;
            Evo = new Evolver(new ImageSimilarityAutoScorer(new uc4Image(argv[i + 1])));
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-seed") {
            unsigned int seed = atoi(argv[i + 1]);
            unsigned int s = SRand(seed);
            std::cerr << "Using Seed = " << s << '\n';
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-device") {
            CUDADevice = atoi(argv[i + 1]);
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-test") {
            Test();
            RemoveArgs(argc, argv, i);
        } else if (starg == "-testex") {
            TestExpressions(argv[i + 1]);
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-thsize") {
            RMan->thWid = atoi(argv[i + 1]);
            float a = atof(argv[i + 2]);
            if (a < 3.0f)
                RMan->thHgt = (int)(RMan->thWid / a);
            else {
                RMan->thHgt = atoi(argv[i + 2]);
                a = RMan->thWid / (float)RMan->thHgt;
            }
            RMan->finalHgt = (int)(RMan->finalWid / a);

            Pop->ClearImages();
            RemoveArgs(argc, argv, i, 3);
        } else if (starg == "-size") {
            RMan->finalWid = atoi(argv[i + 1]);
            float a = atof(argv[i + 2]);
            if (a < 3.0f)
                RMan->finalHgt = (int)(RMan->finalWid / a);
            else {
                RMan->finalHgt = atoi(argv[i + 2]);
                a = RMan->finalWid / (float)RMan->finalHgt;
            }
            RMan->thHgt = (int)(RMan->thWid / a);

            Pop->ClearImages();
            RemoveArgs(argc, argv, i, 3);
        } else if (starg == "-gpuinfo") {
            GUI->GetOpenGLVersion(argc, argv);
            getCUDADeviceInfo();
            exit(0);
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-fmt") {
            RMan->imageSaveFormat = argv[i + 1];
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-qual") {
            Quality_t Q;
            Q.MinSamples = atof(argv[i + 1]);
            RMan->setQuality(RMan->finalQuality, Q);
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-thqual") {
            Quality_t Q;
            Q.MinSamples = atof(argv[i + 1]);
            RMan->setQuality(RMan->thumbQuality, Q);
            RemoveArgs(argc, argv, i, 2);
        } else if (starg == "-noopt") {
            MSEng->setOptimize(false);
            RemoveArgs(argc, argv, i);
        } else if (starg == "-colormapimg") { // Makes an indiv and loads it like a gnx file
            MSEng->ImageColorMapToIndiv(argv[i + 1], atoi(argv[i + 2]));
            RemoveArgs(argc, argv, i, 3);
        } else if (starg == "-colormapsonly") {
            onlyColorMaps = true;
            RemoveArgs(argc, argv, i, 1);
        } else if (argv[i][0] == '-') {
            Usage("Unknown option.");
        } else {
            // Read a population file on the command line

            // Remove suffix
            char tmpFName[1000];
            sprintf(tmpFName, "%s", argv[i]);
            char* dot = strrchr(tmpFName, '.');
            *dot = '\0';

#ifdef WIN32
            // Change forward slashes to backslashes
            char* N = tmpFName;
            while (*N) {
                if (*N == '/') *N = '\\';
                N++;
            }
#endif
            // Set the global filename each time. Thus, the results are saved to the last loaded file.
            Pop->setFNameBase(tmpFName);

            MSEng->OnlyColorMaps(onlyColorMaps);
            MSEng->LoadPopulation(Pop->getFNameBase() + SEng->getPopFileSuffix());
            CopyFile(Pop->getFNameBase() + SEng->getPopFileSuffix(), Pop->getFNameBase() + "_old" + SEng->getPopFileSuffix());
        }
    }

    MSRend = new MathStyleCUDARender(CUDADevice);
    Rend = MSRend;

    if (!RMan->DoFinalRenderQueueWork(true)) { // If doing requested rendering, don't do mainloop.
        GUI->MainLoop();
    }

    finishCUDA();
}

int main(int argc, char** argv)
{
    try {
        unsigned int s = SRand();
        std::cerr << "Seed = " << s << '\n';

        Args(argc, argv);
        std::cerr << "Bye\n";
    }
    catch (DMcError& Er) {
        std::cerr << "DMcError: " << Er.Er << std::endl;
        std::cerr.flush();
        throw Er;
    }
    catch (...) {
        std::cerr << "Exception caught. Bye.\n";
        std::cerr.flush();
        throw;
    }

    return 0;
}
