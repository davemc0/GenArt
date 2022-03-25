#include "AutoScorer.h"
#include "Counters.h"
#include "Evolver.h"
#include "ExprTools.h"
#include "RenderManager.h"

#include "Image/ColorMap.h"
#include "Image/Quant.h"
#include "Math/MiscMath.h"
#include "Math/Random.h"

extern Evolver* Evo;
extern Population* Pop;
extern RenderManager* RMan;
extern Style* SEng;

/////////////////////////////////////////////////////////////////////////////////////////////////
// My notes on artistic evolution

// Automatic metrics:
// Blue sky, green ground
// Fade to blue near horizon
// Follows the rule of thirds: lines or features 1/3 of way in, such as horizon, not in the center
// Home corner has heavier stuff - darker? Higher contrast?

// Something about segments, but I'm not sure what.
// Have a certain percent of pixels be smooth gradient, a percent be edge, a percent be high frequency (hair, ripples, rocks, etc.)

// Take the FFT of real images and see what they have and simply match the FFT.
// Do this for image classes: old master paintings, nature photos, modern art, etc.

// Try to discern a foreground and background. If possible, reward it.

// How to detect movement in a picture?

// Horizontal lines, commonly found in landscape photography, gives the impression of calm, tranquility, and space.
// An image filled with strong vertical lines tends to have the impression of height, and grandeur.
// Tightly angled convergent lines give a dynamic, lively, and active effect to the image.
// Viewpoint is very important when dealing with lines particularly in photography, because every different perspective elicits a different response to the photograph.
// Dark colors on the bottom and light on the top gives a feeling of solidity.
// Horizontal lines on the bottom and vertical on the top gives a feeling of solidity,.

// Try a Hough transform? Use this to determine a vanishing point

// Rule of odds

// What does the eye like?
// A: Acute elements - If the whole picture is blurry and there is one sharp object.
// B: Bright elements – If the whole picture is dark except one bright object.
// C: Colorful elements – If the whole picture is black and white or monochrome and there is one color saturated object.

// Emphasize foreground object by making sure its color contrasts with the background

// Pattern: Need a way to detect patterns

// Higher saturation objects appear closer than faded ones.
// Blurred objects appear further due to DOF.
// A warm color can appear to project and cool color can appear to recede, other things being equal. 
// A light tone (value) can appear to project and dark tone can appear to recede. 

// Color ColorMaps: Find out what math the online color pickers use

// Stuff I like: Glow, solid regions, smooth curves, sharp edges, high frequency, looks like nature, looks sci fi, hard to paint, hard to conceive

/////////////////////////////////////////////////////////////////////////////////////////////////

extern Counters* C;

Evolver::Evolver(AutoScorer* Scorer_) : m_Scorer(Scorer_), m_indivsCreated(0), m_maxZooSize(50), m_childrenPerGen(2)
{
}

Evolver::~Evolver()
{
	if (m_Scorer)
		delete m_Scorer;
}

// Figure variability = 0.2 for autoEvolve
// Figure variability = 0.8 for interactive
Individual::shp Evolver::Generate(Population::iterator& beginParent, Population::iterator& endParent, bool autoEvolve)
{
	C->inc("Generate");
	float v = SEng->getVariability();

	Individual::shp Aa = ChooseParent(beginParent, endParent, autoEvolve);
	Individual::shp Bb = chance(v) ? ChooseParent(beginParent, endParent, autoEvolve) : NULL; // Usually choose a second one randomly.
	if (Aa == Bb)
		Bb = NULL;

	int Generation = 1 + std::max(Aa ? Aa->Generation : 0, Bb ? Bb->Generation : 0);

	if (Aa)
		C->inc("Generate.A");
	if (Bb)
		C->inc("Generate.B");

	float XMin, YMin, BoxWid;
	if (Aa == NULL || chance(v * 0.4f)) {
		C->inc("Generate.Viewport");
		if (Aa)
			C->inc("Generate.Viewport.OneAa");
		// Get funky with the viewport
		XMin = RandCoord();
		YMin = RandCoord();
		BoxWid = (XMin < -0.7f && YMin < -0.7f) ? 2.0f : 1.0f;
	}
	else {
		if (Bb == NULL || chance(1, 2)) {

			C->inc("Generate.Viewport.Aa");
			XMin = Aa->Xmin;
			YMin = Aa->Ymin;
			BoxWid = Aa->BoxWid;
		}
		else {
			C->inc("Generate.Viewport.Bb");
			XMin = Bb->Xmin;
			YMin = Bb->Ymin;
			BoxWid = Bb->BoxWid;
		}
	}

	m_indivsCreated++;

	return SEng->BreedIndividual(-1, Generation, Aa, Bb, XMin, YMin, BoxWid);
}

void Evolver::EvolveGeneration()
{
	C->inc("EvolveGeneration");
	// We know that all children are rendered and scored because this function is called from the *bottom* of the Idle function.

	// Move the best parent into the Zoo to preserve it
	//if (Pop->sizeParents() > 1) // && (Pop->sizeZoo() == 0 || (*Pop->beginParents())->GetScore() > (*Pop->beginZoo())->GetScore()))
		// Move the second best parent to the zoo if it's a new leader. Recall that it has already reproduced.
	//	Pop->MoveParentToZoo(1);

	// Delete all Parents
	// Except current best
	Pop->DeleteSpanOfIndividuals(1, 0, true);

	// ASSERT_R(Pop->sizeParents() <= 1);
	
	// Move all Children to Parents
	for (size_t i = Pop->sizeChildren(); i; --i) {
        Pop->MoveChildToParent(i - 1);
    }

	ASSERT_R(Pop->sizeChildren() == 0);
	ASSERT_R(Pop->sizeParents() > 0);

	// Re-sort Parents and Zoo
    Pop->Sort(SORTBY_SCORE, true, true, false);

	// Generation score is best new parent's score
	float genScore = (*Pop->beginParents())->GetScore();

#if 0
	// Remove duplicates; identical ones will have same score, and will be adjacent.
    for (size_t i = 0; i < Pop->sizeParents() - 1; i++) {
        Individual::shp I = Pop->getParents(i);
        Individual::shp J = Pop->getParents(i + 1);

        if (I->Score == J->Score && I->equal(*J)) {
            std::cerr << "Got an equal one: " << i << std::endl;
            Pop->DeleteSpanOfIndividuals(i--);
        }
    }
#endif

    // Delete excess Zoo
    Pop->DeleteSpanOfIndividuals(m_maxZooSize, 0);

    std::cerr << "IndivsCreated = " << m_indivsCreated << " Max score = " << Pop->getParents(0)->GetScore()
        << " GenScore = " << genScore << " Variability = " << SEng->getVariability() << std::endl;
}

float scoreFunc(const float s, const float mn, const float mx)
{
	float sn = (s - mn) / (mx - mn);

	return sn * sn * sn; //  powf(sn, 3.0f);
}

void Evolver::computeScoreStats(Population::iterator& beginParent, Population::iterator& endParent)
{
	// Sum up the scores
	m_totalScore = 0;
	m_maxScore = (*beginParent)->GetScore();
	m_minScore = (*(endParent - 1))->GetScore();
	ASSERT_R(endParent - beginParent > 0);

	for (auto ind = beginParent; ind != endParent; ind++)
		m_totalScore += scoreFunc((*ind)->Score, m_minScore, m_maxScore);
	std::cerr << "total = " << m_totalScore << " max = " << m_maxScore << " min = " << m_minScore << '\n';
}

// Pick a random Parent based on their scores
Individual::shp Evolver::ChooseParent(Population::iterator& beginParent, Population::iterator& endParent, bool useScores)
{
	int n = endParent - beginParent;
	if (n == 0)
		return NULL;
	
	if (useScores) {
		// This is expensive. Don't do it once per Indiv created.
		if (m_maxScore == 0)
			computeScoreStats(beginParent, endParent);

        float r = DRandf() * m_totalScore;
        float totalScore = 0;
		for (auto ind = beginParent; ind != endParent; ind++) {
			float sc = scoreFunc((*ind)->Score, m_minScore, m_maxScore);
			// std::cerr << (*ind)->Score << "=>" << sc << ' ';
			totalScore += sc;
			if (r <= totalScore) {
				// std::cerr << "\n\n";
				return *ind;
			}
		}

        return NULL;
    }
    else {
        // In interactive mode I really want all of the parents to be selected equally, not based on score.
        return *(beginParent + randn(n));
    }
}

float Evolver::RandCoord()
{
    float XMin = 0.0f;
    switch (randn(6)) {
    case 0:
        XMin = 0.0f;
        break;
    case 1:
        XMin = -0.3333f;
        break;
    case 2:
        XMin = -0.5f;
        break;
    case 3:
        XMin = -0.6666f;
        break;
    case 4:
        XMin = -1.0f;
        break;
    }
    return XMin;
}
