#include "AutoScorer.h"

#include <Image/ImageAlgorithms.h>
#include <Image/Quant.h>

float speedScore(float ms)
{
    float tscore = 1.0f / (5.0f + ms); // Fast = 0.1; slow = 0.0

    return tscore;
}

void RandomAutoScorer::ComputeScore(Individual* Ind)
{
    Ind->ThumbImD()->renderBlock();
    float t = Ind->ThumbImD()->renderTime();
    // Ind->ThumbImD()->pp(); // Copy the data to the host

    float tscore = speedScore(t);
    Ind->SetScore(tscore);
}

void ColorfulnessAutoScorer::ComputeScore(Individual* Ind)
{
    Ind->ThumbImD()->renderBlock();
    float t = Ind->ThumbImD()->renderTime();
    Ind->ThumbImD()->pp(); // Copy the data to the host

    uc3Image uc3I = *Ind->ThumbIm();
    QuantParams QP;
    QP.maxColorPalette = 0;
    QP.makeArtisticPalette = true;
    Quantizer<uc3Pixel, unsigned char> Qnt(uc3I.pp(), uc3I.size(), false, QP);
    float score = 10.0f * Qnt.GetHistogramCount(); // % of color buckets used (4096 buckets)

    float tscore = speedScore(t);
    // std::cerr << score << " " << t << " " << tscore << '\n';
    score += tscore;

    Ind->SetScore(score);
}

ImageSimilarityAutoScorer::ImageSimilarityAutoScorer(uc4Image* Img) : TargetImg(Img) {}

void ImageSimilarityAutoScorer::init(Individual* Ind)
{
    if (!TargetImg) {
        // Nothing provided by user so make an image that's blue on top and green on the bottom
        TargetImg = new uc4Image(Ind->ThumbIm()->w(), Ind->ThumbIm()->h(), uc4Pixel(0, 255, 0, 255));

        for (int i = 0; i < TargetImg->size() / 2; i++) (*TargetImg)[i] = uc4Pixel(0, 0, 255, 255);
    }

    if (Ind->ThumbIm()->w() != TargetImg->w() || Ind->ThumbIm()->h() != TargetImg->h()) {
        uc4Image* newTarg = new uc4Image;
        Resample(*newTarg, *TargetImg, Ind->ThumbIm()->w(), Ind->ThumbIm()->h());
        // std::cerr << newTarg->w() << 'x' << newTarg->h() << '\n';
        // std::cerr << Ind->ThumbIm()->w() << 'x' << Ind->ThumbIm()->h() << '\n';

        delete TargetImg;
        TargetImg = newTarg;
    }
}

ImageSimilarityAutoScorer::~ImageSimilarityAutoScorer()
{
    delete TargetImg;
}

int pixDiff(const uc4Pixel& p0, const uc4Pixel& p1)
{
    int errcount;

    errcount = dmcm::Sqr(int(p0.r()) - int(p1.r()));
    errcount += dmcm::Sqr(int(p0.g()) - int(p1.g()));
    errcount += dmcm::Sqr(int(p0.b()) - int(p1.b()));

    return errcount;
}

// Score the image based on its similarity to the other image
void ImageSimilarityAutoScorer::ComputeScore(Individual* Ind)
{
    Ind->ThumbImD()->renderBlock();
    float t = Ind->ThumbImD()->renderTime();
    Ind->ThumbImD()->pp(); // Copy the data to the host

    if (!TargetImg) init(Ind);

    if (Ind->ThumbImD()->w() != TargetImg->w() || Ind->ThumbImD()->h() != TargetImg->h()) // Reshape or synthesize a target if there's not one defined of the right size
        init(Ind);

    ASSERT_R(Ind->ThumbImD()->size() == TargetImg->size());

    double errcount = 0;
    for (int i = 0; i < Ind->ThumbImD()->size(); i++) {
        int err = pixDiff((*Ind->ThumbImD())[i], (*TargetImg)[i]);
        errcount += float(err);
    }

    float errval = sqrtf(errcount);
    float scale = sqrtf(255.0f * 255.0f * 3.0f * float(Ind->ThumbImD()->size()));
    float score = 1.0f - errval / scale;
    float tscore = speedScore(t);
    score += tscore;
    if (t > 9.0f) score = 0;

    // score = powf(score, 6);

    // std::cerr << errcount << " " << errval << " " << score << " " << t << " " << tscore << '\n';

    Ind->SetScore(score);
}
