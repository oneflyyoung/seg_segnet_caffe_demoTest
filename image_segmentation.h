#ifndef IMAGE_SEGMENTATION_H
#define IMAGE_SEGMENTATION_H

#include "segnet_classifier.h"

class image_segmentation
{
public:
    static void process_image(Classifier &classifier,  Mat &img, Mat &output);
    static void process_image_with_overlap(Classifier &classifier,  Mat &img, Mat &output);

    static void process_image_with_multiscale(Classifier &classifier, const Mat &img, Mat &output);
    static void process_image_with_low_resolution(Classifier &classifier, const Mat &img, Mat &output);
};

#endif // IMAGE_SEGMENTATION_H
