/*
 *
 *  Created on: Jan 21, 2017
 *      Author: Timo Sämann
 *
 *  The basis for the creation of this script was the classification.cpp example (caffe/examples/cpp_classification/classification.cpp)
 *
 *  This script visualize the semantic segmentation for your input image.
 *
 *  To compile this script you can use a IDE like Eclipse. To include Caffe and OpenCV in Eclipse please refer to
 *  http://tzutalin.blogspot.de/2015/05/caffe-on-ubuntu-eclipse-cc.html
 *  and http://rodrigoberriel.com/2014/10/using-opencv-3-0-0-with-eclipse/ , respectively
 *
 *
 */

#include "CommonDef.h"
#define USE_OPENCV 1

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
//#include <chrono> //Just for time measurement
#include "CommonFunction.h"
#include "CFilePath.h"
#include "image_segmentation.h"

#ifdef USE_OPENCV

using std::string;
using namespace cv;

int main(int argc, char** argv)
{
  ::google::InitGoogleLogging(argv[0]);

    //string model_file   = argv[1];
    string model_file = "../segnet_deploy.prototxt";
  //string trained_file = argv[2]; //for visualization
    string trained_file = "../Inference-2015/test_weights_iter_210000.caffemodel";
    string inDir = "/home/quad/LJH/ICDAR/ICDAR2015_ch4_training_images";
    string outDir = "/home/quad/LJH/ICDAR/2015_train_predict";


    std::list<string> lstFiles;
    CommonFunction::TraverseDir(inDir.c_str(), lstFiles, GET_FILE);
    //lstFiles.push_back("/home/quad/LJH/ICDAR/ICDAR2013_Test_Task12_Images/img_187.jpg");
    if (lstFiles.empty())
        return 1;

    Classifier classifier(model_file, trained_file); 
  //classifier.Predict(img, LUT_file);
    std::list<string>::const_iterator iter = lstFiles.begin();
    for (; iter != lstFiles.end(); iter++)
    {
        CFilePath fp(iter->c_str());
        string ext =  CommonFunction::StringToUpper(fp.GetFileExtention().c_str());
        if (ext == "PNG" || ext == "TIF" || ext == "TIFF" || ext == "JPEG"  || ext == "JPG" )
        {
            printf("Do image segmentation for the image of %s.\n", iter->c_str());
            string outfile = outDir + "/" + fp.GetFileNameNoExt() + ".png";
            cv::Mat img = cv::imread(iter->c_str(), IMREAD_UNCHANGED);

            //cv::Mat prediction;
            //classifier.Predict(img, prediction);
            //imwrite(outfile, prediction);

            cv::Mat output = cv::Mat(img.size(), CV_8UC1, Scalar(0));
            //image_segmentation::process_image_with_overlap(classifier, img, output);
            image_segmentation::process_image_with_multiscale(classifier, img, output);
            imwrite(outfile, output);
        }
    }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV


