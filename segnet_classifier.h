#ifndef SEGNET_CLASSIFIER_H
#define SEGNET_CLASSIFIER_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace  cv;
using namespace caffe;
using namespace boost;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file);


  void Predict(const cv::Mat& img, string LUT_file);
  void Predict(const cv::Mat& img, Mat &output);

 private:
  void SetMean(const string& mean_file);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  void Visualization(Blob<float>* output_layer, string LUT_file);
  void Convert2Image(Blob<float>* output_layer, Mat &output);

 private:
  boost::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;

};

#endif // SEGNET_CLASSIFIER_H
