#include "segnet_classifier.h"

Classifier::Classifier(const string& model_file,
                       const string& trained_file) {


  Caffe::set_mode(Caffe::GPU);

  //加载网络
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}


void Classifier::Predict(const cv::Mat& img, string LUT_file)
{
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  //前向运算前调整各层为正确维度
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  //存储输出层数据
  Blob<float>* output_layer = net_->output_blobs()[0];

  Visualization(output_layer, LUT_file);
}

void Classifier::Predict(const cv::Mat& img, Mat &output)
{
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  //前向运算前调整各层为正确维度
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  //存储输出层数据
  Blob<float>* output_layer = net_->output_blobs()[0];
  Convert2Image(output_layer, output);
}

void Classifier::Visualization(Blob<float>* output_layer, string LUT_file) {

  std::cout << "output_blob(n,c,h,w) = " << output_layer->num() << ", " << output_layer->channels() << ", "
              << output_layer->height() << ", " << output_layer->width() << std::endl;

  cv::Mat merged_output_image = cv::Mat(output_layer->height(), 
                                        output_layer->width(), 
                                        CV_32F, 
                                        const_cast<float *>(output_layer->cpu_data())
                                        );
  //merged_output_image = merged_output_image/255.0;

  merged_output_image.convertTo(merged_output_image, CV_8U);
  cv::cvtColor(merged_output_image.clone(), merged_output_image, CV_GRAY2BGR);
  cv::Mat label_colours = cv::imread(LUT_file,1);
  cv::Mat output_image;
  LUT(merged_output_image, label_colours, output_image);

  //cv::imshow( "Display window", output_image);
  //cv::waitKey(0);
}

void Classifier::Convert2Image(Blob<float>* output_layer, Mat &output)
{
  std::cout << "output_blob(n,c,h,w) = " << output_layer->num() << ", " << output_layer->channels() << ", "
              << output_layer->height() << ", " << output_layer->width() << std::endl;

  cv::Mat merged_output_image = cv::Mat(output_layer->height(), output_layer->width(), CV_32F, const_cast<float *>(output_layer->cpu_data()));
  //merged_output_image = merged_output_image/255.0;

  //merged_output_image.convertTo(output, CV_8U, 255);

  merged_output_image.convertTo(output, CV_8UC1, 255);

  //cv::cvtColor(merged_output_image.clone(), merged_output_image, CV_GRAY2BGR);
  //cv::Mat label_colours = cv::imread(LUT_file,1);
  //cv::Mat output_image;
  //LUT(merged_output_image, label_colours, output_image);
  //cv::imshow( "Display window", output_image);
  //cv::waitKey(0);
}


void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

//预处理
void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  //分离通道作为输入层
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
