/*
 *  This script visualize the semantic segmentation for your input image.
 */
#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <opencv2/flann.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "segmentator.hpp"
//#include <chrono> //Just for time measurement. This library requires compiler and library support for the ISO C++ 2011 standard. This support is currently experimental in Caffe, and must be enabled with the -std=c++11 or -std=gnu++11 compiler options.

using std::clock;
using std::clock_t;
using std::string;
using std::vector;
using namespace cv;
using namespace xfeatures2d;
using namespace caffe;

namespace enetCaffe {

Segmentator *Segmentator::caffe_mobile_ = 0;
string Segmentator::model_path_ = "";
string Segmentator::trained_file_ = "";

Segmentator *Segmentator::Get() {
  CHECK(caffe_mobile_);
  return caffe_mobile_;
}

Segmentator *Segmentator::Get(const string &model_path,
                              const string &trained_file) {
  if (!caffe_mobile_ || model_path != model_path_ ||
      trained_file != trained_file_) {
    caffe_mobile_ = new Segmentator(model_path, trained_file);
    model_path_ = model_path;
    trained_file_ = trained_file;
  }
  return caffe_mobile_;
}

Segmentator::Segmentator(const string& model_file, const string& trained_file) {


  Caffe::set_mode(Caffe::CPU);

  /* Load the network. */
  clock_t t_start = clock();
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  clock_t t_end = clock();
  LOG(INFO) << "Loading time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC
            << " ms.";

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = Size(input_layer->width(), input_layer->height());
}

Mat Segmentator::Predict(const Mat& img, string LUT_file) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);


  struct timeval time;
  gettimeofday(&time, NULL); // Start Time
  long totalTime = (time.tv_sec * 1000) + (time.tv_usec / 1000);
  //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //Just for time measurement

  net_->Forward();

  gettimeofday(&time, NULL);  //END-TIME
  totalTime = (((time.tv_sec * 1000) + (time.tv_usec / 1000)) - totalTime);
  std::cout << "Processing time = " << totalTime << " ms" << std::endl;

  //std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  //std::cout << "Processing time = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << " sec" <<std::endl; //Just for time measurement


  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];

  int width = output_layer->width();
  int height = output_layer->height(); 
  int channels = output_layer->channels();
  int num = output_layer->num();

  std::cout << "output_blob(n,c,h,w) = " << num << ", " << channels << ", "
			  << height << ", " << width << std::endl;

  // compute argmax
  Mat class_each_row (channels, width*height, CV_32FC1, const_cast<float *>(output_layer->cpu_data()));
  class_each_row = class_each_row.t(); // transpose to make each row with all probabilities
  Point maxId;    // point [x,y] values for index of max
  double maxValue;    // the holy max value itself
  Mat prediction_map(height, width, CV_8UC1);
  for (int i=0;i<class_each_row.rows;i++){
      minMaxLoc(class_each_row.row(i),0,&maxValue,0,&maxId);  
      prediction_map.at<uchar>(i) = maxId.x;     
  }

  return Visualization(prediction_map, LUT_file);
}


Mat Segmentator::Visualization(Mat prediction_map, string LUT_file) {

  cvtColor(prediction_map.clone(), prediction_map, CV_GRAY2BGR);
  Mat label_colours = imread(LUT_file,1);
  cvtColor(label_colours, label_colours, CV_RGB2BGR);
  Mat output_image;
  LUT(prediction_map, label_colours, output_image);

  //imshow( "Display window", output_image);
  //waitKey(0);
  return output_image;
}


/* Wrap the input layer of the network in separate Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Segmentator::WrapInputLayer(std::vector<Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Segmentator::Preprocess(const Mat& img, std::vector<Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cvtColor(img, sample, COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cvtColor(img, sample, COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cvtColor(img, sample, COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cvtColor(img, sample, COLOR_GRAY2BGR);
  else
    sample = img;

  Mat sample_resized;
  if (sample.size() != input_geometry_)
    resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the Mat
   * objects in input_channels. */
  split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}




}//namespace


/*int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " \ndeploy.prototxt \nnetwork.caffemodel"
              << " \nimg.jpg" << " \ncityscapes19.png (for example: /ENet/scripts/cityscapes19.png)" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2]; //for visualization


  Segmentator *caffe_mobile =
      Segmentator::Get(model_file, trained_file);

  string file = argv[3];
  string LUT_file = argv[4];

  std::cout << "---------- Semantic Segmentation for "
            << file << " ----------" << std::endl;

  Mat img = imread(file, 1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  
  Mat output_image=caffe_mobile->Predict(img, LUT_file);
  imshow( "Display window", output_image);
  waitKey(0);
}*/

My_Stitcher *My_Stitcher::stitcher_ = 0;
My_Stitcher *My_Stitcher::Get() {
  if (!stitcher_ ) {
    stitcher_ = new My_Stitcher();
  }
  return stitcher_;
}


void My_Stitcher::detect_and_compute(const cv::Mat& frame,std::vector<cv::KeyPoint> *kp,cv::Mat* desc){
    
    detector->detect(frame,*kp);
    if(!kp->empty())
        extractor->compute(frame, *kp, *desc);
    std::cout<<"kp: "<<kp->size();
}

cv::Mat My_Stitcher::vis(const cv::Mat& image2, const cv::Mat& image1){
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    cv::Rect2d rect_for_searching;

    detect_and_compute(image1,&kp1,&desc1);
    detect_and_compute(image2,&kp2,&desc2);

    std::vector< std::vector<cv::DMatch> > matches;
    std::vector<cv::Point2f> matched1, matched2;
    std::vector<cv::KeyPoint> matched1_, matched2_;
    matcher->knnMatch(desc1, desc2, matches, 2);
    for(unsigned i = 0; i < matches.size(); i++) {
        //0.8: David Lowe’s ratio test for false-positive match pruning
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(cv::Point2f(kp1[matches[i][0].queryIdx].pt.x,kp1[matches[i][0].queryIdx].pt.y));
            matched2.push_back(cv::Point2f(kp2[matches[i][0].trainIdx].pt.x,kp2[matches[i][0].trainIdx].pt.y));
            matched1_.push_back(kp1[matches[i][0].queryIdx]);
            matched2_.push_back(kp2[matches[i][0].trainIdx]);
        }
    }

    cv::Mat inlier_mask, homography;
    if(matched1.size() >= 4) {
        homography = cv::findHomography(matched1, matched2,
                                    CV_RANSAC, ransac_thresh, inlier_mask);
    }
    if(matched1.size() < 4 || homography.empty()) {
        cv::Mat vis;
        hconcat(image1, image2, vis);
        return vis;
    }

    //save visualzation of matches
    std::vector<cv::KeyPoint> inliers1, inliers2;
    std::vector<cv::DMatch> inlier_matches;
    for(unsigned i = 0; i < matched1.size(); i++) {
        //only process the match if the keypoint was successfully matched
        if(inlier_mask.at<uchar>(i)) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1_[i]);
            inliers2.push_back(matched2_[i]);
            inlier_matches.push_back(cv::DMatch(new_i, new_i, 0));
        }
    }
    cv::Mat vis;
    cv::drawMatches(image1, inliers1, image2, inliers2,
                inlier_matches, vis,
                cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));

    
    return vis;
}

cv::Mat My_Stitcher::stitch(const cv::Mat& image2, const cv::Mat& image1){
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    cv::Rect2d rect_for_searching;

    detect_and_compute(image1,&kp1,&desc1);
    detect_and_compute(image2,&kp2,&desc2);

    std::vector< std::vector<cv::DMatch> > matches;
    std::vector<cv::Point2f> matched1, matched2;
    matcher->knnMatch(desc1, desc2, matches, 2); //дольше всех, другие матчинги????
    std::cout<<"matches: "<<matches.size()<<std::endl;
    for(unsigned i = 0; i < matches.size(); i++) {
        //0.8: David Lowe’s ratio test for false-positive match pruning
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(cv::Point2f(kp1[matches[i][0].queryIdx].pt.x,kp1[matches[i][0].queryIdx].pt.y));
            matched2.push_back(cv::Point2f(kp2[matches[i][0].trainIdx].pt.x,kp2[matches[i][0].trainIdx].pt.y));
        }
    }

    cv::Mat inlier_mask, homography;
    if(matched1.size() >= 4) {
        homography = cv::findHomography(matched1, matched2,
                                    CV_RANSAC, ransac_thresh, inlier_mask);//CV_RANSAC
    }
    if(matched1.size() < 4 || homography.empty()) {
        cv::Mat res;
        return res;
    }

    //stitch
    cv::Mat res;
    cv::warpPerspective(image1,res,homography,cv::Size(image2.cols + image1.cols, image1.rows));

    cv::Mat roi1(res, cv::Rect(0, 0,  image2.cols, image2.rows));
    image2.copyTo(roi1);
    return res;
}