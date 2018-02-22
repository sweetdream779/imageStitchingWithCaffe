#ifndef TEST_SEGMENTATION_HPP_
#define TEST_SEGMENTATION_HPP_

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include "boost/algorithm/string.hpp"

#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <opencv2/flann.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <time.h>

using std::string;
using std::vector;
using caffe::Net;
using namespace cv;
using namespace cv::xfeatures2d;

const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box

namespace enetCaffe {

class Segmentator {
 public:
  static Segmentator *Get();
  static Segmentator *Get(const string &model_path, const string &trained_file);

  cv::Mat Predict(const cv::Mat& img, string LUT_file);

 private:
  Segmentator(const string& model_file,
             const string& trained_file);

  void SetMean(const string& mean_file);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  cv::Mat Visualization(cv::Mat prediction_map, string LUT_file);

  static Segmentator *caffe_mobile_;
  static string model_path_;
  static string trained_file_;

  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;

};

}// namespace
class My_Stitcher
{
public:
    static My_Stitcher *Get();
    cv::Mat vis(const cv::Mat& image1, const cv::Mat& image2);
    cv::Mat stitch(const cv::Mat& image1, const cv::Mat& image2);
private:
    cv::Ptr<cv::FastFeatureDetector> detector;
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    static My_Stitcher *stitcher_;
    void detect_and_compute(const cv::Mat& frame,std::vector<cv::KeyPoint> *kps,cv::Mat* descs);
    My_Stitcher(){
        detector=cv::FastFeatureDetector::create();
        //detector->setThreshold(100);
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    }
};

#endif