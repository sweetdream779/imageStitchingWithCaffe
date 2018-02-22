#ifndef IMAGESTITCHING_HPP_
#define IMAGESTITCHING_HPP_
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

const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box

class Stitcher
{
public:
    static Stitcher *Get();
    cv::Mat vis(const cv::Mat& image1, const cv::Mat& image2);
    cv::Mat stitch(const cv::Mat& image1, const cv::Mat& image2);
private:
    cv::Ptr<cv::FastFeatureDetector> detector;
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    static Stitcher *stitcher_;
    void detect_and_compute(const cv::Mat& frame,std::vector<cv::KeyPoint> *kps,cv::Mat* descs);
    Stitcher(){
        detector=cv::FastFeatureDetector::create();
        //detector->setThreshold(100);
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    }
};
#endif