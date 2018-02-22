#include "stitcher.hpp"

namespace stitching {

Stitcher *Stitcher::Get() {
  if (!stitcher_ ) {
    stitcher_ = new Stitcher();
  }
  return stitcher_;
}


void Stitcher::detect_and_compute(const cv::Mat& frame,std::vector<cv::KeyPoint> *kp,cv::Mat* desc){
    
    detector->detect(frame,*kp);
    if(!kp->empty())
        extractor->compute(frame, *kp, *desc);
    std::cout<<"kp: "<<kp->size();
}

cv::Mat Stitcher::vis(const cv::Mat& image2, const cv::Mat& image1){
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

cv::Mat Stitcher::stitch(const cv::Mat& image2, const cv::Mat& image1){
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
}//namespace
/*
using stitching::Stitcher;

int main(int argc, char **argv){
    cv::Mat image1,image2,res;

	image1 = cv::imread(argv[1]);
    image2 = cv::imread(argv[2]); 

    Stitcher *stitcher = Stitcher::Get();
    res=stitcher->stitch(image1,image2);

    return 0;


}*/