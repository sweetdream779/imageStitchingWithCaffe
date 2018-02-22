//
// Created by irina on 20.02.18.
//
#ifndef TEST_MYSTITCHER_H
#define TEST_MYSTITCHER_H


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
#include <opencv2/highgui.hpp>

const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box

namespace stitching {
    class MyStitcher {
    public:
        static MyStitcher *Get();

        cv::Mat vis(const cv::Mat &image1, const cv::Mat &image2);

        cv::Mat stitch(const cv::Mat &image1, const cv::Mat &image2);

    private:
        cv::Ptr<cv::FastFeatureDetector> detector;
        cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor;
        cv::Ptr<cv::DescriptorMatcher> matcher;
        static MyStitcher *stitcher_;

        void
        detect_and_compute(const cv::Mat &frame, std::vector<cv::KeyPoint> *kps, cv::Mat *descs);

        MyStitcher() {
            detector = cv::FastFeatureDetector::create();
            //detector->setThreshold(100);
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
            matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        }
    };
}



#endif //TEST_MYSTITCHER_H
#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::string;
using std::vector;

namespace stitching {

    MyStitcher *MyStitcher::stitcher_ = 0;

    MyStitcher *MyStitcher::Get() {
        if (!stitcher_ ) {
            stitcher_ = new MyStitcher();
        }
        return stitcher_;
    }


    void MyStitcher::detect_and_compute(const cv::Mat& frame,std::vector<cv::KeyPoint> *kp,cv::Mat* desc){

        detector->detect(frame,*kp);
        if(!kp->empty())
            extractor->compute(frame, *kp, *desc);
        std::cout<<"kp: "<<kp->size();
    }

    cv::Mat MyStitcher::vis(const cv::Mat& image2, const cv::Mat& image1){
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

    cv::Mat resizeImg(const cv::Mat& im){
        cv::Mat  img = im.clone();
        int newWidth = 600;
        double scale = (double)newWidth/(double)im.cols;
        int newHeight = (double)im.rows * (double)scale;
        std::cout<<scale<<std::endl;
        cv::resize(img, img, cv::Size(), scale, scale);
        return img;
    }

    cv::Mat MyStitcher::stitch(const cv::Mat& img2, const cv::Mat& img1){
        cv::Mat image1 = resizeImg(img1);
        cv::Mat image2 = resizeImg(img2);
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
            cv::Mat res(image1.rows, image1.cols, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::putText(res,"Stitching failed",cv::Point2f(image1.rows/3,image1.cols/3),1,3,cv::Scalar(0,0,255),2);
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


#ifdef __cplusplus
extern "C" {
#endif
using stitching::MyStitcher;

string jstring2string(JNIEnv *env, jstring jstr) {
    const char *cstr = env->GetStringUTFChars(jstr, 0);
    string str(cstr);
    env->ReleaseStringUTFChars(jstr, cstr);
    return str;
}

string bytes2string(JNIEnv *env, jbyteArray buf) {
    jbyte *ptr = env->GetByteArrayElements(buf, 0);
    string s((char *)ptr, env->GetArrayLength(buf));
    env->ReleaseByteArrayElements(buf, ptr, 0);
    return s;
}

cv::Mat imgbuf2mat(JNIEnv *env, jbyteArray buf, int width, int height) {
    jbyte *ptr = env->GetByteArrayElements(buf, 0);
    cv::Mat img(height + height / 2, width, CV_8UC1, (unsigned char *)ptr);
    cv::cvtColor(img, img, CV_YUV2RGBA_NV21);
    env->ReleaseByteArrayElements(buf, ptr, 0);
    return img;
}

cv::Mat getImage(JNIEnv *env, jbyteArray buf, int width, int height) {
    return (width == 0 && height == 0) ? cv::imread(bytes2string(env, buf), -1)
                                       : imgbuf2mat(env, buf, width, height);
}


JNIEXPORT void JNICALL
Java_com_example_irina_test_Stitching_stitchImages(
        JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height, jbyteArray buf2, jint width2, jint height2, jlong addrResult) {
    MyStitcher *stitcher = MyStitcher::Get();
    cv::Mat* pMat=(cv::Mat*)addrResult;
    cv::Mat result =
            stitcher->stitch(getImage(env, buf, width, height), getImage(env, buf2, width2, height2));
    result.copyTo(*pMat);
    __android_log_print(ANDROID_LOG_VERBOSE, "Stitching", "DONE");
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved)
{
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return -1;
    }

    // Get jclass with env->FindClass.
    // Register methods with env->RegisterNatives.

    return JNI_VERSION_1_6;
}

#ifdef __cplusplus
}
#endif