#include "caffe/caffe.hpp"

#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include <cblas.h>


#include "segmentator.hpp"

#ifdef __cplusplus
extern "C" {
#endif

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

using std::string;
using std::vector;
using enetCaffe::Segmentator;

int getTimeSec() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (int)now.tv_sec;
}

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
Java_com_example_irina_test_CaffeMobile_setNumThreads(JNIEnv *env,
                                                             jobject thiz,
                                                             jint numThreads) {
  int num_threads = numThreads;
  openblas_set_num_threads(num_threads);
}

JNIEXPORT jint JNICALL Java_com_example_irina_test_CaffeMobile_loadModel(
    JNIEnv *env, jobject thiz, jstring modelPath, jstring weightsPath) {
  Segmentator::Get(jstring2string(env, modelPath),
                   jstring2string(env, weightsPath));
  return 0;
}

JNIEXPORT void JNICALL
Java_com_example_irina_test_CaffeMobile_predictImage(
    JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height, jstring LUT_file, jlong addrResult) {
  Segmentator *caffe_mobile = Segmentator::Get();
  cv::Mat* pMat=(cv::Mat*)addrResult;
  cv::Mat result =
      caffe_mobile->Predict(getImage(env, buf, width, height), jstring2string(env, LUT_file));
  result.copyTo(*pMat);
  //__android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The value of n is %d", 3);
}


JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env = NULL;
  jint result = -1;

  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    LOG(FATAL) << "GetEnv failed!";
    return result;
  }

  FLAGS_redirecttologcat = true;
  FLAGS_android_logcat_tag = "caffe_jni";

  return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL
Java_com_example_irina_test_MyStitcher_stitchImages(
    JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height, jbyteArray buf2, jint width2, jint height2, jlong addrResult) {
  My_Stitcher *stitcher = My_Stitcher::Get();
  cv::Mat* pMat=(cv::Mat*)addrResult;
  cv::Mat result =
      stitcher->stitch(getImage(env, buf, width, height), getImage(env, buf2, width2, height2));
  result.copyTo(*pMat);
  __android_log_print(ANDROID_LOG_VERBOSE, "Stitching", "DONE");
}


#ifdef __cplusplus
}
#endif