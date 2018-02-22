#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "stitcher.hpp"
using std::string;
using std::vector;
using stitching::Stitcher;

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
Java_com_example_irina_test_MyStitcher_stitchImages(
    JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height, jbyteArray buf2, jint width2, jint height2, jlong addrResult) {
  Stitcher *stitcher = Stitcher::Get();
  cv::Mat* pMat=(cv::Mat*)addrResult;
  cv::Mat result =
      stitcher->stitch(getImage(env, buf, width, height), getImage(env, buf2, width2, height2));
  result.copyTo(*pMat);
  __android_log_print(ANDROID_LOG_VERBOSE, "Stitching", "DONE");
}