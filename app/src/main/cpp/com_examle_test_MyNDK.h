//
// Created by irina on 23.10.17.
//
#include <jni.h>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

#ifndef TEST_COM_EXAMLE_TEST_MYNDK_H
#define TEST_COM_EXAMLE_TEST_MYNDK_H

extern "C"{

JNIEXPORT jstring JNICALL Java_com_example_irina_test_MainActivity_getMyString
        (JNIEnv *, jobject);


}
#endif //TEST_COM_EXAMLE_TEST_MYNDK_H
