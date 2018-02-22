//
// Created by irina on 23.10.17.
//
#include "com_examle_test_MyNDK.h"
#include <jni.h>

JNIEXPORT jstring JNICALL Java_com_example_irina_test_MainActivity_getMyString
        (JNIEnv * env, jobject){
    return env->NewStringUTF("This is MyLib.");
}


