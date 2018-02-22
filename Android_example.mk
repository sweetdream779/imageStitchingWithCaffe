LOCAL_PATH := $(call my-dir)
PROJECT_ROOT:= $(call my-dir)/../../../../..

include $(CLEAR_VARS)
LOCAL_MODULE    := lib_caffe
LOCAL_SRC_FILES := libcaffe.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE    := lib_caffe_jni
LOCAL_SRC_FILES := libcaffe_jni.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
OPENCV_INSTALL_MODULES:=on

include /home/irina/caffe-android-lib/android_lib/opencv/sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := libopencv_ndk
LOCAL_CFLAGS    := -Werror -Wno-write-strings -std=c++11
LOCAL_SRC_FILES := MyLib.cpp \
				   mystither.cpp \
				   mystitcher_jni.cpp

LOCAL_SHARED_LIBRARIES := lib_caffe lib_caffe_jni
LOCAL_LDLIBS    := -llog -landroid
include $(BUILD_SHARED_LIBRARY)
