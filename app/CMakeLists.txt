# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

include_directories(/home/irina/caffe-android-lib/android_lib/opencv/sdk/native/jni/include)

#add_library( MyLib SHARED src/main/cpp/MyLib.cpp )

############################################################################
#add_library(lib_opencv SHARED IMPORTED)

#set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION
 #                   /home/irina/Desktop/AndroidStudioProjects/Test/app/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so)
set(OpenCV_DIR "/home/irina/caffe-android-lib/android_lib/opencv/sdk/native/jni")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11

###############################################################################
#add_library(lib_caffe SHARED "src/main/jniLibs/arm64-v8a/libcaffe.so")

#set_target_properties(lib_caffe PROPERTIES IMPORTED_LOCATION
#                    /home/irina/caffe-android-lib/android_lib/caffe/lib/libcaffe.so)

#add_library(lib_caffe_jni SHARED "src/main/jniLibs/arm64-v8a/libcaffe_jni.so")

#set_target_properties(lib_caffe-jni PROPERTIES IMPORTED_LOCATION
#                    /home/irina/caffe-android-lib/android_lib/caffe/lib/libcaffe_jni.so)
###############################################################################
add_library( mystitcher SHARED "src/main/cpp/mystitcher_jni.cpp")

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

#target_link_libraries( MyLib ${log-lib})
target_link_libraries( mystitcher ${log-lib} ${OpenCV_LIBS} )