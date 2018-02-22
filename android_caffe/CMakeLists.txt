cmake_minimum_required(VERSION 2.8)

if(ANDROID)
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11
  #target_link_libraries(MY_TARGET_NAME ${OpenCV_LIBS})
  add_library(caffe_jni SHARED segmentator_stitcher_jni.cpp segmentator_stitcher.cpp)
  #add_executable(caffe_jni caffe_jni.cpp caffe_mobile.cpp)
  target_link_libraries(caffe_jni ${Caffe_LINK} ${OpenCV_LIBS})
  caffe_default_properties(caffe_jni)

  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(caffe_jni PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/android")

  caffe_set_solution_folder(caffe_jni android)

  # install
  install(TARGETS caffe_jni DESTINATION lib)

endif()

