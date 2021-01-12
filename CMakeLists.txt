cmake_minimum_required(VERSION 3.17)
project(opencv)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDA  REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
message(${OpenCV_INCLUDE_DIRS})
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv
        detection_and_tracking.cpp
        DeepLearning.cpp
        DeepLearning.hpp
        )
target_link_libraries(opencv opencv_core opencv_videoio opencv_highgui opencv_tracking)