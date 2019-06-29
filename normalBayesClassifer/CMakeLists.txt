cmake_minimum_required( VERSION 2.8 )
project( nbcDemo )

set( CMAKE_BUILD_TYPE Debug )
set( CMAKE_CXX_FLAGS "-std=c++11 -O0" )

# opencv
set(OpenCV_DIR /home/intesight/opencv/opencv/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( nbcDemo demo.cpp )
target_link_libraries( nbcDemo ${OpenCV_LIBS} )
