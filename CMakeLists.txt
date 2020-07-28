cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sample)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(sample sample.cpp)
target_link_libraries(sample "${TORCH_LIBRARIES}")
target_link_libraries(fine_tuning "${TORCH_LIBRARIES}")
set_property(TARGET sample PROPERTY CXX_STANDARD 14)
