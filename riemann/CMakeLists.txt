cmake_minimum_required(VERSION 2.8)
project(hpcse_riemann)

find_package(Threads REQUIRED)
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 ${OpenMP_CXX_FLAGS}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
add_library(riemann ${CMAKE_CURRENT_SOURCE_DIR}/src/RiemannSum.cpp)
target_link_libraries(riemann ${CMAKE_THREAD_LIBS_INIT})
