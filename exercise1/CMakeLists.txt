cmake_minimum_required(VERSION 2.8)
project(hpcse_exercise1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
add_executable(RunRiemann ${CMAKE_CURRENT_SOURCE_DIR}/RunRiemann.cpp)
target_link_libraries(RunRiemann riemann) 
