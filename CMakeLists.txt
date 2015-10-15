cmake_minimum_required(VERSION 2.8)
project(hpcse)

option(HPCSE_OPENMP "Accelerate using OpenMP where available." ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wall -Wextra -Weffc++")

add_subdirectory(riemann)
add_subdirectory(diffusion)

include_directories(${CMAKE_SOURCE_DIR}/riemann/include)
include_directories(${CMAKE_SOURCE_DIR}/diffusion/include)

add_subdirectory(exercise1)
add_subdirectory(exercise2)
add_subdirectory(exercise3)
