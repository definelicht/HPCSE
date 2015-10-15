cmake_minimum_required(VERSION 2.8)
project(hpcse)

option(HPCSE_OPENMP "Accelerate using OpenMP where available." ON)

find_package(Threads REQUIRED)
if (HPCSE_OPENMP)
  find_package(OpenMP)
endif()
if (OpenMP_FOUND OR OPENMP_FOUND)
  message(STATUS "Building HPCSE code WITH OpenMP support.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
else()
  message(STATUS "Building HPCSE code WITHOUT OpenMP support.")
endif()
set(HPCSE_LIBS ${HPCSE_LIBS} ${CMAKE_THREAD_LIBS_INIT})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wall -Wextra -Weffc++")

add_subdirectory(riemann)
add_subdirectory(diffusion)

include_directories(${CMAKE_SOURCE_DIR}/riemann/include)
include_directories(${CMAKE_SOURCE_DIR}/diffusion/include)

add_subdirectory(exercise1)
add_subdirectory(exercise2)
add_subdirectory(exercise3)
