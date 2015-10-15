cmake_minimum_required(VERSION 2.8)
project(hpcse_diffusion)

find_package(Threads REQUIRED)
if (HPCSE_OPENMP)
  message(STATUS "Activated OpenMP acceleration of scalar diffusion implementation.")
  find_package(OpenMP REQUIRED)
  set(HPCSE_LIBS ${HPCSE_LBIS} ${CMAKE_THREAD_LIBS_INIT})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
else()
  message(STATUS "Compiling diffusion without OpenMP acceleration.")
endif()
set(HPCSE_LIBS ${HPCSE_LIBS} ${CMAKE_THREAD_LIBS_INIT})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wall -Wextra -Weffc++")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DIFFUSION_SRC 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DiffusionJob.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DiffusionParallel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DiffusionSequential.cpp)
add_library(diffusion ${DIFFUSION_SRC})
target_link_libraries(diffusion ${HPCSE_LIBS})
