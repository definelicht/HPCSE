cmake_minimum_required(VERSION 2.8)
project(dhcp_exercise2)

find_package(Threads REQUIRED)
set(HPCSE_LIBS "${CMAKE_THREAD_LIBS_INIT}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wall -Wextra -Weffc++")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(DIFFUSION_SRC 
  ${CMAKE_CURRENT_SOURCE_DIR}/DiffusionJob.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DiffusionParallel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DiffusionSequential.cpp)
add_library(diffusion ${DIFFUSION_SRC})
target_link_libraries(diffusion ${HPCSE_LIBS})

add_executable(RunDiffusion ${CMAKE_CURRENT_SOURCE_DIR}/RunDiffusion.cpp)
add_executable(BarrierTest ${CMAKE_CURRENT_SOURCE_DIR}/BarrierTest.cpp)
target_link_libraries(RunDiffusion ${HPCSE_LIBS} diffusion) 
target_link_libraries(BarrierTest ${HPCSE_LIBS} diffusion)
