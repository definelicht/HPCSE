cmake_minimum_required(VERSION 2.8)
project(hpcse_diffusion)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DIFFUSION_SRC 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DiffusionJob.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DiffusionParallel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DiffusionSequential.cpp)
add_library(diffusion ${DIFFUSION_SRC})
target_link_libraries(diffusion ${HPCSE_LIBS})
