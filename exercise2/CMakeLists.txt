cmake_minimum_required(VERSION 2.8)
project(hpcse_exercise2)

add_executable(RunDiffusion ${CMAKE_CURRENT_SOURCE_DIR}/RunDiffusion.cpp)
add_executable(BarrierTest ${CMAKE_CURRENT_SOURCE_DIR}/BarrierTest.cpp)
target_link_libraries(RunDiffusion diffusion)
target_link_libraries(BarrierTest diffusion)
