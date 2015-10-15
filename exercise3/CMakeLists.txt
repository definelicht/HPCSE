cmake_minimum_required(VERSION 2.8)
project(hpcse_exercise3)

if (NOT HPCSE_OPENMP)
  message(WARNING "Diffusion built without OpenMP. The code will not be accelerated!") 
endif()
