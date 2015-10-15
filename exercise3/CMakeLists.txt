cmake_minimum_required(VERSION 2.8)
project(hpcse_exercise3)

if (NOT HPCSE_OPENMP)
  message(WARNING "Project built without OpenMP. Exercise 3 will not be built.") 
else()
  message(STATUS "Project built with OpenMP support. To run Exercise 3, run the executable of Exercise 2.") 
endif()
