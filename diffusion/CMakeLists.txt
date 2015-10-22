include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DIFFUSION_SRC 
  src/Diffusion.cpp
  src/DiffusionJob.cpp
  src/DiffusionParallel.cpp
  src/DiffusionSequential.cpp
  src/RandomWalk.cpp)
if (HPCSE_OPENMP_FOUND)
  set(DIFFUSION_SRC ${DIFFUSION_SRC} src/RandomWalk.cpp)
else()
  message(WARNING "Diffusion: compiling without OpenMP. Random walk not available.")
endif()
add_library(diffusion ${DIFFUSION_SRC})
target_link_libraries(diffusion ${HPCSE_LIBS})
