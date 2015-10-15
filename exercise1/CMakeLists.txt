cmake_minimum_required(VERSION 2.8)
project(hpcse_exercise1)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
set(HPCSE_LIBS "${CMAKE_THREAD_LIBS_INIT}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 ${OpenMP_CXX_FLAGS}")
add_executable(RiemannSum ${CMAKE_CURRENT_SOURCE_DIR}/RiemannSum.cpp)
target_link_libraries(RiemannSum ${CMAKE_THREAD_LIBS_INIT}) 
