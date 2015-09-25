cmake_minimum_required(VERSION 3.0)
project(dhcp_exercise1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
find_package(Threads REQUIRED)
add_executable(RiemannSum ${CMAKE_CURRENT_SOURCE_DIR}/RiemannSum.cpp)
target_link_libraries(RiemannSum ${CMAKE_THREAD_LIBS_INIT}) 
