include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
add_library(riemann ${CMAKE_CURRENT_SOURCE_DIR}/src/RiemannSum.cpp)
target_link_libraries(riemann ${HPCSE_LIBS})
