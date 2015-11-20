add_executable(BinomialSquares BinomialSquares.cpp)
add_executable(RunLennardJones RunLennardJones.cpp)
target_link_libraries(RunLennardJones ${HPCSE_LIBS} lennardjones)
