add_executable(PushForRoof PushForRoof.cpp)
add_executable(Bandwidth Bandwidth.cpp)
target_link_libraries(PushForRoof ${HPCSE_LIBS})
target_link_libraries(Bandwidth ${HPCSE_LIBS})
