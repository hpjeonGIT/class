include_directories(${CMAKE_SOURCE_DIR}/src)
add_executable(test_2x   unit_test.cpp)
target_link_libraries(test_2x PRIVATE ext_func)
add_test(NAME run_test_2x   COMMAND test_2x )
