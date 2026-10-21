add_executable(test_2x   unit_test.cpp)
target_link_libraries(test_2x PRIVATE ext_func)
add_executable(gtest_2x  using_gtest.cpp) 
add_dependencies(gtest_2x gtest)
target_link_libraries(gtest_2x PRIVATE ext_func gtest)
add_test(NAME run_test_2x   COMMAND test_2x )
add_test(NAME run_gtest_2x   COMMAND gtest_2x )

