find_package(Threads REQUIRED)
message(STATUS ${CMAKE_THREAD_LIBS_INIT})
add_executable(test_2x   unit_test.cpp)
target_link_libraries(test_2x PRIVATE ext_func  ${CMAKE_THREAD_LIBS_INIT}) # the last component adds -lpthread
add_executable(gtest_2x  using_gtest.cpp) 
add_dependencies(gtest_2x gtest  ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(gtest_2x PRIVATE ext_func gtest)
add_test(NAME run_test_2x   COMMAND test_2x )
add_test(NAME run_gtest_2x   COMMAND gtest_2x )

