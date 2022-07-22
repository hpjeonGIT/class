
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

find_package(GTest REQUIRED)
message("GTest_INCLUDE_DIRS = ${GTest_INCLUDE_DIRS}")

set(GMOCK_LIBRARIES /home/hpjeon/sw_local/gtest/lib/libgmock.a /home/hpjeon/sw_local/gtest/lib/libgmock_main.a)
add_library(commonLibrary LibraryCode.cpp)

add_executable(mainApp main.cpp)
target_link_libraries(mainApp commonLibrary)

add_executable(unitTestRunner testRunner.cpp)
target_link_libraries(unitTestRunner commonLibrary ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} pthread)

