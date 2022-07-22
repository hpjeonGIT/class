cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

find_package(GTest REQUIRED)
message("GTest_INCLUDE_DIRS = ${GTest_INCLUDE_DIRS}")
set(GMOCK_LIBRARIES ${GTEST_INCLUDE_DIR}/../lib/libgmock.a)
add_executable(mainApp main.cpp)
target_link_libraries(mainApp ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} pthread)

