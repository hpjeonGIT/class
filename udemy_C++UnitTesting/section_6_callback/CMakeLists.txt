cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(GTEST_INCLUDE_DIRS /home/hpjeon/sw_local/gtest/include)
set(GTEST_INCLUDE_DIR "/home/hpjeon/sw_local/gtest/include")
set(GTEST_LIBRARY "/home/hpjeon/sw_local/gtest/lib/libgtest.a")
set(GTEST_MAIN_LIBRARY "/home/hpjeon/sw_local/gtest/lib/libgtest_main.a")
find_package(GTest REQUIRED)
message("GTEST_INCLUDE_DIRS = ${GTEST_INCLUDE_DIRS}")

include_directories(${GTEST_INCLUDE_DIR})

set(GMOCK_LIBRARIES /home/hpjeon/sw_local/gtest/lib/libgmock.a /home/hpjeon/sw_local/gtest/lib/libgmock_main.a)

add_library(databaseConnectionLibrary IDatabaseConnection.cpp)
add_library(employeeManagerLibrary EmployeeManager.cpp)

set(ACTUAL_LIBRARIES databaseConnectionLibrary employeeManagerLibrary)

add_executable(mainApp main.cpp)
target_link_libraries(${ACTUAL_LIBRARIES})

add_executable(testRunner testRunner.cpp)
target_link_libraries(testRunner 
    ${GTEST_LIBRARIES} 
    ${GMOCK_LIBRARIES} 
    pthread 
    ${ACTUAL_LIBRARIES})

