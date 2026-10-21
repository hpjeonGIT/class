cmake_minimum_required(VERSION 3.0.0)
project(CALC_project VERSION 1.0.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(ExternalProject)
enable_testing() # this must be present prior to including test folders
add_subdirectory(extern)
include_directories(${CMAKE_BINARY_DIR}/extern/include)
add_subdirectory(src)
##
