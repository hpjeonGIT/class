project(vecAddCuda CXX CUDA)
cmake_minimum_required(VERSION 3.0.0)
add_definitions(-std=c++17)
find_package(CUDA REQUIRED)
set(default_build_type "Debug") 
# CMAKE_BUILD_TYPE can be over-ridden by -DCMAKE_BUILD_TYPE=Release
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CUDA_ROOT "/usr/local/cuda")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -Wall")
#
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17" )
include_directories(${CUDA_INCLUDE_DIRS})
add_subdirectory(src)

