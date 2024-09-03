project(vecAddCuda CXX HIP) 
set(ROCM_PATH /opt/rocm)
set(HIP_PATH /opt/rocm)
add_definitions(-std=c++17)

add_definitions(-DROCM  -D__HIP_PLATFORM_AMD__)
set(default_build_type "Debug") 
include_directories(${HIP_PATH}/include)
include_directories(/usr/local/cuda/include)

# CMAKE_BUILD_TYPE can be over-ridden by -DCMAKE_BUILD_TYPE=Release
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -Wall")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++17" )
add_subdirectory(src)

