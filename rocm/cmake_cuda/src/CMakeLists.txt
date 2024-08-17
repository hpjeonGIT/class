set(cuda_source vectorAdd.cu helper_cuda.h  helper_string.h )
add_library(vcuda_lib SHARED ${cuda_source})
target_link_libraries(vcuda_lib -L${CUDA_ROOT}/lib64 -lcudart )
include_directories(${CMAKE_SOURCE_DIR}/src)
# 
set(main_source main.cpp)
add_executable(a.exe ${main_source})
target_link_libraries(a.exe vcuda_lib)
install(TARGETS a.exe DESTINATION ${CMAKE_BINARY_DIR}/bin)

