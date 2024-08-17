set(hip_source vectorAdd.hip helper_hip.h  helper_string.h )
add_library(vhip_lib SHARED ${hip_source})
include_directories(${CMAKE_SOURCE_DIR}/src)
# 
set(main_source main.cpp)
add_executable(a.exe ${main_source})
target_link_libraries(a.exe vhip_lib)
install(TARGETS a.exe DESTINATION ${CMAKE_BINARY_DIR}/bin)

