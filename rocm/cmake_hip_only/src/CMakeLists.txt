set(hip_source vectorAdd.hip helper_hip.h  helper_string.h test.hiph )
add_library(vhip_lib ${hip_source})
#set_target_properties(vhip_lib
#  PROPERTIES
#    CXX_STANDARD 17
#    CXX_STANDARD_REQUIRED ON
#    CXX_EXTENSIONS OFF
#    HIP_STANDARD 17
#    HIP_STANDARD_REQUIRED ON
#    HIP_EXTENSIONS OFF
#    HIP_ARCHITECTURES gfx908
)
#target_compile_definitions(vhip_lib PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(vhip_lib PUBLIC ${CMAKE_SOURCE_DIR}/src)
#target_link_directories(vhip_lib PRIVATE /opt/rocm/lib)
#target_compile_options(vhip_lib PUBLIC 
#  $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
#  $<$<COMPILE_LANGUAGE:HIP>:${HIP_FLAGS}>
#)
# 
set(main_source main.cpp)
add_executable(a.exe ${main_source})
target_link_libraries(a.exe vhip_lib)
install(TARGETS a.exe DESTINATION ${CMAKE_BINARY_DIR}/bin)

