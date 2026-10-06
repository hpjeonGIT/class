set(ext_func_src func01.h func01.cpp)
add_library(ext_func SHARED ${ext_func_src})
add_executable(a.exe main.cpp)
target_link_libraries(a.exe ext_func)
include_directories(${CMAKE_SOURCE_DIR}/src)
install(TARGETS a.exe DESTINATION ${CMAKE_BINARY_DIR}/bin)
install(TARGETS ext_func DESTINATION ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(test)

