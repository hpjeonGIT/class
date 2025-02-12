include(ExternalProject)
ExternalProject_Add(
  SUPERLU_project
  URL ${CMAKE_SOURCE_DIR}/../archive/superlu-7.0.0
  CMAKE_ARGS 
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_SOURCE_DIR}/../SHARE
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAkE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${SU_PREFIX}
 #SOURCE_DIR ${CMAKE_SOURCE_DIR}/../archive/superlu-7.0.0
 BUILD_IN_SOURCE 0
 BUILD_COMMAND make -j 4
 BINARY_DIR "${CMAKE_SOURCE_DIR}/../build/superlu-7.0.0"
 BUILD_ALWAYS OFF
 STAMP_DIR "${CMAKE_SOURCE_DIR}/../stamp/superlu-7.0.0"
)

#ExternalProject_Get_Property(SUPERLU_PROJECT install_dir)
include_directories(${CMAKE_SOURCE_DIR}/../SHARE/include)
add_library(lib_su STATIC IMPORTED GLOBAL)
set_target_properties(lib_su PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../SHARE/lib/libblas.a)
add_dependencies(lib_su SUPERLU_project)
