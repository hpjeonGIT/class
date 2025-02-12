include(ExternalProject)
ExternalProject_Add(
   SUPERLU_project
  URL ${CMAKE_SOURCE_DIR}/external/superlu-7.0.0
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/superlu
)

include_directories(${install_dir}/include)
add_library(lib_superlu STATIC IMPORTED)
set_target_properties(lib_superlu PROPERTIES IMPORTED_LOCATION ${install_dir}/lib/libblas.a)
add_dependencies(lib_superlu SUPERLU_project)
