include(ExternalProject)
ExternalProject_Add(
   SUPERLU_project
  URL ${CMAKE_SOURCE_DIR}/../archive/superlu-7.0.0
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_SOURCE_DIR}/../SHARE
)

include_directories(${CMAKE_SOURCE_DIR}/../SHARE/include)
add_library(lib_su STATIC IMPORTED GLOBAL)
set_target_properties(lib_su PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../SHARE/lib/libblas.a)
add_dependencies(lib_su SUPERLU_project)
