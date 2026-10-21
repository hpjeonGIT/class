ExternalProject_Add(
    GTEST_PROJECT
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/extern/googletest-1.18.0"
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz #"${CMAKE_SOURCE_DIR}/extern/googletest-1.18.0.tar.gz"
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DBUILD_TESTING:BOOL=OFF
    BUILD_IN_SOURCE 0
    BUILD_COMMAMD make -j 5
    #BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
    #INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/gtest
    #INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)
IF($BUILD_SHARED_BUILD)
  add_library(gtest SHARED IMPORTED GLOBAL)
  set_target_properties(gtest PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lib/libgtest.so)
ELSE()
  add_library(gtest STATIC IMPORTED GLOBAL)
  set_target_properties(gtest PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lib/libgtest.a)
ENDIF()
add_dependencies(gtest GTEST_PROEJCT)
