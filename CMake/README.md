# Sample CMake projects and templates for CPP/MPI code

## 1. sample1_gtest
    - Downloand/install googletest on-the-fly
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make -j 3
...
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- 
-- INCLUDE_DIRECTORIES are:
--   /XXXX/sample1_gtest/build/extern/include
--   /XXXX/sample1_gtest/src
-- Configuring done (0.3s)
-- Generating done (0.0s)
...
$ ctest
Test project /home/hpjeon/hw/class/cmake/sample1_gtest/build
    Start 1: run_test_2x
1/2 Test #1: run_test_2x ......................   Passed    0.00 sec
    Start 2: run_gtest_2x
2/2 Test #2: run_gtest_2x .....................***Failed    0.00 sec  # ---> This is planned

50% tests passed, 1 tests failed out of 2

Total Test time (real) =   0.01 sec

The following tests FAILED:
	  2 - run_gtest_2x (Failed)
Errors while running CTest
```

## 2. sample2_mpi
    - Coupling MPI for CMake
```bash
$ module load openmpi_5.0.10
$ mkdir build
$ cd build
$ cmake -DCMAKE_CXX_COMPILER=mpicxx ..
$ make -j 3
$ mpirun -n 2 src/a_mpi.exe 
 I am rank 0: Input = 123 Answer = 246
$ ctest
Test project /XXXX/sample2_mpi/build
    Start 1: run_test_2x
1/5 Test #1: run_test_2x ......................   Passed    0.00 sec
    Start 2: run_gtest_2x
2/5 Test #2: run_gtest_2x .....................   Passed    0.00 sec
    Start 3: run_mpitest_rank1
3/5 Test #3: run_mpitest_rank1 ................   Passed    0.09 sec
    Start 4: run_mpitest_rank2
4/5 Test #4: run_mpitest_rank2 ................   Passed    0.10 sec
    Start 5: run_mpitest_rank3
5/5 Test #5: run_mpitest_rank3 ................   Passed    0.10 sec

100% tests passed, 0 tests failed out of 5

Label Time Summary:
parallel_test    =   0.28 sec*proc (3 tests)
serial_test      =   0.00 sec*proc (2 tests)

Total Test time (real) =   0.29 sec
$ ctest -L serial_test
Test project /ZZZZZ/sample2_mpi/build
    Start 1: run_test_2x
1/2 Test #1: run_test_2x ......................   Passed    0.00 sec
    Start 2: run_gtest_2x
2/2 Test #2: run_gtest_2x .....................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 2

Label Time Summary:
serial_test    =   0.00 sec*proc (2 tests)

Total Test time (real) =   0.00 sec
$ ctest -L parallel_test
Test project /YYYY/sample2_mpi/build
    Start 3: run_mpitest_rank1
1/3 Test #3: run_mpitest_rank1 ................   Passed    0.09 sec
    Start 4: run_mpitest_rank2
2/3 Test #4: run_mpitest_rank2 ................   Passed    0.09 sec
    Start 5: run_mpitest_rank3
3/3 Test #5: run_mpitest_rank3 ................   Passed    0.09 sec

100% tests passed, 0 tests failed out of 3

Label Time Summary:
parallel_test    =   0.26 sec*proc (3 tests)

Total Test time (real) =   0.27 sec
```

## 3. sample3_cyclic_depend
- Cyclic dependency
  - lib2x needs the definition of ret_3x() and ret_4x() from lib3x and lib4x
  - lib3x needs the definition of ret_4x() and ret_2x() from lib4x and lib2x
  - lib4x needs the definition of ret_2x() and ret_3x() from lib2x and lib3x
  - At link stage (not compiling), repeat library calls then the linker can handle cyclic dependency: `g++ main.cpp.o -o a.exe  liblib2x.a liblib3x.a liblib4x.a liblib2x.a liblib3x.a liblib4x.a` 
- For STATIC libraries, cmake will scan twice for cyclic dependency. If more repetitions are required, then use LINK_INTERFACE_MULTIPLICITY such as: `set_target_properties(lib2x PROPERTIES LINK_INTERFACE_MULTIPLICITY 3)`
- Scenario 1
  - lib2x contains ret_2x() and comb_A() functions
  - main.cpp uses ret_2x() only
  - `g++ main.cpp.o -o a.exe liblib2x.a` fails as comb_A() needs the definition of ret_3x() and ret_4x(), which are from lib3x and lib4x respectively
    ```bash
    $ nm src/liblib2x.a
    func_2x.cpp.o:
    0000000000000016 T _Z6comb_ARi
    0000000000000000 T _Z6ret_2xRi  #<-- defined in the lib2x
                    U _Z6ret_3xRi  #<-- not defined but referred
                    U _Z6ret_4xRi  #<-- not defined but referred
    ```  
  - Solution:
    - Add lib3x and lib4x into linker. In other words, link all lib2x, lib3x, lib4x, and this could be waste of resource (too much redundancy)
    - As comb_A() function is not needed in main.cpp, we may ignore such undefined references. In CMakeLists.txt:
        ```cmake
        target_link_libraries(a.exe lib2x )
        target_link_options(a.exe PRIVATE "LINKER:--unresolved-symbols=ignore-all")
        ```    
        - In CLI: `/usr/bin/c++ -Wl,--unresolved-symbols=ignore-all CMakeFiles/a.exe.dir/main.cpp.o -o a.exe  liblib2x.a`
