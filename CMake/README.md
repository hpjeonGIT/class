# Sample CMake projects and templates for CPP/MPI code
1. sample1_gtest
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

2. sample2_mpi
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
