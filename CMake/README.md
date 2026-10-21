# Sample CMake projects
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
--   /XXXX/sample1_gtest/b2/extern/include
--   /XXXX/sample1_gtest/src
-- Configuring done (0.3s)
-- Generating done (0.0s)
...
$ ctest
Test project /home/hpjeon/hw/class/cmake/sample1_gtest/b2
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
