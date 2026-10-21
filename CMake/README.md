# Sample CMake projects
1. sample1_gtest
    - Downloand/install googletest on-the-fly
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make -j 3
# if google test build is later than src build, it crashes. Run make -j 3 again then
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
