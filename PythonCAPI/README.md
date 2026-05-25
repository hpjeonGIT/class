# Objectives
- How to integrate C/C++ code into Python?
  - Run C/C++ functions in Python environment
  - Script or interactive
- For HPC
  - Must be fast - no overhead of memory copy/data marshaling
- Multi-purpose
  - Data from C/C++ then ML in Python. Results to C/C++ back
- Available integration methods
  - ctypes: simple but infamous for the overhead of data marshaling
  - cython: need to rewrite a new code of cython
  - pybind11: known as slower than cython
  - swig: highly complicated but applicable to large-scale projects
  - numpy: bare PyObject API (?)

## ctypes

### Functions of a single argument
- sum_test.py 
  - Ref: https://stackoverflow.com/q/78571450
```py
def sumTo(x):
  y = 0
  for i in range(x):
    y += i
  return y
```
- sum_test.c
```c
int sumTo(int x){
  int y = 0;
  for(int i =0;i<x;i++) {
    y+=i;
  }
  return y;
}
```
- Building library: `gcc -shared -o libsum_test.so -fPIC sum_test.c -Ofast`
- test_run.py:
```py
# Source - https://stackoverflow.com/q/78571450
# Posted by rakmo97, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-25, License - CC BY-SA 4.0
import sum_test
from ctypes import *
import time
numRuns = 10
x = 1_000_000

# Run test on python script
tic = time.perf_counter()
for i in range(numRuns):
    sum_test.sumTo(x)
py_runtime = time.perf_counter() - tic

# Run test on c script
libCalc = CDLL("./libsum_test.so")
tic = time.perf_counter()
for i in range(numRuns):
    libCalc.sumTo(x)
c_runtime = time.perf_counter() - tic
# Print results
print(py_runtime, c_runtime)
print('Ctypes is {}x faster'.format(py_runtime/c_runtime))
```
- ctypes is faster than Python code by ~330x
- The above example has only one argument and may not be appropriate to consider the overhead of memory marshaling

### Functions of array arguments
- Note that we use array.array instead of list
- arr_test.py:
```py
def sumArray(arr):
  y = 0
  for i in arr:
    y += i
  return y
```
- arr_test.c:
```c
int sumArrayC(int *x, int n)
{
  int y = 0;
  for(int i=0;i<n; i++) y += x[i];
  return y;
}
```
- test_arr.py
```py
import arr_test
import ctypes
import time
import random
import array
nsize = 10_000_000
arr = array.array('i')
tic = time.perf_counter()
for i in range(nsize):
  arr.append(random.randint(-5,5))
print(f"random list took {(time.perf_counter() - tic)} sec") # took 3.4sec
# Run test on python script
tic = time.perf_counter()
y = arr_test.sumArray(arr)
py_runtime = time.perf_counter() - tic
print(f'Python results= {y} {py_runtime:5.3f} sec')
# C types: gcc -Ofast -fPIC -shared -o libsum_arr.so arr_test.c
libarr = ctypes.CDLL("./libsum_arr.so")
tic = time.perf_counter()
ctarr = (ctypes.c_int * nsize)(*arr)
y= libarr.sumArrayC(ctarr,nsize)
c_runtime = time.perf_counter() - tic
# Print results
print(f'ctype results={y} {c_runtime:5.3f}sec')
print(f'Ctypes is {(py_runtime/c_runtime):3.1f}x faster')
```
- ctypes is faster than python code by 68.6x
  - However, including the conversion of Python list data into ctypes array, ctypes is 0.2x than Python code
    - Python took 0.204sec
    - Ctype took 0.952sec
  - Data conversion is very heavy

## cffi

## Cython
- A new language (?) of C + Python
- Why ctypes is slower than Cython?
  - Ref: https://stackoverflow.com/questions/78571450/execution-speed-cython-vs-ctypes

### Functions of a single argument  
- sum_test_cy.pyx:
  - Ref: https://stackoverflow.com/q/78571450
```py
cpdef int sumTo(int x):
    cdef int y = 0
    cdef int i
    for i in range(x):
        y += i
    return y
```    
- setup.py
```py
from setuptools import setup, Extension
from distutils.core import setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "sum_test_cy",
        ["sum_test_cy.pyx"],
        extra_compile_args=["-Ofast"],
    )
]
setup(ext_modules = cythonize(ext_modules))
```
- Compilation command: `python3 setup.py build_ext --inplace`
- run_test.py:
```py
# Source - https://stackoverflow.com/q/78571450
# Posted by rakmo97, modified by community. See post 'Timeline' for change histo
ry
# Retrieved 2026-05-25, License - CC BY-SA 4.0

import time
import sum_test_cy
import sum_test

numRuns = 10
x = 1_000_000

# Run test on python script
tic = time.perf_counter()
for i in range(numRuns):
    sum_test.sumTo(x)
py_runtime = time.perf_counter() - tic

# Run test on cython
tic = time.perf_counter()
for i in range(numRuns):
    sum_test_cy.sumTo(x)
cy_runtime = time.perf_counter() - tic

# Print results
print(py_runtime, cy_runtime)
print('Cython is {}x faster'.format(py_runtime/cy_runtime))
```
- Cython is faster than Python code by ~430x

### Functions of array arguments
- Ref: https://www.geeksforgeeks.org/python/high-performance-array-operations-with-cython-set-1/
- As list is not allowed for cython argument, we use array.array
- arr_test.py:
```py
def sumArray(arr):
  y = 0
  for i in arr:
    y += i
  return y
```
- arr_test_cy.pyx:
```py
cimport cython
cpdef sum_array_cy(int[:] a):
  cdef int y = 0
  cdef int i
  for i in range(a.shape[0]):
    y += a[i]
  return y
```
- setup.py:
```py
from setuptools import setup, Extension
from distutils.core import setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "arr_test_cy",
        ["arr_test_cy.pyx"],
        extra_compile_args=["-Ofast"],
    )
]
setup(ext_modules = cythonize(ext_modules))
```
- Command to compile: `python3 setup.py build_ext --inplace`
- run_arr.py:
```py
import arr_test
import arr_test_cy
import time
import random
import array
nsize = 10_000_000
a_list = array.array('i')
tic = time.perf_counter()
for i in range(nsize):
  a_list.append(random.randint(-5,5))
print(f"random list took {(time.perf_counter() - tic):3.1f} sec") # took 3.4sec
# Run test on python script
tic = time.perf_counter()
y = arr_test.sumArray(a_list)
py_runtime = time.perf_counter() - tic
print(f'Python results= {y} {py_runtime:5.3f} sec')
# cython
tic = time.perf_counter()
y = arr_test_cy.sum_array_cy(a_list)
cy_runtime = time.perf_counter() - tic
# Print results
print(f'Cython results= {y} {cy_runtime:5.3f} sec')
print('Cython is {}x faster'.format(py_runtime/cy_runtime))
```
- Cython is 20-30x faster than Python code
- There is no overhead of data conversion
  - Python took 0.228 sec
  - Cython took 0.011 sec

### Functions of numpy array arguments

## Pybind11
- Easy to use but performance would be lower than Cython

## Numpy

## SWIG

## Benchmark

## References
- https://levelup.gitconnected.com/programming-language-efficiency-deep-dive-choosing-the-right-tool-for-the-job-f08397982638
- https://stackoverflow.com/questions/7799977/numpy-vs-cython-speed#10486566
