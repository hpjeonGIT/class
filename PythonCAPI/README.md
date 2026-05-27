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
  - cffi: ?
  - pybind11: known as slower than cython
  - swig: highly complicated but applicable to large-scale projects
  - numpy: bare PyObject API (?)
  - CPython: bare PyObject API - what benefit would be there?

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
# Posted by rakmo97, modified by community. See post 'Timeline' for change history
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
- Same Python/Cython code above
- run_arr.py
```py
import arr_test
import arr_test_cy
import time
import numpy as np
import array
nsize = 10_000_000
tic = time.perf_counter()
a_list = np.random.randint(-5,5,nsize,np.int32)
print(f"random list took {(time.perf_counter() - tic):3.1f} sec") # took 0.1sec
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
print(f'Cython is {(py_runtime/cy_runtime):3.1f}x faster')
```
- Cython is 50-70x faster than Python
  - Python took 0.438 sec
    - This is 2x slower than array.array(). Why?
  - Cython took 0.006 sec

## cffi
- ABI vs API
- Ref: 
  - https://kindatechnical.com/high-performance-python-guide/using-cffi-for-c-interfacing.html
  - https://www.iditect.com/faq/python/how-to-pass-a-numpy-array-into-a-cffi-function-and-how-to-get-one-back-out.html

### ABI example
- hello.c:
```c
#include <stdio.h>
void hello() {
    printf("Hello from C!\n");
}
```
- `gcc -shared -o libhello.so -fPIC hello.c -Ofast`
- run.py:
```py
from cffi import FFI
ffi = FFI()
ffi.cdef("void hello();")
C = ffi.dlopen("./libhello.so")
C.hello()
```

### API example
```py
from cffi import FFI
ffi = FFI()

ffi.cdef("""
    void hello();
""")

C = ffi.verify("""
    #include <stdio.h>

    void hello() {
        printf("Hello from C!\\n");
    }
""")

C.hello()
```

### Functions of numpy array arguments
- arr_test.py:
```py
def sumArray(arr):
  y = 0
  for i in arr:
    y += i
  return y
```
- myarr.c:
```c
int arr_sum(int *x, int nsize)
{
  int y = 0;
  for(int i=0; i<nsize; i++) y += x[i];
  return y;
}
```
- `gcc -Ofagcc -Ofast -shared -fPIC -o libmyarr.so myarr.c`
- run_arr.py:
```py
import arr_test
import time
import numpy as np
from cffi import FFI
nsize = 10_000_000
tic = time.perf_counter()
a_list = np.random.randint(-5,5,nsize,np.int32)
print(f"random list took {(time.perf_counter() - tic):3.1f} sec") # took 0.1sec
# Run test on python script
tic = time.perf_counter()
y = arr_test.sumArray(a_list)
py_runtime = time.perf_counter() - tic
print(f'Python results= {y} {py_runtime:5.3f} sec')
# cffi
ffi = FFI()
tic = time.perf_counter()
mylib = ffi.dlopen('./libmyarr.so')
ffi.cdef("""
int arr_sum(int *x, int nsize);
""")
y = mylib.arr_sum(ffi.cast("int*", a_list.ctypes.data), nsize)
cy_runtime = time.perf_counter() - tic
# Print results
print(f'cffi results= {y} {cy_runtime:5.3f} sec')
print(f'cffi is {(py_runtime/cy_runtime):3.1f}x faster')
```
- cffi is 34-40x faster than Python code
  - Python took 0.436 sec
  - cffi took 0.012 sec

## Pybind11
- Easy to use but performance would be lower than Cython
- Ref: https://medium.com/@ahmedfgad/pybind11-tutorial-binding-c-code-to-python-337da23685dc
- myarr.cpp:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
int arr_sum(pybind11::array_t<int> x, int nsize)
{
  auto buf = x.request(); // buffer info from numpy array
  int *ptr = static_cast<int *>(buf.ptr);  
  int y = 0;
  for(int i=0; i<nsize; i++) y += ptr[i];
  return y;
}
PYBIND11_MODULE(myarr, m) {
    m.doc() = "pybind11 myarr module"; // Optional module docstring
    m.def("arr_sum", &arr_sum, "A function that sums the elements of an array",
          pybind11::arg("x"), pybind11::arg("nsize"));
}
```
- setup.py:
```py
from setuptools import setup, Extension
import pybind11
ext_modules = [
    Extension(
        "myarr",  # Module name
        ["myarr.cpp"],  # Source files
        include_dirs=[pybind11.get_include()],  # Include pybind11 headers
        language="c++",  # Specify C++ as the language
    )
]
setup(
    name="myarr",
    version="0.1",
    ext_modules=ext_modules,
)
```
  - Adding `extra_compile_args=['-Ofast'],` doesn't work, still compiling with -O2
- Build command: `python3 setup.py build`
  - `python3 setup.py install` will install the produced library into python's site-packages
  - Just copy the produced build/lib.linux-x86_64-cpython-313/myarr.cpython-313-x86_64-linux-gnu.so into the current path
- arr_test.py:
```py
def sumArray(arr):
  y = 0
  for i in arr:
    y += i
  return y
```
- run_arr.py:
```py
import arr_test
import time
import numpy as np
import myarr # from myarr.cpython-313-x86_64-linux-gnu.so
nsize = 10_000_000
tic = time.perf_counter()
a_list = np.random.randint(-5,5,nsize,np.int32)
print(f"random list took {(time.perf_counter() - tic):3.1f} sec") # took 0.1sec
# Run test on python script
tic = time.perf_counter()
y = arr_test.sumArray(a_list)
py_runtime = time.perf_counter() - tic
print(f'Python results= {y} {py_runtime:5.3f} sec')
# cffi
tic = time.perf_counter()
y = myarr.arr_sum(a_list, nsize)
py11_runtime = time.perf_counter() - tic
# Print results
print(f'Pybind11 results= {y} {py11_runtime:5.3f} sec')
print(f'Pybind11 is {(py_runtime/py11_runtime):3.1f}x faster')
```
- Pybind11 is 80-90x faster than Python
  - Python took 0.458 sec
  - Pybind11 took 0.006 sec
    - Almost same speed of Cython!

## CPython

### cosine function module
- Ref: https://lectures.scientific-python.org/advanced/interfacing_with_c/interfacing_with_c.html 
- cos_module.c
```c
/*  Example of wrapping cos function from math.h with the Python-C-API. */

#include <Python.h>
#include <math.h>

/*  wrapped cosine function */
static PyObject* cos_func(PyObject* self, PyObject* args)
{
    double value;
    double answer;

    /*  parse the input, from python float to c double */
    if (!PyArg_ParseTuple(args, "d", &value))
        return NULL;
    /* if the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */

    /* call cos from libm */
    answer = cos(value);

    /*  construct the output from cos, from c double to python float */
    return Py_BuildValue("f", answer);
}

/*  define functions in module */
static PyMethodDef CosMethods[] =
{
     {"cos_func", cos_func, METH_VARARGS, "evaluate the cosine"},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "cos_module", "Some documentation",
    -1,
    CosMethods
};

PyMODINIT_FUNC
PyInit_cos_module(void)
{
    return PyModule_Create(&cModPyDem);
}
```
- setup.py
```py
from setuptools import setup, Extension
# define the extension module
cos_module = Extension("cos_module", sources=["cos_module.c"])
# run the setup
setup(ext_modules=[cos_module])
```
- Demo:
```bash
$ python setup.py build_ext --inplace # produces cos_module.cpython-313-x86_64-linux-gnu.so
$ python3
Python 3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 16:09:02) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cos_module
>>> dir(cos_module)
['__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'cos_func']
>>> cos_module.cos_func(1.0)
0.5403023058681398
```

### Coupling with Numpy



## SWIG

## Benchmark

## References
- https://levelup.gitconnected.com/programming-language-efficiency-deep-dive-choosing-the-right-tool-for-the-job-f08397982638
- https://stackoverflow.com/questions/7799977/numpy-vs-cython-speed#10486566
