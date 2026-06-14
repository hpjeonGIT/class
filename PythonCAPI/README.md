# Objectives
- How to integrate C/C++ code into Python?
  - Run C/C++ functions in Python environment
    - You may not be able to run C/C++ functions directly from Python environment. You will need helper functions, running brokers b/w those Python C/C++ APIs and existing C/C++ source codes
  - Script or interactive
- For HPC
  - Must be fast - no overhead of memory copy/data marshaling
- Multi-purpose
  - Data from C/C++ then ML in Python. Results to C/C++ back
- Available integration methods
  - ctypes: simple but infamous for the overhead of data marshaling
    - When numpy array is used, the performance is equivalent to other APIs
  - cython: need to rewrite a new code of cython
  - cffi: ?
    - Similar to ctypes, loading external shared libs and simple. Performance must be tested though.
  - pybind11: known as slower than cython
    - In this study, the performance is quite equivalent to cython/swig/CPython
  - CPython: bare PyObject API - what benefit would be there?
  - swig: highly complicated but applicable to large-scale projects
    - C/C++ code must be written according to the swig *.i files. Existing C/C++ code cannot be used without modification
  
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

### Using numpy array
- Same arr_test.c and arr_test.py above
- test_arr_np.py
```py
import arr_test
import ctypes
import time
import random
import numpy as np
nsize = 10_000_000
tic = time.perf_counter()
a_list = np.random.randint(-5,5,nsize,np.int32)
print(f"np random list took {(time.perf_counter() - tic)} sec") # took 
# Run test on python script
tic = time.perf_counter()
y = arr_test.sumArray(a_list)
py_runtime = time.perf_counter() - tic
print(f'Python results= {y} {py_runtime:5.3f} sec')
# C types: gcc -Ofast -fPIC -shared -o libsum_arr.so arr_test.c
tic = time.perf_counter()
libarr = ctypes.cdll.LoadLibrary("./libsum_arr.so")
libarr.sumArrayC.argtypes= [ np.ctypeslib.ndpointer(dtype=np.int32,ndim=1, flags
='C_CONTIGUOUS'), ctypes.c_size_t]
libarr.restype = ctypes.c_size_t
y= libarr.sumArrayC(a_list,nsize) # no conversion of data type
c_runtime = time.perf_counter() - tic
# Print results
print(f'ctype results={y} {c_runtime:5.3f}sec')
print(f'Ctypes is {(py_runtime/c_runtime):3.1f}x faster')
```
- Ctypes is 40-120x faster than Python
  - Ctypes took 0.004-0.014 sec
  - Python took 0.4-0.5 sec

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
- Ref:
  - https://realpython.com/build-python-c-extension-module/
  - https://lectures.scientific-python.org/advanced/interfacing_with_c/interfacing_with_c.html
- sum_array.c:
```c
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
 
static PyObject* calc(PyObject* self, PyObject* args)
{
    PyObject *input_obj;
    if(!PyArg_ParseTuple(args,"O", &input_obj)) { return NULL;}
    if(!PyArray_Check(input_obj)) { PyErr_SetString(PyExc_TypeError, "not numpy 
array"); return NULL; }
    PyArrayObject *clean_array = NULL;
    clean_array = (PyArrayObject*) PyArray_FROM_OTF( input_obj,
                   NPY_INT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    npy_intp asize = PyArray_SIZE(clean_array);   
    int *data_ptr = (int *)PyArray_DATA(clean_array);
    npy_intp sum = 0;
    for (npy_intp i=0;i<asize;i++) sum +=data_ptr[i];
    Py_DECREF(clean_array);
    return PyLong_FromLong((long)sum);
}
 
/*  define functions in module */
static PyMethodDef SumMethods[] =
{
     {"func_sum_np_array", calc, METH_VARARGS,
         "evaluate the sum on a NumPy array"},
     {NULL, NULL, 0, NULL}
};
 
 
static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "module_sum_np", "Some documentation",
    -1,
    SumMethods
};
PyMODINIT_FUNC PyInit_module_sum_np(void) {
    PyObject *module;
    module = PyModule_Create(&cModPyDem);
    if(module==NULL) return NULL;
    /* IMPORTANT: this must be called */
    import_array();
    if (PyErr_Occurred()) return NULL;
    return module;
}
```
- setup.py:
```py
from setuptools import setup, Extension
import numpy
# define the extension module
my_ext_module = Extension(
    "module_sum_np", sources=["sum_array.c"], include_dirs=[numpy.get_include()]
)
# run the setup
setup(ext_modules=[my_ext_module])
```
- Build command: `python3 setup.py build_ext --inplace` # This produces sum_module_np.cpython-313-x86_64-linux-gnu.so
  - Or `python3 setup.py build_ext -i`
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
import module_sum_np
nsize = 10_000_000
tic = time.perf_counter()
a_list = np.random.randint(-5,5,nsize,np.int32)
print(f"random list took {(time.perf_counter() - tic):3.1f} sec") 
# Run test on python script
tic = time.perf_counter()
y = arr_test.sumArray(a_list)
py_runtime = time.perf_counter() - tic
print(f'Python results= {y} {py_runtime:5.3f} sec')
# cpython with numpy
tic = time.perf_counter()
y = module_sum_np.func_sum_np_array(a_list)
cp_runtime = time.perf_counter() - tic
# Print results
print(f'CPython results= {y} {cp_runtime:5.3f} sec')
print(f'CPython is {(py_runtime/cp_runtime):3.1f}x faster')
```
- CPython is 70-100x faster than python
  - Python took 0.447sec
  - CPython took 0.006sec 
- Discuss why a following cpp function is slower:
```cpp
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <iostream>
 
static PyObject* process_data(PyObject* self, PyObject* const* args,
                             size_t nargs, PyObject* kwnames) {
    if (nargs < 1) {
        PyErr_SetString(PyExc_TypeError, "Expected at least 1 argument");
        return NULL;
    }
 
    // Treat the first argument as a NumPy array
    PyArrayObject* arr = (PyArrayObject*)args[0];
    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a numpy array");
        return NULL;
    }
 
    // 1. Extract data into std::vector
    int* data_ptr = (int*)PyArray_DATA(arr);
    npy_intp size = PyArray_SIZE(arr);
    std::vector<int> vec(data_ptr, data_ptr + size);
 
    // 2. Perform a simple operation with std::vector
    int sum = 0;
    for (int val : vec) sum += val;
    return PyLong_FromLong((long)sum);
}
 
// Module definition
static PyMethodDef MyMethods[] = {
    {"func_sum_np_array", (PyCFunction)process_data, METH_FASTCALL, "Process num
py array"},
    {NULL, NULL, 0, NULL}
};
 
static struct PyModuleDef mymodule = {
    PyModuleDef_HEAD_INIT, "mod_sum_np_array", NULL, -1, MyMethods
};
 
PyMODINIT_FUNC PyInit_mod_sum_np_array(void) {
    import_array(); // Initialize NumPy C-API
    return PyModule_Create(&mymodule);
}
```
- setup.py
```py
from setuptools import setup, Extension
import numpy
# define the extension module
my_mod = Extension(
    "mod_sum_np_array", sources=["sum_array.cpp"],
    include_dirs=[numpy.get_include()],
    language="c++")
# run the setup
setup(ext_modules=[my_mod])
```

## SWIG
- Download numpy.i: https://github.com/numpy/numpy/blob/main/tools/swig/numpy.i
- Download source of swig: https://github.com/swig/swig

### Simple functions
- calc.h
```cpp
double add(double a, double b);
double subtract(double a, double b);
```
- calc.cpp
```cpp
#include "calc.h"

double add(double a, double b) {
    return a + b;
}

double subtract(double a, double b) {
    return a - b;
}
```
- calc.i
```js
%module calc
%{
#include "calc.h"
%}

/* Tell SWIG to wrap everything in this header */
%include "calc.h"
```
- setup.py
```py
from setuptools import setup, Extension

calc_module = Extension(
    '_calc',
    sources=['calc_wrap.cxx', 'calc.cpp'],
)

setup(
    name='calc',
    version='0.1',
    ext_modules=[calc_module],
    py_modules=["calc"],
)
```
- Steps to compile:
  - `swig -python -c++ calc.i` # this produces calc_wrap.cxx and calc.py
  - `python3 setup.py build_ext -i` # this produces _calc.cpython-312-x86_64-linux-gnu.so
- Running setup.py is equivalent to:
```bash
g++ -g -O2 -Wall -fPIC -I/usr/include/python3.12 -c calc.cpp -o calc.o
g++ -g -O2 -Wall -fPIC -I/usr/include/python3.12 -c calc_wrap.cxx -o calc_wrap.o
g++  -shared calc.o calc_wrap.o -L/usr/lib/x86_64-linux-gnu -o _calc.cpython-312-x86_64-linux-gnu.so 
```
- Demo:
```bash
$ python3
>>> import calc
>>> calc.add(4.567,3.14)
7.707000000000001
>>> calc.subtract(4.567,3.14)
1.427
```

### With numpy array
- ex.h:
```cpp
void sum_array(int * vec1, int n1, int *res);
```
- ex.cpp:
```cpp
#include "ex.h"
void sum_array(int* vec1, int n1, int *res) { // note that the function return type is void, not int. 
    *res = 0;  // Return value res is defined as a pointer
    for(int i=0; i<n1; i++) *res += vec1[i];
}
```
- ex.i:
```js
%module ex
%{
#define SWIG_FILE_WITH_INIT
#include "ex.h"
%}

/* Include the NumPy typemaps */
%include "numpy.i"

%init %{
import_array();
%}

/* Map (double* vec, int n) to a 1D NumPy input array */
%apply (int* IN_ARRAY1, int DIM1) {(int* vec1, int n1)};
%apply (int* OUTPUT) {int* res}; /* return type is defined as a pointer */

%include "ex.h"
```
- setup.py:
```py
from setuptools import setup, Extension
import numpy
ex_module = Extension(
    '_ex',
    sources=['ex_wrap.cxx', 'ex.cpp'],
    include_dirs = [numpy.get_include()],
)

setup(
    name='ex',
    version='0.1',
    ext_modules=[ex_module],
    py_modules=["ex"],
)
```
- LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 swig -python -c++ calc.i
- python3 setup.py build_ext -i
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
import ex
nsize = 10_000_000
tic = time.perf_counter()
a_list = np.random.randint(-5,5,nsize,np.int32)
print(f"random list took {(time.perf_counter() - tic):3.1f} sec") # took 3.4sec
# Run test on python script
tic = time.perf_counter()
y = arr_test.sumArray(a_list)
py_runtime = time.perf_counter() - tic
print(f'Python results= {y} {py_runtime:5.3f} sec')
# cffi
tic = time.perf_counter()
y = ex.sum_array(a_list)
swig_runtime = time.perf_counter() - tic
# Print results
print(f'swig results= {y} {swig_runtime:5.3f} sec')
print(f'swig is {(py_runtime/swig_runtime):3.1f}x faster')
```
- Swig is ~80x faster than Python
  - Python took 0.445 sec
  - Swig took 0.006 sec
  
## Benchmark

## References
- https://levelup.gitconnected.com/programming-language-efficiency-deep-dive-choosing-the-right-tool-for-the-job-f08397982638
- https://stackoverflow.com/questions/7799977/numpy-vs-cython-speed#10486566
