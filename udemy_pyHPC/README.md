# High Performance Python: Master Speed, Scale, and Efficiency
- Instructor: Eduero Academy, Inc.

### 1. Introduction

### 2. How to Write High-Performance Python Code
- Python is a versatile and widely-used programming language known for its readability, extensive ecosystem, and rapid development capabilities. However, one common perception is that Python is slower than many compiled languages like C or C++. While this can be true in certain scenarios, Python’s performance potential is often underestimated. By using the right techniques, tools, and best practices, you can write Python code that runs efficiently, even for computationally intensive tasks. In this guide, we will explore a variety of strategies and considerations to help you write high-performance Python code.

1. Understand the Nature of Your Problem
- Before diving into optimizations, it’s important to understand what “performance” means for your specific scenario.

  - CPU-Bound vs. I/O-Bound:

        CPU-Bound Tasks: Operations that spend most of their time using the CPU (e.g., image processing, heavy numerical computations) often require optimized algorithms, vectorization, or native extensions.

        I/O-Bound Tasks: Operations slowed down by input/output, like reading from disk, making network calls, or interacting with databases, can often be improved by using asynchronous I/O or concurrent programming techniques.

  - Algorithmic Complexity:
    Start by examining the time complexity (Big O notation) of your algorithms. A more efficient algorithm often yields bigger performance improvements than any language-level optimization.

- Key takeaway: Determine whether you need to focus on raw CPU speed, smarter algorithms, or better concurrency and I/O handling. This understanding guides all subsequent optimization efforts.

2. Profile Before You Optimize
- Optimizing prematurely can waste time. Instead, use profiling tools to identify the bottlenecks in your code:

  - Built-in Profilers:
    Python’s built-in cProfile and profile modules can measure how much time is spent in each function. You can run cProfile to pinpoint where your code is spending most of its time.

  - Third-Party Tools:
    Tools like line_profiler and py-spy offer finer-grained profiling capabilities to help you understand performance at a line-by-line level.

- By profiling, you ensure that you direct optimization efforts where they matter most, rather than guessing.

3. Choose Efficient Data Structures and Algorithms
- Python offers a variety of built-in data structures and modules that are highly optimized in C. Utilizing these effectively can provide significant gains.

  - Data Structures:

        Use lists for dynamic arrays, but consider the cost of insertions at the front.

        Use collections.deque for efficient FIFO operations.

        Use set and dict for fast membership tests and lookups.

        Avoid unnecessary data structure conversions inside loops.

  - Algorithms:
    Revisit your algorithmic approach. A well-chosen algorithm (e.g., using a heap for priority operations or binary search for sorted data) can dramatically reduce execution time.

- Key takeaway: Optimize logic first. Efficient data structures and algorithmic improvements often outshine micro-optimizations.

4. Leverage Built-In and Standard Library Functions
- The Python standard library and built-in functions are usually implemented in C and are highly optimized. Whenever possible, lean on them instead of writing your own Python-level loops.

  - Built-in Functions:
    Functions like sum(), min(), max(), and sorted() are often faster than manual loops due to their internal optimizations.

  - Standard Library Modules:
    Modules such as itertools and functools provide efficient building blocks for iteration and function operations. For example, itertools.accumulate() and itertools.groupby() can handle complex iteration tasks more efficiently than custom loops.

- Key takeaway: Reuse the wheel—built-in and standard library functions are typically more efficient than “rolling your own.”

5. Vectorize Computations with NumPy and Pandas
- For numerical computations, Python’s speed can be significantly improved by using array-based operations via NumPy or similar libraries.

  - NumPy Arrays:
    NumPy arrays are implemented in C and are memory-efficient. Vectorized operations allow you to apply mathematical operations to an entire array without Python-level loops, resulting in orders-of-magnitude speedups over pure Python lists.

  - Pandas:
    When dealing with tabular data, Pandas takes advantage of NumPy under the hood. Aggregations, merges, and filtering operations are generally much faster than manual loops over Python lists or dictionaries.

- Key takeaway: For data-heavy numerical tasks, vectorization is one of the most powerful optimization techniques in Python.

6. Minimize Overhead in Inner Loops
- If you must use loops, pay attention to what happens inside them. Even small overhead repeated millions of times adds up.

  - Reduce Function Calls:
    Inline small computations rather than calling a helper function repeatedly inside a tight loop.

  - Local Variable Access:
    Accessing local variables is faster than accessing globals, so consider binding frequently used global functions to local variables before loops:

        python
        local_append = my_list.append
        for x in iterable:
            local_append(x)

  - Use Iterators Efficiently:
    Replace manual indexing with direct iteration, and avoid unnecessary range checks or conditionals inside loops.

- Key takeaway: Micro-optimizations inside hot loops can pay dividends, but always focus on clarity first and ensure you really need them.

7. Consider Alternate Python Interpreters and Tools
  - PyPy:
    PyPy is a just-in-time (JIT) compiling Python interpreter that can significantly speed up pure Python code, especially long-running programs.

  - Cython:
    Cython allows you to write Python-like code that compiles to C. By adding type annotations, you can achieve speeds closer to C performance, making it ideal for CPU-bound tasks.

  - Numba:
    Numba uses JIT compilation (via LLVM) to accelerate numerical Python code, making it a great choice for scientific and data analysis workloads that rely on NumPy arrays.
- Key takeaway: If high performance is critical, consider moving performance-sensitive parts of your code to Cython or using a JIT-accelerated interpreter like PyPy or Numba.

8. Utilize Concurrency and Parallelism
- Python’s Global Interpreter Lock (GIL) limits true parallel execution of Python bytecode. However, there are ways to circumvent or alleviate this limitation:

  - Multiprocessing:
    Spawning multiple processes, each running on a separate CPU core, can speed up CPU-bound tasks. The multiprocessing module makes it relatively simple to parallelize computations.

  - Multithreading for I/O:
    For I/O-bound tasks, threads can be beneficial, since while one thread waits for data, another can proceed. The threading module is suitable for this.

  - Async I/O:
    For high-throughput network operations, asyncio and libraries like aiohttp enable asynchronous programming, allowing your code to handle thousands of concurrent I/O tasks efficiently.

  - Offloading to Native Code:
    If you use extensions that release the GIL (like those in NumPy or other C/C++ extensions), you can achieve parallel execution of certain parts of the code.
- Key takeaway: Match your concurrency strategy to the nature of your task. Multiprocessing for CPU-bound, threading or async I/O for I/O-bound.

9. Optimize Memory Usage
- Memory layout and access patterns influence performance, especially for large datasets.

  - Avoid Unnecessary Copies:
    Operations that create copies of large data structures can be expensive. Use views (e.g., NumPy array slices) instead of copies whenever possible.

  - Efficient Object Types:
    In Python, integers and objects have overhead. If possible, use more memory-efficient data representations (e.g., arrays of a fixed type rather than lists of objects).

  - Garbage Collection:
    Python’s garbage collector can introduce overhead in memory-heavy applications. Sometimes, disabling or tuning the garbage collector during critical sections can improve performance.
- Key takeaway: Reducing memory usage often improves CPU performance, because of fewer cache misses and simpler memory management.

10. Test, Measure, and Iterate
- High-performance Python coding is an iterative process. After implementing a change, re-profile and measure performance:

  - Benchmarking:
    Use stable and repeatable benchmarks to measure changes. The timeit module is handy for small snippets.

  - A/B Testing:
    Try multiple optimization strategies and compare their performance. Keep what works best and discard the rest.

  - Maintainability:
    Always balance performance with code readability and maintainability. Highly optimized code can become unreadable. Document your optimizations and ensure your team understands why they are in place.
- Key takeaway: Performance optimization is never one-and-done. Continuous profiling, benchmarking, and refinement ensure you maintain high performance as your code evolves.

#### Conclusion
- High-performance Python is achievable with the right mindset and toolkit. Start by choosing efficient algorithms and data structures, leverage built-in functions, and profile your code to identify bottlenecks. Consider array-based computation frameworks like NumPy for numeric workloads, and explore Cython, Numba, or PyPy for more specialized performance gains. Take advantage of concurrency, and be mindful of memory patterns. Above all, remember that performance optimization should be guided by data, not guesswork.
- By following these best practices and continuously refining your approach, you can write Python code that meets both your functionality and performance requirements, allowing you to enjoy the productivity benefits of Python without compromising on speed.

## Section 2: Getting Started With This Course

### 3. Expressions and Statements

### 4. Slice sequences
- alist[start:end]
  - start: inclusive index
  - end: exclusive index
```bash
>>> a = [1,2,3,4]
>>> b = a    # shallow copy
>>> c = a[:] # deep copy
>>> id(a), id(b), id(c)
(125483892755264, 125483892755264, 125483892756992)
```

### 5. Some Tips
- alist[start:end:stride]
```bash
>>> a = [1,2,3,4,5,6,7]
>>> print(a[::2])
[1, 3, 5, 7]
>>> print(a[::-1])
[7, 6, 5, 4, 3, 2, 1]
>>> print(a[::-2])
[7, 5, 3, 1]
```

### 6. Preference
- enumerate() to yield the index and element from a list

### 7. Process iterators
- zip(): coupling multiple lists
  - In python2, zip() is not a generator
  - In python3, zip() is a lazy generator
- For different-length lists, zip_longest from itertools may work:
```py
alist = ['a','b','c']
blist = [1,2,3,4]
for a,b in zip(alist,blist):
  print(a,b)
'''
a 1
b 2
c 3
'''  
from itertools import zip_longest
for a,b in zip_longest(alist,blist):
  if a is None:
    print("only", b)
  else:
    print(a,b)
'''    
a 1
b 2
c 3
only 4
'''
```

### 8. Mistakes to avoid
- Do not use `else:` block after `for` loop

### 9. Take advantage of each block
```
try:
    # do something
except MyException as e:
    # handle exception
else:
    # runs when there are not exception
finally:
    # always runs after try:
```    
- Sample code:
```py
import json
UNDEFINED = object()
def divide_json(path):
  handle = open(path,'r+')
  try:
    data = handle.read()
    op = json.loads(data)
    value = op['numerator']/op['denominator']
  except ZeroDivisionError:
    print("ZeroDivisionError")
    return UNDEFINED
  else:
    op['result'] = value
    result = json.dumps(op)
    handle.seek(0)
    handle.write(result)
    print("else block")
    return value
  finally:
    print("finally")
    handle.close()  # always run regardless of return value in else: or other blocks
temp_path = '/tmp/random_data.json'    
with open(temp_path,'w') as handle:
  handle.write('{"numerator":1, "denominator":0}')
print(divide_json(temp_path))  
'''
ZeroDivisionError
finally
<object object at 0x74dd4d28c540>
'''
```

### 10. Contextlib
```py
import logging
def my_function():
  logging.debug("Some debug info")
  logging.error("A real error!")
logging.getLogger().setLevel(logging.WARNING)  
my_function()
#ERROR:root:A real error!
logging.getLogger().setLevel(logging.DEBUG)  
my_function()
#DEBUG:root:Some debug info
#ERROR:root:A real error!
logging.getLogger().setLevel(logging.ERROR)  
my_function()
#ERROR:root:A real error!
```
- @contextmananger can work like `with` block for a generator
- A regular code with `with` block:
```py
with open('/tmp/sample.txt','w') as f:
  f.write("hello_world")
```
- A generator version can be written as:
```py
from contextlib import contextmanager
@contextmanager
def write_manager():
  f = open('/tmp/sample2.txt','w')
  try:
    yield f
  finally:
    f.close()
    print("Completed")
with write_manager() as f:
  f.write("hello_world2")
```

## Section 3: Generators & Comprehensions

### 11. Introduction
- Use list comprehensions instead of map and filter
- Avoid more than two expressions in list comprehension
- Consider generator expressions for large comprehensions
- Consider generators instead of returning lists
- Be defensive when iterating over arguments

### 12. Use list comprehensions
```py
a = [1,2,3,4]
sq_lc = [x**2 for x in a]
sq_map = map(lambda x: x**2,a) # map object is an iterator
print(sq_lc, sq_map)
# [1, 4, 9, 16] <map object at 0x701ee576a4a0>
for el in sq_map:
  print(el)
'''
1
4
9
16
'''  
```  

### 13. Avoid more than two expressions
```py
matrix = [[1,2,3],[4,5,6],[7,8,9]]
flat = [x for row in matrix for x in row ]
print(flat) # [1, 2, 3, 4, 5, 6, 7, 8, 9]
sq_matrix = [[x**2 for x in row] for row in matrix]
print(sq_matrix) # [[1, 4, 9], [16, 25, 36], [49, 64, 81]]
matrix = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
flat = [x for subm in matrix for row in subm for x in row] # quite complicated, and using for loop might be better
print(flat)
#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

### 14. Consider generator expressions for large comprehensions
```py
import random
a = [random.randint(0,100)*'a' for _ in range(10)]
lstapp = [len(x) for x in a]
genexp = (len(x) for x in a) # generator expressions
print(lstapp,genexp)
# [80, 21, 82, 48, 26, 63, 14, 26, 1, 95] <generator object <genexpr> at 0x701ee430b920>
print(next(genexp)) # 80
print(next(genexp)) # 21
```

### 15. Consider generators instead of returning lists
```py
def index_words(text):
  result = []
  if text:
    result.append(0)
  for index, letter in enumerate(text):
    if letter == ' ':
      result.append(index+1)
  return result
result = index_words('Hello world !')  
print(result) # [0, 6, 12]
# Using generator
def index_words(text):
  if text:
    yield 0
  for index, letter in enumerate(text):
    if letter == ' ':
      yield index + 1
it = index_words('Hello world !')        
print(next(it)) # 0
print(next(it)) # 6
print(next(it)) # 12
```

### 16. Be defensive
- Until the generator is exhausted, total sum or aggregation of all results would not be available
  -  Once a generator is started or exhausted, there is no way to restart or rewind
  - A generator is one-shot iterator
- list vs iterator vs generator: conversion
```py
a = [1,2,3]
it = iter(a) # iterator
print(next(it)); print(next(it)) # 1 2
it = a.__iter__() # iterator
print(it.__next__()); print(it.__next__()) # 1 2
a = [1,2,3]
gen = (x for x in a)
print(type(gen),next(gen)) # <class 'generator'> 1
print(type(gen),list(gen)) # <class 'generator'> [2, 3]
```
- `list(generator)` consumes all the left of the generator

## Section 4: Functions

### 17. Introduction
- Know how closures interact with variable scope
- Accept functions for simple interfaces instead of classes
- Reduce visual noise with variable positional arguments
- Provide optional behavior with keyword arguments
- Enforce clarity with keyword-only arguments
- Use None and docstrings to specify dynamic default arguments

### 18. Learn how closures interact with variable scope
- nonlocal keyword in closure
  - Inner functions can use the variable of the outer function
- __init__() vs __call__()
```py
class Test:
  def __init__(self,msg):
    self.msg = msg
  def __call__(self,txt):
    self.txt = txt
    print(self.msg + self.txt)
t = Test("Hello")    
t("World") # HelloWorld
```

### 19. Accept functions for simple interfaces
- defaultdict(): 
  - The first argument is a callable or a function object, which has no argument, setting up the default value
  - Function object: myfunc() -> myfunc
```py
from collections import defaultdict
class CountMissing(object):
  def __init__(self):
    self.added = 0
  def __call__(self):
    self.added += 1
    return 0
counter = CountMissing()
current = {'green':12, 'blue':3}
# helper = lambda : 0
#result = defaultdict(helper,current)
result = defaultdict(counter,current) # not counter() but counter
print(result) # defaultdict(<__main__.CountMissing object at 0x76a580790b20>, {'green': 12, 'blue': 3})
increments = [('red',5),('blue',17), ('orange',9)]
for key, offset in increments:
  result[key] += offset # 'blue' exists already
print(counter.added) # 2, only red and orange are added
print(result) # defaultdict(<__main__.CountMissing object at 0x76a580790b20>, {'green': 12, 'blue': 20, 'red': 5, 'orange': 9})
```

### 20. Reduce visual noise
- Variable length positional argument in a function using "*"
```py
def log(msg, *val):
  if not val:
    print(msg)
  else:
    vstr = ', '.join(str(x) for x in val)
    print(msg + vstr)
log("hello ", 1,2,3) # hello 1, 2, 3
v = [1,2,3]
log("hello ", v)     # hello [1, 2, 3]
log("hello ", *v)    # hello 1, 2, 3, *v will upack v
def mygen():
  for i in range(5):
    yield i
log("Hello ", mygen())   # Hello <generator object mygen at 0x77774eae0a50>
log("Hello ", *mygen())  # Hello 0, 1, 2, 3, 4
```

### 21. Provide optional behavior
- Using function argument keys
```py
def myfunc(a,b):
  return a+b
myfunc(1,2)
myfunc(a=1,b=2)
#myfunc(a=1,2) # not working
myfunc(1,b=2)
myfunc(b=2,a=1)
#myfunc(b=2,1) # not working
#myfunc(2,a=1) # not working
```
- Default argument:
```py
def test_default(a,b=123):
  return a+b
test_default(1,2) # 3
test_default(a=1,b=2) # 3
test_default(a=1) # 124
test_default(1)   # 124
```

### 22. Enforce clarity with keyword-only arguments
- Using "*"
  - myfunc(a,b, *, c, d)
  - c & d need keywords for function call
- Using "*args"
  - myfunc(a,b, *args, c,d)
  - c & d need keywords for function call
- This is not supported in Python2 and Python2 may use **kwargs
```py
def myf1(a,b,c,d):
  return a+b+c+d
myf1(1,1,1,1)        # works OK
myf1(a=1,b=1,c=1,d=1)# works OK
def myf2(a,b,*,c,d):
  return a+b+c+d
# myf2(1,1,1,1)       # breaks
myf2(a=1,b=1,c=1,d=1) # works OK
myf2(1,1,c=1,d=1)     # works OK
# myf2(1,1,1,d=1)     # breaks
def myf3(a,b,*args,c,d):
  return a+b+c+d
myf3(a=1,b=1,c=1,d=1) # works OK
myf3(1,1,c=1,d=1)     # works OK
#myf3(1,1,1,d=1)      # breaks
# For python2
def myf4(a,b,**kwargs): # kwargs is a dictionary
  c = kwargs['c']
  d = kwargs['d']
  return a+b+c+d
myf4(a=1,b=1,c=1,d=1)
myf4(1,1,c=1,d=1)
#
def safe_div(n,d, ignore_overflow, ignore_zero_div):
  try:
    return n/d
  except OverflowError:
    if ignore_overflow:
      return 0
    else:
      raise
  except ZeroDivisionError:
    if ignore_zero_div:
      return float('inf')
    else:
      raise
#safe_div(1., 10**500, False, False) # OverflowError: int too large to convert to float
#safe_div(1., 0, False, False) # ZeroDivisionError: float division by zero
```

### 23. Specify dynamic default arguments
```py
from time import sleep
from datetime import datetime
def log(msg, when=datetime.now()):
  print(msg+"@" + str(when))
log("hello") # hello@2026-05-16 20:52:46.986313
sleep(1)
log("hello") # hello@2026-05-16 20:52:46.986313
```
- log() prints the same timestamp as `when=datetime.now()` is executed when the function is defined
- We need to measure the timestamp dynamically
```py
from time import sleep
from datetime import datetime
def log(msg, when=None):
  if when is None:
    when = datetime.now()
  print(msg+"@" + str(when))
log("hello") # hello@2026-05-16 20:54:31.961321
sleep(1)
log("hello") # hello@2026-05-16 20:54:32.962362
```
- Now 1sec difference is found
- When a default argument is returned:
```py
import json
def decode(data,val={}):
  try:
    return json.loads(data)
  except ValueError:
    return val
foo = decode('bad data') # will return {}
print(foo, id(foo)) # {} 131354797982720
foo['sample'] = 5 
print(foo, id(foo)) # {'sample': 5} 131354797982720
bar = decode('another')
print(bar, id(bar)) # {'sample': 5} 131354797982720
bar['hello'] = 8
print(bar, id(bar)) # {'sample': 5, 'hello': 8} 131354797982720
print(foo, id(foo)) # {'sample': 5, 'hello': 8} 131354797982720
```
- When decode() returns, it will return the same object, which is created when the function is defined. Will be aggregated as another function call is made
- In order to avoid the same object, default value must be created when the function runs dynamically
```py
import json
def decode(data,val=None):
  if val is None:
    val = {} 
  try:
    return json.loads(data)
  except ValueError:
    return val
foo = decode('bad data') # will return {}
print(foo, id(foo)) # {} 131354301808128
foo['sample'] = 5 
print(foo, id(foo)) # {'sample': 5} 131354301808128
bar = decode('another')
print(bar, id(bar)) # {} 131354797772160
bar['hello'] = 8
print(bar, id(bar)) # {'hello': 8} 131354797772160
print(foo, id(foo)) # {'sample': 5} 131354301808128
```
- Now foo and bare are different objects

## Section 5: Classes

### 24. Introduction
- Prefer help classes over book-keeping with dictionaries and tuples
- Use plain attributes instead of get/set methods
- Prefer public attributes over private ones
- Use @classmethod polymorphism to construct objects generically

### 25. Prefer helper classes over bookkeeping
- The instructor claims: With many classes and good abstractions and good interfaces, it's really easy to expand and understand. And so when you when you see yourself using a lot of dictionaries and tuples, considered refactoring your code into classes so that you have something that's better.

### 26. Use plain attributes instead of get and set methods
- A class using setter/getter:
```py
class oldR(object):
  def __init__(self, ohms):
    self.ohms = ohms
  def get_ohms(self):
    return self.ohms
  def set_ohms(self,ohms):
    self.ohms = ohms
r0 = oldR(50e3)
r0.set_ohms(r0.get_ohms() + 5e3)
```
- In pythonic way:
```py
class pyR(object):
  def __init__(self,ohms):
    self.ohms = ohms # no data validation
r1 = pyR(50e3)    
r1.ohms += 5e3
print(r1.ohms)
```
- This code may accept negative resistance value, which is not valid
```py
class pyR(object):
  def __init__(self,val):
    self.rsst = val # this calls the setter below
  @property
  def ohms(self): # note that the name of the function is ohm while the actual attribute name is rsst
    return self.rsst
  @ohms.setter
  def ohms(self, val):
    if val <= 0:
      raise ValueError('%f R must be >0'% val)
    self.rsst = val
r1 = pyR(1000)    
# r1.ohms(1000e3) This is illegal
r1.ohms = 1000e3 # This is allowed
r1.ohms += -1000e3 # Value error as expected
print(r1.ohms)
```
- `@property`: defines a method but accesses it like an attribute. This works like a getter
- `@<name>.setter`: When an value is assigned to the attribute, the method below is called, working like a setter
  - Type checker might be implemented
  - __init__() calls the function below for initialization
  - The function below is NOT callable
- `@<name>.deleter`: works like `del` keyword

### 27. Prefer public attributes over private ones
```py
class MyTest(object):
  def __init__(self):
    self.public_data = 123
    self.__private_data = 456
  def return_private(self):
    return self.__private_data
t1 = MyTest()
print(t1.public_data)     # 123
#print(t1.__private_data) # 'MyTest' object has no attribute '__private_data'
print(t1.return_private())# 456
```
- `__` triggers the variable as private

### 28. Learn to Use @classmethod polymorphism
- Ref: https://medium.com/@kqy7yu/python-polymorphic-constructors-and-utility-functions-the-staticmethod-and-classmethod-advantages-ec53282e4cbf
- Ref: https://gist.github.com/kyoungrok0517/4e08b4bbf187bcf6edad38107bca8632
  - In Python, __init__() is allowed only once
  - To overload class construct like c++, we may use @classmethod, employing different construct argument


## Section 6: Parllelism & Concurrency

### 29. Introduction
- use subprocess to manage child processes
- Use threads for blocking IO, avoid for parallelism
- Use Lock to prevent data races in threads
- Use Queue to coordinate work b/w threads
- Consider concurrent, futures for true parallelism

### 30. Learn to Use subprocess to manage child processes
- ?

### 31. Learn to Use threads for blocking I/O, avoid for parallelism
```py
import time
def factorize(number):
  for i in range(1, number+1):
    if number%i == 0:
      yield i
numbers = [2139079,1232472,3213932,2314921]
start = time.time()
for n in numbers:
  list(factorize(n))
end = time.time()
print('Took %.3f' % (end-start)) # 0.299
```
- Parallelize the above python code
```py
import time
import threading
def factorize(number):
  for i in range(1, number+1):
    if number%i == 0:
      yield i
class FactorizeThread(threading.Thread):
  def __init__(self,number):
    super().__init__()
    self.number = number
  def run(self):
    self.factor = list(factorize(self.number))
numbers = [2139079,1232472,3213932,2314921]
start = time.time()
threads = []
for n in numbers:
  thread = FactorizeThread(n)
  thread.start()
  threads.append(thread)
for thread in threads:
  thread.join()
end = time.time()
print('Took %.3f' % (end-start)) # 0.314 - slower than above
```
- Actually this is slower than serial code
  - Too many syscalls?

### 32. Learn to Use Lock to prevent data races in threads
```py
import threading
class Counter(object):
  def __init__(self):
    self.count = 0
    self.lock = threading.Lock()
  def increment(self,offset):
    with self.lock:
      self.count += offset
worker_count = 5
barrier = threading.Barrier(worker_count)    
def worker(sensor_index, how_many, counter):
  barrier.wait()
  for _ in range  (how_many):
    counter.increment(1)
threads = []
how_many = 1000000
counter = Counter()
for i in range(5):
  args = (i,how_many, counter)
  thread = threading.Thread(target=worker,args=args)
  thread.start()
  threads.append(thread)
for thread in threads:
  thread.join()
print(counter.count)
```

### 33. Learn to Use Queue to coordinate work between threads
```py
from queue import Queue
from threading import Thread
import time
queue = Queue(1)
def consumer():
  time.sleep(0.1)
  queue.get()
  print('Consumer got 1')
  queue.get()
  print('Consumer got 2')
thread = Thread(target=consumer)  
thread.start()
print('Producer putting')
queue.put(object())
print('Producer put 1')
queue.put(object())
print('Producer put 2')
queue.put(object())
thread.join()
print('Producer done')
'''
Producer putting
Producer put 1
Consumer got 1
Producer put 2
Consumer got 2
Producer done
'''
from queue import Queue
from threading import Thread
in_queue = Queue()
def consumer():
  print('Consumer waiting')
  work = in_queue.get()
  print('Consumer working')
  print('Consumer done')
  in_queue.task_done()
thread = Thread(target=consumer)
thread.start()
in_queue.put(object())
print('Producer is waiting')
in_queue.join()
print('Producer is done')
'''
Consumer waiting
Producer is waiting
Consumer working
Consumer done
Producer is done
'''
from queue import Queue
from threading import Thread
class ClosableQueue(Queue):
  SENTINEL = object()
  def close(self):
    self.put(self.SENTINEL)
  def __iter__(self):
    while True:
      item = self.get()
      try:
        if item is self.SENTINEL:
          return
        yield item
      finally:
        self.task_done()
class StoppableWorker(Thread):
  def __init__(self,func,in_queue, out_queue):
    super().__init__()
    self.func = func
    self.in_queue = in_queue
    self.out_queue = out_queue
  def run(self):
    for item in self.in_queue:
      result = self.func(item)
      self.out_queue.put(result)
def download(item):
  return item
def resize(item):
  return item
def upload(item):
  return item
download_queue = ClosableQueue()
resize_queue = ClosableQueue()
upload_queue = ClosableQueue()
done_queue = ClosableQueue()
threads = [
  StoppableWorker(download, download_queue,resize_queue),
  StoppableWorker(resize, resize_queue, upload_queue),
  StoppableWorker(upload, upload_queue, done_queue),
]
for thread in threads:
  thread.start()
for _ in range(1000):
  download_queue.put(object())
download_queue.close()
download_queue.join()
resize_queue.close()
resize_queue.join()
upload_queue.close()
upload_queue.join()
print(done_queue.qsize(), 'item finished')
```

### 34. Consider concurrent.futures for true parallelism
- A sample serial code:
```py
def gcd(pair):
  a,b = pair
  low = min(a,b)
  for i in range(low, 0, -1):
    if a%i ==0 and b%i == 0:
      return i
  return 1
print(gcd((128,76))) # 4
n = [(1963309,2265973),(2030677,3814172),
     (1551645,2229620),(2039045,2020802)]
import time
start = time.time()
result = list(map(gcd, n))
end = time.time()
print(end-start) # 0.26
```
- A sample multi-threads code:
```py
from concurrent.futures import ThreadPoolExecutor
def gcd(pair):
  a,b = pair
  low = min(a,b)
  for i in range(low, 0, -1):
    if a%i ==0 and b%i == 0:
      return i
  return 1
print(gcd((128,76))) # 4
n = [(1963309,2265973),(2030677,3814172),
     (1551645,2229620),(2039045,2020802)]
pool = ThreadPoolExecutor(max_workers=2)
import time
start = time.time()
result = list(pool.map(gcd, n))
end = time.time()
print(end-start) # 0.25
```
  - Almost same speed
- A sample multi-processes code
```py
from concurrent.futures import ProcessPoolExecutor
def gcd(pair):
  a,b = pair
  low = min(a,b)
  for i in range(low, 0, -1):
    if a%i ==0 and b%i == 0:
      return i
  return 1
print(gcd((128,76))) # 4
n = [(1963309,2265973),(2030677,3814172),
     (1551645,2229620),(2039045,2020802)]
pool = ProcessPoolExecutor(max_workers=2)
import time
start = time.time()
result = list(pool.map(gcd, n))
end = time.time()
print(end-start) # 0.17
```
  - Now 1.5x faster
- How multiprocessing works
  - Takes each item from the n input data to map
  - Serializes it into binary data using the pickle module
  - Copies the serialized data from the main interpreter process to a child interpreter proces over a local socket
  - Deserializes the data back into Python objects using pickle in the child processes
  - Imports the Python module containing the gcd function
  - Runs the function on the input data in parallel with other child processes
  - Serializes the result back into bytes
  - Copies those bytes back through the socket
  - Deserializes the bytes back into Python objects in the parent process
  - Merges the results from multiple children into a single list to return
- When to use multiprocessing
  - Isolated
    - Functions that don't share data
  - High leverage
    - Small amount of data -> large amount of computation

### 35. Python Concurrency Approaches
- async/await and concurrent.futures are two different approaches for achieving concurrency in Python. Both methods are used to run tasks concurrently, allowing you to improve the performance of your programs by executing tasks in parallel, especially when tasks are I/O-bound.
  - async/await: This is a more modern approach to concurrency introduced in Python 3.5, built on top of the asyncio library. It allows you to write asynchronous code using a more readable and concise syntax. With async/await, you define coroutines using the async def keyword, and you can use await to pause the execution of the coroutine until a specific operation is completed. The asyncio event loop handles the execution of these coroutines concurrently.
- Here's a basic example of using async/await:
```py
    pythonCopy code
    import asyncio
     
    async def my_coroutine():
        print("Start")
        await asyncio.sleep(1)
        print("End")
     
    async def main():
        tasks = [my_coroutine() for _ in range(3)]
        await asyncio.gather(*tasks)
     
    asyncio.run(main())
```
  - concurrent.futures: This is an older approach to concurrency, introduced in Python 3.2. It provides a high-level interface for asynchronously executing callables using thread or process-based parallelism. The ThreadPoolExecutor and ProcessPoolExecutor classes are used to manage the execution of tasks in a pool of worker threads or processes, respectively.
- Here's a basic example using concurrent.futures with a thread pool:
```py
    pythonCopy code
    import concurrent.futures
    import time
     
    def my_function():
        print("Start")
        time.sleep(1)
        print("End")
     
    def main():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = [executor.submit(my_function) for _ in range(3)]
            concurrent.futures.wait(tasks)
     
    main()
```
- In general, async/await is more suitable for I/O-bound tasks, such as networking or file I/O, whereas concurrent.futures can be used for both I/O-bound and CPU-bound tasks. When using concurrent.futures, you should choose the appropriate executor based on the nature of the tasks you're running (thread-based for I/O-bound tasks and process-based for CPU-bound tasks)

## Section 7: Learn How To Make Your Programs Robust

### 36. Introduction
- Use virtual environments isolated and reproducible dependencies
- Consider interactive debugging with pdb
- Profile before optimization
- Use tracemalloc to understand memory usage and leaks

### 37. Learn to Use virtual environments
~~- pyvenv -h~~ This command is obsolete since 3.8
- python -m venv myproject 
- cd myproject
- source bin/activate
  - Now you're on venv

### 38. Test with unittest
- There is no static type checking in Python
```py
from unittest import TestCase, main
from tempfile import TemporaryDirectory
class UtilsTestCase(TestCase):
  def setUp(self):
    self.test_dir = TemporaryDirectory()
  def tearDown(self):
    self.test_dir.cleanup()
  def test_a(self):
    print(self.test_dir)
  def test_addition(self):
    self.assertEqual(1+1, 2)
if __name__ == '__main__':
  main()
```
- Demo:
```bash
$ python3 ch38.py 
<TemporaryDirectory '/tmp/tmprpwzxe_9'>
..
----------------------------------------------------------------------
Ran 2 tests in 0.001s

OK
```
- When to write tests
  - One TestCase for each set of related tests
  - One TestCase for testing a single class its methods
  - Unit tests: Testing functionality in isolatin
  - Integration tests: Verifying interactions b/w functionality
  - Need to write both types of tests

### 39. Consider interactive debugging with pdb
- Inject `import pdb; pdb.set_trace()` into ROI
- A sample code:
```py
def bubble_sort(data):
  for _ in range(len(data)):
    one_pass(data)
def one_pass(data):
  for i in range(len(data)):
    compare_and_swap(data,i)
def compare_and_swap(data,i):
  import pdb; pdb.set_trace() # <-- inject pdb here
  if data[i]>data[i+1]:
    data[i],data[i+1] = data[i+1],data[i]
from random import randint
data = [randint(0,1000) for _ in range(10)]
print(data)
bubble_sort(data)
print(data)
```
- Demo:
```bash
python3 ch39.py 
[843, 737, 363, 2, 486, 387, 949, 334, 466, 790]
> /.../udemy_pyHPC/myproject/ch39.py(9)compare_and_swap()
-> if data[i]>data[i+1]:
(Pdb) i
0
(Pdb) bt
  /.../udemy_pyHPC/myproject/ch39.py(14)<module>()
-> bubble_sort(data)
  /.../udemy_pyHPC/myproject/ch39.py(3)bubble_sort()
-> one_pass(data)
  /.../udemy_pyHPC/myproject/ch39.py(6)one_pass()
-> compare_and_swap(data,i)
> /.../udemy_pyHPC/myproject/ch39.py(9)compare_and_swap()
-> if data[i]>data[i+1]:
(Pdb) up
> /.../udemy_pyHPC/myproject/ch39.py(6)one_pass()
-> compare_and_swap(data,i)
(Pdb) bt
  /.../udemy_pyHPC/myproject/ch39.py(14)<module>()
-> bubble_sort(data)
  /.../udemy_pyHPC/myproject/ch39.py(3)bubble_sort()
-> one_pass(data)
> /.../udemy_pyHPC/myproject/ch39.py(6)one_pass()
-> compare_and_swap(data,i)
  /.../udemy_pyHPC/myproject/ch39.py(9)compare_and_swap()
-> if data[i]>data[i+1]:
(Pdb) data
[843, 737, 363, 2, 486, 387, 949, 334, 466, 790]
(Pdb) down
> /.../udemy_pyHPC/myproject/ch39.py(9)compare_and_swap()
-> if data[i]>data[i+1]:
(Pdb) continue
> /.../udemy_pyHPC/myproject/ch39.py(9)compare_and_swap()
-> if data[i]>data[i+1]:
```
- If some condition is required (like i==9), you can inject like:
```py
    if (i==9):
      import pdb; pdb.set_trace() 
```      

### 40. Profile before optimizing
- Dynamic behavior has surprising performance impact
- Profiling is easy to do using built-in modules
- Focus on measurable sources of trouble
- A sample code:
```py
from random import randint
from cProfile import Profile
from pstats import Stats
def insertion_sort(data):
  result = []
  for value in data:
    insert_value(result,value)
  return result
def insert_value(array,value):
  for i, existing in enumerate(array):
    if existing > value:
      array.insert(i,value)
      return
  array.append(value)
max_size = 10000
data = [randint(0,max_size) for _ in range(max_size)]
profiler = Profile()
profiler.runcall(lambda: insertion_sort(data))
print('Done')
stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats()
```
- Result
```
Done
         20003 function calls in 0.900 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.900    0.900 3307648883.py:18(<lambda>)
        1    0.002    0.002    0.900    0.900 3307648883.py:4(insertion_sort)
    10000    0.894    0.000    0.899    0.000 3307648883.py:9(insert_value)
     9989    0.005    0.000    0.005    0.000 {method 'insert' of 'list' objects}
       11    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```
- Or try `stats.print_callers()` to see cumulative time
```
Done
   Ordered by: cumulative time

Function                                          was called by...
                                                      ncalls  tottime  cumtime
3258165172.py:18(<lambda>)                        <- 
3258165172.py:4(insertion_sort)                   <-       1    0.002    0.906  3258165172.py:18(<lambda>)
3258165172.py:9(insert_value)                     <-   10000    0.899    0.904  3258165172.py:4(insertion_sort)
{method 'insert' of 'list' objects}               <-    9988    0.005    0.005  3258165172.py:9(insert_value)
{method 'append' of 'list' objects}               <-      12    0.000    0.000  3258165172.py:9(insert_value)
{method 'disable' of '_lsprof.Profiler' objects}  <- 
```

### 41. Learn to Use tracemalloc to understand memory usage and leaks
- Python uses reference counting for garbage collection
- Cycle detector for looping references
  - In practice, hard to figure out why references are held
- waste_memory.py
```py
import os
import hashlib
class MyObj(object):
  def __init__(self):
    self.x = os.urandom(100)
    self.y = hashlib.sha1(self.x).hexdigest()
def get_data():
  values = []
  for _ in range(100):
    obj = MyObj()
    values.append(obj)
  return values
def run():
  deep_values = []
  for _ in range(100):
    dimport os
import hashlib
class MyObj(object):
  def __init__(self):
    self.x = os.urandom(100)
    self.y = hashlib.sha1(self.x).hexdigest()
def get_data():
  values = []
  for _ in range(100):
    obj = MyObj()
    values.append(obj)
  return values
def run():
  deep_values = []
  for _ in range(100):
    deep_values.append(get_data())
  return deep_values
eep_values.append(get_data())
  return deep_values
```
- Garbage collection:
```py
import gc
found_objects = gc.get_objects()
print('%d object before' % len(found_objects))
import waste_memory
x = waste_memory.run()
found_objects = gc.get_objects()
print('%d object after ' % len(found_objects))
for obj in found_objects[:3]:
  print(repr(obj)[:100])
'''
201989 object before
212128 object after  # 212128 > 201989
<waste_memory.MyObj object at 0x7722a3d28850>
<waste_memory.MyObj object at 0x7722a3d288b0>
<waste_memory.MyObj object at 0x7722a3d28910>
'''
```
- Tracemalloc:
```py
import tracemalloc
tracemalloc.start(10)
t1 = tracemalloc.take_snapshot()
import waste_memory
x = waste_memory.run()
t2 = tracemalloc.take_snapshot()
stats = t2.compare_to(t1,'lineno')
for stat in stats[:3]:
  print(stat)
'''
/.../udemy_pyHPC/waste_memory.py:5: size=2314 KiB (+2314 KiB), count=29988 (+29988), average=79 B
/.../udemy_pyHPC/waste_memory.py:6: size=869 KiB (+869 KiB), count=10000 (+10000), average=89 B
/.../udemy_pyHPC/waste_memory.py:10: size=469 KiB (+469 KiB), count=10000 (+10000), average=48 B
'''
import tracemalloc
tracemalloc.start(10)
t1 = tracemalloc.take_snapshot()
import waste_memory
x = waste_memory.run()
t2 = tracemalloc.take_snapshot()
stats = t2.compare_to(t1,'traceback')
top = stats[0] # The highest only
print('\n'.join(top.traceback.format()))
'''
  File "/.../anaconda3/2023.03/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 2961
    result = self._run_cell(
  File "/.../anaconda3/2023.03/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3016
    result = runner(coro)
  File "/.../anaconda3/2023.03/lib/python3.10/site-packages/IPython/core/async_helpers.py", line 129
    coro.send(None)
  File "/.../anaconda3/2023.03/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3221
    has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
  File "/.../anaconda3/2023.03/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3400
    if await self.run_code(code, result, async_=asy):
  File "/.../anaconda3/2023.03/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3460
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "/tmp/ipykernel_12478/3480769391.py", line 5
    x = waste_memory.run()
  File "/.../udemy_pyHPC/waste_memory.py", line 16
    deep_values.append(get_data())
  File "/.../udemy_pyHPC/waste_memory.py", line 10
    obj = MyObj()
  File "/.../udemy_pyHPC/waste_memory.py", line 5
    self.x = os.urandom(100)
'''  
```

## Section 8: Outro

### 42. Course Summary

## Section 9: Source Code

### 43. Source Code

### 44. You Are Now a High-Performance Developer!
