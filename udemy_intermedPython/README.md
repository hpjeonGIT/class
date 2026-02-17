## Intermediate Python: Memory, Decorator, Async, Cython & more
- Instructor: Jan Schaffranek

## Section 1: Chapter 1: Introduction and Software

### 1. Introduction to the course

### 2. Course manual
- https://github.com/franneck94/UdemyPythonProEng
- Python 3.8 or newer
  - 3.10 is recommended in this class

### 3. Course materials
- Github Repository: https://github.com/franneck94/UdemyPythonInt
- Downloads:
  - Anaconda (optional): https://www.anaconda.com/downloads   
  - VS Code: https://code.visualstudio.com/
- Installations Commands: 
  - conda create -n pyUdemy python=3.10
  - pip install -r requirements.txt
- VS Code Extensions:
  - Python Dev Extension Pack
  - Coding Tools Extension Pack

### 4. The creation of the environment
- pip install -r requirements.txt

### 5. Visual Studio Code Setup

## Section 2: Chapter 2-0: Python Pro 101

### 6. Simple Type Annotations
- Since Python 3.5
```py
def f1(a: float) -> float:
  return a*3.0
```
  - Argument type overloading like C++ is not supported
```py
from typing import Union
def f1(a: dict) -> dict:
  return a
def f2(a: int | float) -> int | float:
  return a*11
def f3(a: Union[int, float]) -> int | float:
  return a*3
print(f1([1,2,'a']), f2(3), f3(4))
# 1, 2, 'a'] 33 12
```

### 7. f-Strings - Part 1
- Strin concatenation is sluggish
- Using f-string is the fastest and easy
```py
def concat_string(name: str , age: str) -> str:
  return "Hello my name is " + name + " and age is " + str(age)
def old_format(name: str, age: str) -> str:
  return "Hello my name is %s  and age is %d" % (name,age)
def f_format(name: str, age: str) -> str: # this is the fastest
  return f"Hello my name is {name}  and age is {age}" 
# print(concat_string("John", 22), old_format("John", 22), f_format("John", 22))  
```

### Quiz 1: Python Pro 101

## Section 3: Chapter 2-1: Numeric values

### 8. Integers
```py
import sys
x = 123
print(sys.getsizeof(x)) # 28, class size, not int type in bytes
x = 1_234_567_890
print(sys.getsizeof(x)) # 32 bytes
x = 4561_234_567_890
print(sys.getsizeof(x)) # still 32 bytes
```
- After division, int/int becomes float
```py
result = 2+1; print(result, type(result)) # 3 <class 'int'>
result = 2*1; print(result, type(result)) # 2 <class 'int'>
result = 2/1; print(result, type(result)) # 2.0 <class 'float'>
result = 3//2; print(result, type(result)) # 1 <class 'int'>
```
- Different int base
```py
my_dec = int("42", base=10)
my_oct = int("42", base=8)
print(my_dec, my_oct) # 42 34
```

### 9. Floats
- Python float (64 bits=8 Byte)
  - sign: 1bit
  - exponent: 11bits
  - significant bits: 52bits
- Do not use equal test on floating numbers !!!
```py
x = 0.123456789
print(f"{x:.20f}", x)
# 0.12345678899999999734 0.123456789
print(round(421.23456,2)) # 421.23
print(round(421.23456,-1)) # 420.0
print(round(421.23456,-2)) # 400.0
421.2346
420.0
400.0
```

### Quiz 2: Integers und Floats

## Section 4: Chapter 2-2: Logical expression

### 10. Booleans
```py
my_b = False
print(sys.getsizeof(my_b)) # 28
my_b = True
print(sys.getsizeof(my_b)) # 28
print(issubclass(bool,int)) # True
print(isinstance(True,bool)) # True
print(isinstance(False,bool)) # True
print(isinstance(1,bool)) # False
print(isinstance(0,bool)) # False
```
- is vs equal
  - `is`/`is not`: memory equality
  - `==`/`!=`: value equality
```py
print( 1 == True) # True
print( 1 is True) # False
my_b1 = True
my_b2 = True
print(my_b1 is my_b2) # True
print(my_b1 == my_b2) # True
```
- An instance is True if it is not None, False, 0, empty (list)
```py
v1 = None; print(bool(v1)) # False
v1 = False; print(bool(v1)) # False
v1 = 0; print(bool(v1)) # False
v1 = []; print(bool(v1)) # False
v1 = True; print(bool(v1)) # True
v1 = 1; print(bool(v1)) # True
v1 = ['a']; print(bool(v1)) # True
```
- Shortcut trick in Python
  - `if A and B` will not check B if A is False
  - Skips B, regardless of the definition
  - Saves the time of running B
```py
def f1(a):
  print("F1 is called")
  return a
def f2(b):
  print("F2 is called")
  return b
if f1(True) and f2(False):
  print("Final called")
'''
F1 is called
F2 is called
'''  
if f1(False) and f2(True):
  print("Final called")
'''
F1 is called # f2() is not called
'''
```    

### 11. Match-Statement
```py
def pattmatch(a: int) -> None:
  match a:
    case a if a < 0:
      print(f"{a} is not valid")
    case 2:
      print(f"{a} is 2")
    case 3:
      print(f"{a} is 3")
    case _:
      print(f"{a} IS ?")
pattmatch(-1) # -1 is not valid
pattmatch(1) # 1 IS ?
pattmatch(2) # 2 is 2
```

### Quiz 3: Logical expressions

## Section 5: Chapter 2-3: Memory management

### 12. Variables and references
- Python uses the same memory address for immutable objects (int, bool, None)
  - Saving memory footprint
  - There is only one None object !!!
```py
def print_memory_address(var):
  print(hex(id(var)))
my_v = 10
print_memory_address(my_v) # 0x938250
my_v2 = 12
print_memory_address(my_v) # 0x938290 - different address than above
my_v3 = 42 # immmutable object
my_v4 = 42
my_v5 = 42
print_memory_address(my_v3) # 0x938650 - all same addresses !!!
print_memory_address(my_v4) # 0x938650
print_memory_address(my_v5) # 0x938650
```
- For float, they point differnt addresses (no float interning)
  - Float is immutable but not Singleton
```py
my_f1 = 1.23
my_f2 = 1.23
print_memory_address(my_f1) # 0x7ad9f95d6a90
print_memory_address(my_f2) # 0x7ada16638310
```

### 13. Mutability of data types
- Immutable types: int, float, bool, str, tuple, None
- Mutable types: list, dict, set, etc
  - Editing/appending the object still points the same memory address

### 14. In-Place-Operations and Shallow/Deep Copy
- In-place operation: directly changes the content of an object without creating a new one
  - Mutable objects will keep the address, as new object is not made
```py
def concat_list(l1,l2)-> list:
  return l1+l2 # no-inplace. Return new object
def concat_list_inplace(l1,l2):
  l1 += l2 # inplace operation
l1 = [1,2]
l2 = [3,4]
print_memory_address(l1) # 0x7ad9f83aa9c0
l1 = concat_list(l1,l2)
print(l1)
print_memory_address(l1) # 0x7ad9f83aaa00 - list changed
#
l1 = [1,2]
l2 = [3,4]
print_memory_address(l1) # 0x7ad9f83a9300
concat_list_inplace(l1,l2)
print(l1)
print_memory_address(l1) # 0x7ad9f83a9300 - same address
```
- Shallow and deep copy
```py
import copy
l1 = [[1,2],[3,4]]
l2 = l1
print_memory_address(l1) # 0x7ad9f83c3ec0
print_memory_address(l2) # 0x7ad9f83c3ec0 - same
print_memory_address(l1[0]) # 0x7ad9f83c2840
print_memory_address(l2[0]) # 0x7ad9f83c2840
print("#shallow copy")
l3 = copy.copy(l1) # shallow copy
print_memory_address(l3) # 0x7ad9f83f0800 - new
print_memory_address(l3[0]) # 0x7ad9f83c2840 - same to l1[0]
print("#deep copy")
l4 = copy.deepcopy(l1) # deep copy
print_memory_address(l4) # 0x7ad9f83e5c40 - new
print_memory_address(l4[0]) # 0x7ad9f83f12c0 - new
```

### Quiz 4: Memory management

## Section 6: Chapter 3-1: Container

### 15. Lists
```py
from typing import Any
from typing import List
from typing import Union
def memory_address(var: Any):
  return hex(id(var)%0xFFFF)
def print_list_info(lst: Union[List[Any], List[List[Any]]]):
  print(f"List address: {memory_address(lst)}")
  if len(lst) == 0:
    return
  if not isinstance(lst[0], list):
    return
  for i in range(len(lst)):
    print(f"List[{i}]: {memory_address(lst[i])}")
    if isinstance(lst[i], list):
      for j in range(len(lst[i])):
        print(f"List[{i},{j}]: {memory_address(lst[i][j])}")
  print("\n")
my_l = [1,2,3]
print(my_l)
print_list_info(my_l)
# [1, 2, 3]
# List address: 0x446  
my_l2 = [[1,2],[3]]
print(my_l2)
print_list_info(my_l2)
'''
[[1, 2], [3]]
List address: 0x2233
List[0]: 0x2c33
List[0,0]: 0x81c3
List[0,1]: 0x81e3
List[1]: 0xa471
List[1,0]: 0x8203
'''
my_l2[1] = [3,4]
print(my_l2)
print_list_info(my_l2)
'''
[[1, 2], [3, 4]]
List address: 0x2233
List[0]: 0x2c33 # same address still
List[0,0]: 0x81c3 
List[0,1]: 0x81e3
List[1]: 0x178e # new address
List[1,0]: 0x8203
List[1,1]: 0x8223
'''
```

### 16. Tuples
- Tuple is immutable while list is mutable
  - (10) is NOT a tuple
  - (10,) is a tuple: coman(,) is the primary tuple constructor
```py
from typing import Any
from typing import Tuple
from typing import Union
def memory_address(var: Any):
  return hex(id(var)%0xFFFF)
def print_tuple_info(tpl: Tuple[Any]):
  print(f"Tuple address: {memory_address(tpl)}")
  for val in tpl:
    print(f"Val: {memory_address(val)}")
  print("\n")
my_t2 = (10,20,30)
print(my_t2)
print_tuple_info(my_t2)
'''
(10, 20, 30)
Tuple address: 0x1bf6
Val: 0x82e3
Val: 0x8423
Val: 0x8563
'''  
# my_t2[0] = 3 ; invalid operation as tuple's appress is fixed
my_l = [1]
my_t2 = (10,20,my_l)
print(my_t2)
print_tuple_info(my_t2)
'''
(10, 20, [1])
Tuple address: 0x9a3a
Val: 0x82e3
Val: 0x8423
Val: 0xcf32
'''
my_t2[2].append(2) # can edit tuple element
print(my_t2)
print_tuple_info(my_t2)
'''
(10, 20, [1, 2])
Tuple address: 0x9a3a
Val: 0x82e3
Val: 0x8423
Val: 0xcf32 <--- address is still same
'''
```
- We can change the tuple's element but still the address is same
- Tuple packing & unpacking
```py
tpl = (1,2,3, True, False)
m1,m2,m3,m4,m5 = tpl
print(m1,m2,m3,m4,m5) # 1 2 3 True False
a1,_,a3,_, a5 = tpl
print(a1,a3,a5) # 1 3 False
b1,b2,*b3 = tpl
print(b3) # [3, True, False]
```

### Quiz 5: Lists and Tuples

### 17. Dictionaries
- Types of keys: bool, int, float, str, tuple, Non (immutable types)
- Types for values: any type
- For each item, the dictionary calculates a hash of the key based on its content. If the value changes, the hash will change
```py
# Dictionary comprehension
my_d1 = { i: i*3 for i in range (3)}
print(my_d1) # {0: 0, 1: 3, 2: 6}
# Dictinary merging
myd1 = {'a':1, 'b':2}
myd2 = {'c':3, 'd':7}
# < v3.9
myd3 = {**myd1, **myd2}
print(myd3) # {'a': 1, 'b': 2, 'c': 3, 'd': 7}
# >= v3.9
myd4 = myd1 | myd2
print(myd4) # {'a': 1, 'b': 2, 'c': 3, 'd': 7}
```
- If a key reappears in the other dictionary, value is overwritten

### 18. Sets
```py
from typing import Any
from typing import Set
def memory_address(var: Any):
  return hex(id(var)%0xFFFF)
def print_set_info(st: Tuple[Any]):
  print(f"Set address: {memory_address(st)}")
  for val in st:
    print(f"Val: {memory_address(val)}")
  print("\n")
my_s1 = {1,2,3,1}
print(my_s1)
print_set_info(my_s1)
'''
{1, 2, 3}
Set address: 0x52cc
Val: 0x81c3
Val: 0x81e3
Val: 0x8203
'''
```

### Quiz 6: Dictionaries and Sets

## Section 7: Chapter 3-2:  Strings, Files and f-Strings

### 19. Strings
```py
def memory_address(var: Any):
  return hex(id(var)%0xFFFF)
def print_string_info(st: str):
  print(f"String address: {memory_address(st)}")
  for val in st:
    print(f"Val: {memory_address(val)}")
  print("\n")
my_n = 'Jan'
print(my_n)
print_string_info(my_n)
'''
Jan
String address: 0xe01c
Val: 0x79ac
Val: 0x7dfc
Val: 0x806c
'''
```
- We cannot change the element of str variable like tuple
  - `my_n[0] = 'x'` is not a valid operation

### 20. f-Strings
```py
my_i = 12
my_f = 1.23456789
my_n = 'Jan'
my_s = f"my_name: {my_n}, int = {my_i:5d} float = {my_f:.20f}"
print(my_s) # my_name: Jan, int =    12 float = 1.23456788999999989009
```

### 21. Paths and Filesystem
```py
import os
from pathlib import Path # since v3.4
some_path = os.path.abspath("/home/hpjeon/TEMP")
file_path = os.path.join(some_path,"sw","user_data.txt") # no manual slash
print(file_path)
p = Path(file_path) # modern style
print(p)
print(p.parent)
print(p.absolute())
print(p.parent.parent.is_dir()) 
print(p.parent.parent.exists())
```
- Context manager
```py
with open(file_path,'r') as f:
  alltxt = f.readlines()
print(alltxt)
# still f is alive even though the file is closed
```

### Quiz 7: Strings, Files and f-Strings

## Section 8: Chapter 4-1: Functions

### 22. Functions
### 23. Problems with Default Arguments
### 24. *args and **kwargs
### 25. Special Parameters
### Quiz 8: Functions
### 26. Commandline Arguments - Part 1
### 27. Commandline Arguments - Part 2
### 28. Commandline Arguments - Part 3
### Quiz 9: Commandline Arguments

## Section : Chapter : 

### 29. Closures and Decorator
### 30. More about Decorator
### Quiz 10: Closures and Decorators

## Section : Chapter : 

### 31. StaticMethods and ClassMethods
### 32. AbstractMethods
### 33. Property
### 34. Dunder Methods
### Quiz 11: Object orientation - Part 1
### 35. Method Resolution Order
### 36. Type vs. Isinstance vs. Issubclass
### 37. __init__ vs. __new__
### 38. Context Manager
### 39. Iterator and Generator
### 40. ABC Container
### 41. Dataclass and Slots
### 42. NamedTuple and TypedDict
### 43. Enum
### Quiz 12: Object orientation - Part 2

## Section : Chapter : 

### 44. Python Packages 101
### 45. Foreword
### 46. Cython
### 47. Numba
### 48. Mypyc
### 49. CPython
### 50. Another CPython Example
### 51. Pybind11
### 52. Benchmark
### Quiz 13: CPython APIs

## Section : Chapter : 

### 53. Threads, Processes and Async
### 54. Threads
### 55. Global Interpreter Lock
### 56. Thread Pool
### 57. Processes
### 58. Process Pool
### 59. Threads vs. Process - Recap
### 60. Asyncio
### 61. Asyncio Gather
### Quiz 14: Multi-Threads and -Processes

## Section : Chapter : 

### 62. Course conclusion
### 63. Bonus lecture


