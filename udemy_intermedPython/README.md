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
```py
# default argument
def g(p1: int, p2: int, p3: int=0, p4: int=10) ->None:
  print(f'p1: {p1}, p2: {p2}, p3: {p3}, p4: {p4}')
g(p1=1,p2=3) # p1: 1, p2: 3, p3: 0, p4: 10
fptr = g # like function pointer in C++
print(type(fptr)) # <class 'function'>
fptr(p1=2,p2=22,p3=33) # p1: 2, p2: 22, p3: 33, p4: 10
from typing import Any
from typing import Callable
# Use a function as an argument
def print_function_output(fn: Callable, **kwargs: Any) -> None:
  print(fn(**kwargs))
print_function_output(fptr, p1=1,p2=3) # p1: 1, p2: 3, p3: 0, p4: 10
print(fptr.__defaults__) # (0, 10) - print default argument
print(fptr.__name__) # g - function name
print(fptr.__code__) # <code object g at 0x79ea84606730, file "/tmp/ipykernel_7696/989217262.py", line 1>
print(fptr.__code__.co_argcount) # 4 - number of arguments
```

### 23. Problems with Default Arguments
- Do not use mutable (list, dictionary, ...) data type as default argument
```py
# example of a bad code
def grow_list(val: Any, my_list: list=[]):
  my_list.append(val)
  return my_list
my_l1 = grow_list(123)  
print(my_l1) # [123]
my_l2 = grow_list(456)  
print(my_l2) # [123, 456]
```
- When this above function runs first time, it will assign my_list internally, and another call will append the list, which is not expected
```py
# example of a good code
from typing import Optional
def grow_list(val: Any, my_list: Optional[list] = None):
  if my_list:
    my_list.append(val)
  else:
    my_list = [val]
  return my_list
my_l1 = grow_list(123)  
print(my_l1) # [123]
my_l2 = grow_list(456)  
print(my_l2) # [456]  
```
- When Optional argument is used with list, now it works as expected

### 24. *args and **kwargs
- `*args`: variable number of positional arguments
```py
def my_f2(a: Any, b: Any, *args: Any)-> None:
  print(*args, type(args))
  print(f'a: {a}, b: {b}, args: {args}')
my_f2(1,2,3,4) 
# 3 4 <class 'tuple'>
#  a: 1, b: 2, args: (3, 4)
my_f2(1,2,3)  
# 3 <class 'tuple'>
# a: 1, b: 2, args: (3,)
my_f2(1,2)  
# <class 'tuple'>
# a: 1, b: 2, args: ()
```
- Default arguments are located after `*args`
```py
def my_f4(a: Any, *args: Any, b: Optional[Any] = None)-> None:
  print(*args, type(args))
  print(f'a: {a}, b: {b}, args: {args}')
my_f4(1,2,3,4)  
# 2 3 4 <class 'tuple'>
# a: 1, b: None, args: (2, 3, 4)
my_f4(1,2,3,b=4)
# 2 3 <class 'tuple'>
# a: 1, b: 4, args: (2, 3)
```
- `**kwargs`: Keyword arguments
- `*args` vs `**kwargs`
  - `*args`: tuple, positional (non-keyword) arguments
  - `**kwargs`: dict, keyword (named) arguments
```py
def my_f(a: Any, *args: Any, x: int=2, y: int=3, z: int=4, **kwargs: Any) -> None:
  print(args, type(args))
  print(kwargs, type(kwargs))
  print(f'a:{a},x:{x},y:{y},z:{z}\nargs: {args}\nkwargs: {kwargs}')
my_f(1,2,3,y=123,v1=1,v2=2,v3=3)  
'''
(2, 3) <class 'tuple'>
{'v1': 1, 'v2': 2, 'v3': 3} <class 'dict'>
a:1,x:2,y:123,z:4
args: (2, 3)
kwargs: {'v1': 1, 'v2': 2, 'v3': 3}
'''
```

### 25. Special Parameters
- When `/` and `*` are not present in the function definition, arguments may be passed to a function by position or by keyword
```py
def f1(a: Any) -> None:
  print(a)
f1(2)  # by position
f1(a=2)# by keyword 
```
- Positional-only parameters are enforced before a `/`. This is used when the parameter name is to be hidden to the user
```py
def f1(a: Any,/) -> None:
  print(a)
f1(2)  # by position
#f1(a=2)# by keyword -> this is invalid
```
- To mark parameters as keyword-only, place an `*` in the argument list just before the first keyword-only paramete
```py
def f2(a: Any, *, b: Any) -> None:
  print(a,b)
f2(a=1,b=2) # work OK 
f2(1,b=2)   # works OK
# f2(1,2) - by position - invalid
```
- We can combine `/` and `*` in a single function arguments

### Quiz 8: Functions

### 26. Commandline Arguments - Part 1
- Using arvg
```py
import argparse
import sys

def main() -> None:
  print(f"argv: {sys.argv}")
  print(f"argc: {len(sys.argv)}")
  parser = argparse.ArgumentParser(
    prog = "ProgramName",
    usage = "How to",
    description = "What the program does",
    epilog = "Text at the bottom of help",
  )
  parser.add_argument(
    "-a",
    "--age",
    help="Enter your age (int)",
    type=int,
    required=True,
  )
  parser.add_argument(
    "-n",
    "--name",
    help="Enter your name (str)",
    type=str,
    required=True,
  )
  parser.add_argument(
    "-v",
    "--verbose",
    help="Verbose print",
    action = "store_true",
  )
  args = parser.parse_args()
  age = args.age
  name = args.name
  verbose = args.verbose
  if verbose:
    print(age, type(age))
    print(name, type(name))
  else:
    print(age)
    print(name)
if __name__ == "__main__":
  main()
```
- Demo:
```bash
$ python3 ch26.py 
argv: ['ch26.py']
argc: 1
usage: How to
ProgramName: error: the following arguments are required: -a/--age, -n/--name
$ python3 ch26.py -a 26 -n Jan -v
argv: ['ch26.py', '-a', '26', '-n', 'Jan', '-v']
argc: 6
26 <class 'int'>
Jan <class 'str'>
```

### 27. Commandline Arguments - Part 2
- nargs:
  - `*`: If exists, gather
  - `+`: At least one is required
```py
import argparse
import sys

def main() -> None:
  print(f"argv: {sys.argv}")
  print(f"argc: {len(sys.argv)}")
  parser = argparse.ArgumentParser(
    prog = "ProgramName",
    usage = "How to",
    description = "What the program does",
    epilog = "Text at the bottom of help",
  )
  parser.add_argument(
    "-a",
    "--age",
    help="Enter your age (int)",
    nargs = "+",  # <------- here
    type=int,
    required=True,
  )
  parser.add_argument(
    "-n",
    "--name",
    help="Enter your name (str)",
    nargs = "+",  # <------- here
    type=str,
    required=True,
  )
  parser.add_argument(
    "-v",
    "--verbose",
    help="Verbose print",
    action = "store_true",
  )
  args = parser.parse_args()
  age = args.age
  name = args.name
  verbose = args.verbose
  if verbose:
    print(age, type(age))
    print(name, type(name))
  else:
    print(age)
    print(name)
if __name__ == "__main__":
  main()
```
- Demo:
```bash
$ python3 ch26.py -a 26 28 -n Jan Kant -v
argv: ['ch26.py', '-a', '26', '28', '-n', 'Jan', 'Kant', '-v']
argc: 8
[26, 28] <class 'list'> # <--- multiple data allowed
['Jan', 'Kant'] <class 'list'> # <--- multiple data allowed
```

### 28. Commandline Arguments - Part 3
- Writing pip command inside of py code
```py
import argparse
def main() -> None:
    parser = argparse.ArgumentParser(
        description="My pip tool",
        prog="pip",
    )
    subparsers = parser.add_subparsers(
        title="Sub Parsers",
        description="Available subcommands",
        dest="subcommand",
    )
    subparser_install = subparsers.add_parser(
        "install",
        help="Pip Install command",
    )
    subparser_install.add_argument(
        "NAME",
        type=str,
        help="package to install",
    )
    subparser_list = subparsers.add_parser(
        "list",
        help="Pip List Command",
    )
    subparser_list.add_argument(
        "--verbose",
        help="Verbose print",
        action="store_true",
    )
    args = parser.parse_args()
    if args.subcommand == "install":
        print(f"command: pip {args.subcommand} {args.NAME}")
    elif args.subcommand == "list":
        print(f"command: pip {args.subcommand}")
    else:
        print("No subcommand specified.")
if __name__ == "__main__":
    main()
```    

### Quiz 9: Commandline Arguments

## Section 9: Chapter 4-2: Closures and Decorators

### 29. Closures and Decorator
- A closure is an inner function that has access to variables in the local scope of the outer function
```py
import time
from functools import wraps
from typing import Any
from typing import Callable
from typing import Optional
def outer_fn(message: str) -> Any:
  outer_message = "Outer: " + message
  current_time = time.time()
  def inner_fn() -> None:
    print("Inner: " + outer_message )
    print("Current time: ", current_time)
  return inner_fn
#
outer_fn("hello world") # function object - <function __main__.outer_fn.<locals>.inner_fn() -> None>
outer_fn("hello world")() # now call the function objecgt from outer_fn()
'''
Inner: Outer: hello world
Current time:  1771376922.7065992
'''
```
- Decorators
  - Wraps a function by another function
  -  Takes a function as an argument, returns a closure
  - The closure runs the previous passed in function with the *args and **kwargs argument
```py
def outer_fn(fn: Callable):
  def inner_fn() -> Any:
    fn_result = fn()
    return fn_result
  return inner_fn
def print_hello_world() -> None:
  print("Hello world")
decorated_fn = outer_fn(print_hello_world)  
decorated_fn() # Hello world
#
def decorator(fn: Callable) -> Callable:
  print("Start decorator function from: ", fn.__name__)
  def wrapper(*args: Any, **kwargs: Any) -> Any:
    print("Start wrapper function from: ", fn.__name__)
    fn_result = fn(*args, **kwargs)
    print("End wrapper function from: ", fn.__name__)
    return fn_result
  print("End decorator function from: ", fn.__name__)
  return wrapper
@decorator
def print_arguments2(
    a: int,
    b: int,
    c: Optional[int] = None,):
    print(f"A: {a}, B: {b}, C: {c}")
print_arguments2(2,b=3,c=4,)    
'''
Start decorator function from:  print_arguments2
End decorator function from:  print_arguments2
Start wrapper function from:  print_arguments2
A: 2, B: 3, C: 4
End wrapper function from:  print_arguments2
'''
```

### 30. More about Decorator
```py
def timing(fn: Callable) -> Callable:
  @wraps(fn)
  def timer(*args: Any, **kwargs: Any) -> Any:
    print("Start timer!")
    start_time = time.perf_counter()
    fn_result = fn(*args, **kwargs)
    end_time = time.perf_counter()
    time_duration = end_time - start_time
    print(f"Function {fn.__name__} took: {time_duration} s")
    return fn_result
  return timer
@timing
def iterate(n: int) -> int:
  val = 0
  for i in range(n):
    val += i
  return val
iterate(1_000_000)
'''
Start timer!
Function iterate took: 0.03715875200032315 s
'''  
```  

### Quiz 10: Closures and Decorators

## Section 10: Chapter 5: Object orientation

### 31. StaticMethods and ClassMethods
- Both of `@classmethod` and `@staticmethod` can be called without instantiating the class but serve different purposes
- Class method: can only modify the state of the class, not a single instance
  - `@classmethod`: receives the class itself as its first argument
  - Can be called as Class.method() or instance.method() (but cannot modify the state of the instance)
  - Commonly used as a **Factory method - returns a class instance**
  - May override `__init__`, producing an instance with different initial states
- Static method: can neither modify the object state of an instance nor the class state itself. It is rather a way to namespace your methods
  - `@staticmethod`: no access to instance (self) or the class (cls)
  - Class.method() only
  - Serves a class namespace
```py
class Date2:
    def __init__(
        self,
        year: int,
        month: int,
        day: int,
    ) -> None:
        self.year = year
        self.month = month
        self.day = day
    def __repr__(self) -> str:
        return f"{self.day}.{self.month}.{self.year}"
    def print_date(self) -> None:
        print(self)
    @classmethod
    def get_todays_date(cls):
        date = cls.__new__(cls)
        time = localtime()
        date.year = time.tm_year
        date.month = time.tm_mon
        date.day = time.tm_mday
        return date
    @staticmethod
    def is_todays_date(date) -> bool:
        time = localtime()
        return bool(
            date.year == time.tm_year
            and date.month == time.tm_mon
            and date.day == time.tm_mday
        )
# Class method
date2 = Date2.get_todays_date() # Instantiate a class object
date2.print_date()
# Static method
Date2.is_todays_date(date2) # no chnage of class or instance
```

### 32. AbstractMethods
- Like pure virtual in C++
```py
import abc
import math
class Base(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def m1(self)->None:
    return
  @staticmethod
  @abc.abstractmethod
  def m2(self)->None:
    return
  @classmethod
  @abc.abstractmethod
  def m3(self)->None:
    return        
class MyClass1(Base):
  def m1(self)->str:
    return "m1"
  def m2(self)->str:
    return "m2"
class MyClass2(Base):
  def m1(self)->str:
    return "1"
  def m2(self)->str:
    return "2"
  def m3(self)->str:
    return "3"
# var = MyClass1() # invalid as m3() is not defined
var2 = MyClass2()
var2.m1()
```

### 33. Property
- `@property` decorator for the getter method and `@attribute_name.setter` for the setter method
  - We can accss class attribute directly but using `@property` provides more flexibility
  - Ex: Sanitizing the data, min/max check, data type check, ...
```py
class User2:
  def __init__(self, name: str, user_id: int) -> None:
    self._name = name
    self._user_id = user_id
  def __repr__(self) -> str:
    return f"{self._name}, {self._user_id}"
  @property # getter 
  def name(self) -> str:
    return self._name
  @name.setter # setter
  def name(self, new_name: str) -> None:
    if type(new_name) is not str:
      self._name = 'Non integer entered'
    else:
      self._name = new_name
  @name.deleter
  def name(self) -> None:
    self._name = ""
u2 = User2("Jan",123)
print(u2.name, u2._name) # Jan Jan
u2._name = 123
print(u2.name, u2._name) # 123 123
u2.name = 123 # type() will check the data type
print(u2.name, u2._name) # Non integer entered Non integer entered
```

### 34. Dunder Methods
- Aka magic methods/special methods
  - Starts and ends with `__` like `__init__`
- Basic Customizations
  - `__new__(self)` return a new object (an instance of that class).
  - `__init__(self)` is called when the object is initialized.
  - `__del__(self)` for del() function. Called when the object is to be destroyed.
  - `__repr__(self)` for repr() function. It returns a string to print the object.
  - `__str__(self)` for str() function. Return a string to print the object.
  - `__bytes__(self)` for bytes() function. Return a byte object which is the byte string representation.
  - `__format__(self)` for format() function. Evaluate formatted string literals.
- Comparison Operators
  - `__lt__(self, anotherObj)` for < operator.
  - `__le__(self, anotherObj)` for <= operator.
  - `__eq__(self, anotherObj)` for == operator.
  - `__ne__(self, anotherObj)` for != operator.
  - `__gt__(self, anotherObj)` for > operator.
  - `__ge__(self, anotherObj)` for >= operator.
- Arithmetic Operators
  - `__add__(self, anotherObj)` for + operator.
  - `__sub__(self, anotherObj)` for – operation on object.
  - `__mul__(self, anotherObj)` for * operation on object.
  - `__matmul__(self, anotherObj)` for @ operator (numpy matrix multiplication).
  - `__truediv__(self, anotherObj)` for simple / division operation on object.
  - `__floordiv__(self, anotherObj)` for // floor division operation on object.
- Type Conversion
  - `__abs__(self)` make support for abs() function.
  - `__int__(self)` support for int() function. Returns the integer value.
  - `__float__(self)` for float() function support. Returns float equivalent.
  - `__complex__(self)` for complex() function support. Return complex value.
  - `__round__(self, nDigits)` for round() function.
  - `__trunc__(self)` for trunc() function of math module.
  - `__ceil__(self)` for ceil() function of math module.
  - `__floor__(self)` for floor() function of math module.
- Emulating Container Types
  - `__len__(self)` for len() function. Returns the total number in any container.
  - `__getitem__(self, key)` to support index lookup.
  - `__setitem__(self, key, value)` to support index assignment
  - `__delitem__(self, key)` for del() function. Delete the value at the index.
  - `__iter__(self)` returns an iterator when required that iterates all values in the container.

### Quiz 11: Object orientation - Part 1

### 35. Method Resolution Order
- `ClassName.mro()`
```py
class A:
  def __init__(self):
    print("Init A called!")
    self.a = 1
class B(A):
  def __init__(self):
    super().__init__()
    print("Init B called!")
    self.b = self.a * 2
class C(A):
  def __init__(self):
    super().__init__()
    print("Init C called!")
    self.c = self.a / 2
class D(B, C):
  def __init__(self):
    super().__init__()
    print("Init D called!")
    self.d = self.b + self.c
d = D()
'''
Init A called!
Init C called!
Init B called!
Init D called!
'''
D.mro() # [__main__.D, __main__.B, __main__.C, __main__.A, object]
```

### 36. Type vs. Isinstance vs. Issubclass
```py
a = A()
b = B()
print(type(b)) # <class '__main__.B'>
print(type(b) == B) # True
print(type(b) == A) # False
print(isinstance(b,B)) # True
print(isinstance(b,A)) # True
print(issubclass(B, (A,C))) # True - (A,C) -> A or C
print(issubclass(A, B)) # False
```

### 37. __init__ vs. __new__
- If `__new__` returns a class object, then `__init__` is executed after then
```py
class Data:
  def __new__(cls, name):
    print(f"Creating a new {cls.__name__} obj")
    obj = object.__new__(cls)
    return obj
  def __init__(self, name):
    print(f"Initializing a new {self.__class__.__name__} obj")
    self.name = name
u = Data('Jan')
'''
Creating a new Data obj
Initializing a new Data obj
'''
```
- We may create a singleton object using this
```py
class Singleton(object):
  _instance = None  # Keep instance reference
  def __new__(cls, *args, **kwargs):
    if not cls._instance:
      cls._instance = object.__new__(cls, *args, **kwargs)
    else:
      print("Object already created!")
    return cls._instance
s = Singleton()
print(hex(id(s))) # 0x7ea01b36b230 
s = Singleton() # Object already created!
print(hex(id(s))) # 0x7ea01b36b230
```

### 38. Context Manager
- Example:
```py
with ContextManager() as var_name: # __init__ runs here
  # do something # __enter__ runs here
# __exit__ runs here
```  
- At entering the with-Context, the object gets created (`__init__` and `__enter__`)
- At exiting the with-Context, the object calls the `__exit__` method
  - __exit__ needs `self, exception_type, exception_value, traceback` as arguments
  - When no exception occured, they are passed as None
```py
class NameHandler:
  def __init__(self, name: Any) -> None:
    self.name = name
    print(f"Creating {self.name}")
  def __enter__(self):
    print(f"Entering with: {self.name}")
    if not isinstance(self.name, str):
        raise TypeError
    return self
  def __exit__(
    self,
    exception_type,
    exception_value,
    traceback,
  ) -> None:
    print(f"Exiting with: {self.name}")
with NameHandler("Jan") as m:  # __init__
  m.name = 3  # __enter__
# __exit__
'''
Creating Jan
Entering with: Jan
Exiting with: 3
'''
```

### 39. Iterator and Generator
```py
my_list = [1,2,3,4]
for val in my_list:
  print(val)
my_iter = iter(my_list)
print(my_iter, type(my_iter)) # <list_iterator object at 0x7f1470386f80> <class 'list_iterator'>
print(next(my_iter)) # 1
print(next(my_iter)) # 2
print(next(my_iter)) # 3
print(next(my_iter)) # 4
print(next(my_iter)) # StopIteration error 
```
- Defining a class with iteration features:
```py
class PowerOfTwo:
  def __init__(self,N):
    self.N = N
  def __iter__(self):
    self.current_n = 0
    return self
  def __next__(self):
    if self.current_n <= self.N:
      current_result = 2**self.current_n
      self.current_n +=1
      return current_result
    else:
      raise StopIteration
p = PowerOfTwo(4)
p_iter = iter(p)
print(p, type(p), p_iter, type(p_iter))
# <__main__.PowerOfTwo object at 0x7f14700b1be0> <class '__main__.PowerOfTwo'> <__main__.PowerOfTwo object at 0x7f14700b1be0> <class '__main__.PowerOfTwo'>
print(next(p_iter)) # 1
print(next(p_iter)) # 2
print(next(p_iter)) # 4
print(next(p_iter)) # 8
print(p) # <__main__.PowerOfTwo object at 0x7f14700b1be0>
print(p) # <__main__.PowerOfTwo object at 0x7f14700b1be0>
for i in p:
  print("i=",i)
'''
i= 1
i= 2
i= 4
i= 8
i= 16
'''
```
- Generator: function with a least one yield statement
  - Easy to implement, memory efficient, represents infinite stream
```py
def PowerOfTwoGen(N):
  n_now = 0
  while n_now <= N:
    res = 2**n_now
    n_now +=1
    yield res # not return. If yield is not used, __iter__() and __next__() are required
g = PowerOfTwoGen(4)  
print(g, type(g)) 
# <generator object PowerOfTwoGen at 0x7f144fad0c70> <class 'generator'> 
for i in g:
  print(i)
'''
1
2
4
8
16
'''  
g0 = PowerOfTwoGen(4)  
print(next(g0)) # 1
print(next(g0)) # 2
print(next(g0)) # 4
print(next(g0)) # 8
print(next(g0)) # 16
print(next(g0)) # StopIteration
```

### 40. ABC Container
- Abstract Base Classes: https://docs.python.org/3/library/collections.abc.html
```py
from collections.abc import MutableSequence
from typing import Any
class MyOwnList(MutableSequence):
  def __init__(self, values: Any = None) -> None:
    super().__init__()
    if values:
      self._values = values
    else:
      self._values = []
try:
  lst = MyOwnList()
except TypeError as e:
  print(e)
# Can't instantiate abstract class MyOwnList without an implementation for abstract methods '__delitem__', '__getitem__', '__len__', '__setitem__', 'insert'
```
- Need to implement '__delitem__', '__getitem__', '__len__', '__setitem__', 'insert' in the class
```py
class MyOwnList(MutableSequence):
  def __init__(self, values: Any = None) -> None:
    super().__init__()
    if values:
      self._values = values
    else:
      self._values = []
  def __str__(self) -> str:
    return str(self._values)
  def __len__(self) -> int:
    return len(self._values)
  def __getitem__(self, idx: int) -> Any:
    return self._values[idx]
  def __setitem__(self, idx: int, val: Any) -> None:
    self._values[idx] = val
  def __delitem__(self, idx: int) -> None:
    del self._values[idx]
  def insert(self, idx, val: Any) -> None:
    self._values.insert(idx, val)
  def append(self, val: Any) -> None:
    self._values.append(val)
my_list = MyOwnList([1, 2, 3])  # __init__
print(my_list)  # __str__ - [1, 2, 3]
print(len(my_list))  # __len__ - 3
```

### 41. Dataclass and Slots
- Regular class:
```py
from dataclasses import dataclass
from dataclasses import field
class User:
  def __init__(self, name: str, age: int):
    self.name = name
    self.age = age
u = User('Jan',27)    
print(u, u.__dict__) 
# <__main__.User object at 0x7f14700b1a90> {'name': 'Jan', 'age': 27}
```
- Dataclass:
```py
@dataclass
class User:
  name: str
  age: int
u1 = User('jan',27)
print(u1, u1.__dict__)
# User(name='jan', age=27) {'name': 'jan', 'age': 27}
```
  - Much simpler and attributes are mutable:
```py
u1.another_item = 'abc'
print(u1.__dict__)  
# {'name': 'jan', 'age': 27, 'another_item': 'abc'}
```
- If such behavior is to be forbidden, use `slots=True`
  - `__dict__` is not available
  - Much faster and less memory consumption
```py
@dataclass(slots=True)
class User:
  name: str
  age: int
u2 = User('jan',27)
# print(u2.__dict__) # AttributeError. No attibute of __dict__
# u2.another_item = 'abc' # AttributeError  
```
- Creating a new, distinct default value for each instance of the class with default_factory keyword
  - Avoiding unintended shared state b/w instances
```py
@dataclass(slots=True)
class User3:
  name: str
  age: int
  is_active: bool = False
  orders: list[float] = field(
    default_factory=list,
    compare=False,
    hash=False,
    repr=False,
  )
  def method(self):
    print(self.name)
u3 = User3("jan", 27)
print(u3)
print(id(u3.orders))
u3.method()
User3(name='jan', age=27, is_active=False)
u4 = User3("jan", 27)
print(u4)
print(id(u4.orders))
u4.method()
'''
User3(name='jan', age=27, is_active=False)
139725210009984
jan
User3(name='jan', age=27, is_active=False)
139725210022592
jan
'''
```

### 42. NamedTuple and TypedDict
- Named tuple
  - Tuple subclasses with named fields
  - Immutable
  - Hashable
  - `__repr__`, `__str__` and compare are defined
  - Support indexing
  - 3 different containers shown below:
```py
from collections import namedtuple
from typing import NamedTuple
from typing import TypedDict
User = namedtuple( 'User', ['name', 'age', 'is_admin'])
u_jan = User(name='Jan',age=28, is_admin=True)
print(u_jan) # User(name='Jan', age=28, is_admin=True)
# u_jan.age=33 # AttributeError
class User2(NamedTuple):
  name: str
  age: int
  is_admin: bool=False
u_jan2 = User2(name='Jan',age=25,is_admin=True)
print(u_jan2) # User2(name='Jan', age=25, is_admin=True)
class User3(TypedDict):
  name: str
  age: int
  is_admin: bool=False
u_jan3: User3 = {'name':'jan', 'age':29, 'is_admin':True}
print(u_jan3) # {'name': 'jan', 'age': 29, 'is_admin': True}
```  

### 43. Enum
```py
from enum import Enum
from enum import IntEnum
from enum import auto
class Colors(Enum):
  RED=1
  GREEN=2
  BLUE=3
color = Colors.RED
print(issubclass(Colors, int), color) # False Colors.RED
class Colors2(IntEnum):
  RED=1
  GREEN=2
  BLUE=3
color2 = Colors2.RED
print(issubclass(Colors2, int), color2) # True 1
class Colors3(Enum):
  RED = auto()
  GREEN = auto()
  BLUE = auto()
color3 = Colors3.RED
print(list(Colors3), issubclass(Colors3,int), color3)
# [<Colors3.RED: 1>, <Colors3.GREEN: 2>, <Colors3.BLUE: 3>] False Colors3.RED
```

### Quiz 12: Object orientation - Part 2

## Section 11: Chapter 6: Cython and CPython

### 44. Python Packages 101
- How to make a new package
- Required files for test_package
  - `pyproject.toml`
```
[build-system]
requires = ["setuptools", "wheel"]
```
  - `setup.py`
```py
from setuptools import setup
def main() -> None:
    setup(
        name="test_package",
        version="1.0.0",
        packages=["test_package"],
    )
if __name__ == "__main__":
    main()
```
- `test_package/__init__.py`
```py
def hello_world():
    print("Hello World")
```    
- Install command: `pip install --user -e .`
- Demo:
```bash
$ cat main.py 
import test_package
def main() -> None:
    test_package.hello_world()
if __name__ == "__main__":
    main()
$ python3 ./main.py 
Hello World
```

### 45. Foreword

### 46. Cython
- Generates C code from Cython coding then compile
  - We do not write C code
- Demo code
  - `pyproject.toml`
```
[build-system]
requires = ["setuptools", "wheel", "Cython"]
```
  - setup.py:
```py
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup
CYTHON_EXTENSIONS = [
    Extension(
        name="math_cython.cython_computations",
        sources=["math_cython/cython_computations.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]
EXT_MODULES = cythonize(CYTHON_EXTENSIONS, language_level="3")
def main() -> None:
    setup(
        name="math_cython",
        version="1.0.0",
        packages=["math_cython"],
        ext_modules=EXT_MODULES,
    )
if __name__ == "__main__":
    main()
```      
  - math_cython/__init__.py
```py
from .computations import cython_clip_vector
from .computations import naive_cython_clip_vector
from .computations import python_clip_vector
__all__ = [
    "cython_clip_vector",
    "naive_cython_clip_vector",
    "python_clip_vector",
]
```
  - math_cython/computations.py
```py
# pylint: disable=import-error
import array
from .cython_computations import _cython_clip_vector
from .cython_computations import _naive_cython_clip_vector
def python_clip_vector(
    vector_in: list[float],
    min_value: float,
    max_value: float,
    vector_out: list[float],
) -> None:
    for idx in range(len(vector_in)):
        vector_out[idx] = min(max(vector_in[idx], min_value), max_value)
def naive_cython_clip_vector(
    vector_in: array.array,
    min_value: float,
    max_value: float,
    vector_out: array.array,
) -> None:
    _naive_cython_clip_vector(vector_in, min_value, max_value, vector_out)
def cython_clip_vector(
    vector_in: array.array,
    min_value: float,
    max_value: float,
    vector_out: array.array,
) -> None:
    _cython_clip_vector(vector_in, min_value, max_value, vector_out)
```      
- math_cython/cython_computations.pyx
```py
cimport cython
ctypedef fused vector_type:
    float
    double
def _naive_cython_clip_vector(
    list_in,
    min_value,
    max_value,
    list_out
):
    for idx in range(len(list_in)):
        list_out[idx] = min(max(list_in[idx], min_value), max_value)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)  # Deactivate negative indexing
def _cython_clip_vector(
    vector_type[:] list_in,
    vector_type min_value,
    vector_type max_value,
    vector_type[:] list_out
):
    for idx in range(len(list_in)):
        list_out[idx] = min(max(list_in[idx], min_value), max_value)
from cython.parallel import prange
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)  # Deactivate negative indexing
def _cython_clip_vector_parallel(
    vector_type[:] list_in,
    vector_type min_value,
    vector_type max_value,
    vector_type[:] list_out
):
    cdef signed int idx = 0
    cdef signed int length = len(list_in)
    with nogil:
        for idx in prange(length, schedule="guided"):
            if list_in[idx] < min_value:
                list_out[idx] = min_value
            if list_in[idx] > max_value:
                list_out[idx] = max_value
```     
- main.py
```py
import array
import math_cython
def main() -> None:
    lst = [float(i) for i in range(10)]
    arr1 = array.array("f", list(range(10)))
    arr2 = array.array("f", list(range(10)))
    min_value = 2.0
    max_value = 4.0
    math_cython.python_clip_vector(lst, min_value, max_value, lst)
    math_cython.naive_cython_clip_vector(arr1, min_value, max_value, arr1)
    math_cython.cython_clip_vector(arr2, min_value, max_value, arr2)
    print(lst)
    print(arr1)
    print(arr2)
if __name__ == "__main__":
    main()
```           
- Run command:  
```bash
$ pip install --user -e .
Obtaining file:///home/hpjeon/hw/class/udemy_IntermedPython/cython
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Building wheels for collected packages: math_cython
  Building editable for math_cython (pyproject.toml) ... done
  Created wheel for math_cython: filename=math_cython-1.0.0-0.editable-cp313-cp313-linux_x86_64.whl size=2695 sha256=8b6956852239b94fd39a8adceb3c6fce4f4dee1033f5c1ae73c545a89099c4e9
  Stored in directory: /tmp/pip-ephem-wheel-cache-1ebprqe3/wheels/17/d9/46/fbfc89c667f850ec9dac61043b9143676335b0b96a9281093f
Successfully built math_cython
Installing collected packages: math_cython
Successfully installed math_cython-1.0.0
# This will generate math_cython/cython_computations.c
$ python ./main.py
[2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
array('f', [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
array('f', [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
```

### 47. Numba
- Demo
  - pyproject.toml:
```
[build-system]
requires = ["setuptools", "wheel", "numpy", "numba"]
```
  - setup.py:
```py
from distutils.core import setup
from math_numba import cc
def main() -> None:
    setup(
        name="math_numba",
        version="1.0.0",
        ext_modules=[cc.distutils_extension()],
    )
if __name__ == "__main__":
    main()
```
- `math_numba/__init__.py`:
```py
from numba.pycc import CC
cc = CC("math_numba")
cc.verbose = True
@cc.export("clip_vector", "f4[:](f4[:], f4, f4)") # f4: float 4byte, f4[:]: float 4byte array
def clip_vector(
    a: list[float],
    min_value: float,
    max_value: float,
) -> list[float]:
    len_ = len(a)
    for i in range(len_):
        if a[i] < min_value:
            a[i] = min_value
        elif a[i] > max_value:
            a[i] = max_value
    return a
```
  - main.py:
```py
import math_numba
import numpy as np
def main() -> None:
    arr = np.array([float(i) for i in range(10)], dtype=np.float64)
    min_value = 2.0
    max_value = 4.0
    math_numba.clip_vector(arr, min_value, max_value)
    print(arr)
if __name__ == "__main__":
    main()
```
- Running command:
```bash
$ pip install --user -e .
Obtaining file:///home/hpjeon/hw/class/udemy_IntermedPython/numba
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Building wheels for collected packages: math_numba
  Building editable for math_numba (pyproject.toml) ... done
  Created wheel for math_numba: filename=math_numba-1.0.0-0.editable-cp313-cp313-linux_x86_64.whl size=2679 sha256=0eea6e46fa8f10fd42c71fdec40902b5d4509ab49705d573c5846789c6f219e4
  Stored in directory: /tmp/pip-ephem-wheel-cache-_3qc3ejy/wheels/5e/17/91/156e649f75bd9c8a061fb7ae535a417c3c484568969cbad208
Successfully built math_numba
Installing collected packages: math_numba
Successfully installed math_numba-1.0.0
$ python3 ./main.py 
[2. 2. 2. 3. 4. 4. 4. 4. 4. 4.]
```

### 48. Mypyc
- setup.py:
```py
from mypyc.build import mypycify
from setuptools import setup
def main() -> None:
    setup(
        name="math_mypyc",
        version="1.0.0",
        packages=["math_mypyc"],
        ext_modules=mypycify(
            [
                "math_mypyc/__init__.py",
            ],
        ),
    )
if __name__ == "__main__":
    main()
```    
- pyproject.toml:
```
[build-system]
requires = ["setuptools", "wheel", "mypy"]
```
  - Do not use `"mypyc"`
- `math_mypyc/__init__.py`:
```py
def clip_vector(
    a: list[float],
    min_value: float,
    max_value: float,
) -> list[float]:
    len_ = len(a)
    for i in range(len_):
        if a[i] < min_value:
            a[i] = min_value
        elif a[i] > max_value:
            a[i] = max_value
    return a
```
- main.py:
```py
import math_mypyc
def main() -> None:
    lst = list(range(10))
    min_value = 2
    max_value = 4
    math_mypyc.clip_vector(lst, min_value, max_value)
    print(lst)
if __name__ == "__main__":
    main()
```
- Demo:
```bash
$ pip install --user -e .
Obtaining file:///home/hpjeon/hw/class/udemy_IntermedPython/mypc
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Building wheels for collected packages: math_mypyc
  Building editable for math_mypyc (pyproject.toml) ... done
  Created wheel for math_mypyc: filename=math_mypyc-1.0.0-0.editable-cp313-cp313-linux_x86_64.whl size=2679 sha256=8eb945837f4b9714e0ad0b57b75c3cebdc96d6f9b794660f955ca744c8bf892a
  Stored in directory: /tmp/pip-ephem-wheel-cache-4qxhelks/wheels/38/08/de/d28ced40a4cdd91c07d185ef54f77957f34487b09950c69a63
Successfully built math_mypyc
Installing collected packages: math_mypyc
Successfully installed math_mypyc-1.0.0
$ python3 ./main.py
[2, 2, 2, 3, 4, 4, 4, 4, 4, 4]
```

### 49. CPython
- pyproject.toml
```
[build-system]
requires = ["setuptools", "wheel"]
```
- setup.py
```py
from setuptools import Extension
from setuptools import setup
EXTENSIONS = [Extension(name="math_cpython", sources=["math_cpython/clip.c"])]
def main() -> None:
    setup(name="math_cpython", version="1.0.0", ext_modules=EXTENSIONS)
if __name__ == "__main__":
    main()
```    
- math_cpython/clip.c
```c
#define PY_SSIZE_T_CLEAN
#include "Python.h"
static PyObject *method_add_vectors(PyObject *self, PyObject *args)
{
    PyObject *const list_a = NULL;
    PyObject *const list_b = NULL;
    if (!PyArg_ParseTuple(args, "OO", &list_a, &list_b))
    {
        PyErr_SetString(PyExc_TypeError, "error");
        return NULL;
    }
    if (!PyList_Check(list_a) || !PyList_Check(list_b))
    {
        PyErr_SetString(PyExc_ValueError, "error");
        return NULL;
    }
    const Py_ssize_t len_a = PyList_Size(list_a);
    const Py_ssize_t len_b = PyList_Size(list_b);
    if (len_a != len_b)
    {
        PyErr_SetString(PyExc_ValueError, "error");
        return NULL;
    }
    PyObject *result = PyList_New(len_a);
    for (Py_ssize_t i = 0U; i < len_a; ++i)
    {
        PyObject *const item_a = PyList_GetItem(list_a, i);
        PyObject *const item_b = PyList_GetItem(list_b, i);
        if (!PyFloat_Check(item_a) || !PyFloat_Check(item_b))
        {
            PyErr_SetString(PyExc_ValueError, "Items must be 64bit floats..");
            return NULL;
        }
        PyList_SetItem(result, i, PyNumber_Add(item_a, item_b));
    }
    return result;
}
static PyObject *method_clip_vector(PyObject *self, PyObject *args)
{
    PyObject *const list = NULL;
    double min_value = 0.0;
    double max_value = 0.0;
    if (!PyArg_ParseTuple(args, "Odd", &list, &min_value, &max_value))
    {
        PyErr_SetString(PyExc_TypeError, "error");
        return NULL;
    }
    const Py_ssize_t len = PyList_Size(list);
    for (Py_ssize_t i = 0U; i < len; ++i)
    {
        const PyObject *const item = PyList_GetItem(list, i);
        if (!PyFloat_Check(item))
        {
            PyErr_SetString(PyExc_ValueError, "Items must be 64bit floats..");
            return NULL;
        }
        const double temp = PyFloat_AsDouble(item);
        if (temp < min_value)
        {
            PyList_SetItem(list, i, PyFloat_FromDouble(min_value));
        }
        else if (temp > max_value)
        {
            PyList_SetItem(list, i, PyFloat_FromDouble(max_value));
        }
    }
    return Py_None;
}
static PyMethodDef math_cpythonMethods[] = {
    {"add_vectors", method_add_vectors, METH_VARARGS, "CPython Function"},
    {"clip_vector", method_clip_vector, METH_VARARGS, "CPython Function"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef math_cpythonmodule = {
    PyModuleDef_HEAD_INIT, "math_cpython", "CPython Module", -1, math_cpythonMethods
};
PyMODINIT_FUNC PyInit_math_cpython(void)
{
    return PyModule_Create(&math_cpythonmodule);
}
```
- main.py:
```py
import math_cpython
def main() -> None:
    lst = [float(i) for i in range(10)]
    min_value = 2.0
    max_value = 4.0
    math_cpython.clip_vector(lst, min_value, max_value)
    print(lst)
    lst1 = [float(i) for i in range(3)]  # 0, 1, 2
    lst2 = [float(i) * 2.0 for i in range(3)]  # 0, 2, 4
    result = math_cpython.add_vectors(lst1, lst2)
    print(result)
if __name__ == "__main__":
    main()
```
- Demo:
```bash
$ pip install --user -e .
Obtaining file:///home/hpjeon/hw/class/udemy_IntermedPython/cpython
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Building wheels for collected packages: math_cpython
  Building editable for math_cpython (pyproject.toml) ... done
  Created wheel for math_cpython: filename=math_cpython-1.0.0-0.editable-cp313-cp313-linux_x86_64.whl size=2712 sha256=21024b22e0c1b0d7b1e99d1b04568ba7b91f73a10bf1a72b405ebebd8e449ff7
  Stored in directory: /tmp/pip-ephem-wheel-cache-xrb3mtav/wheels/78/07/d0/b9c4210a090f4ad8e0a9369194bf5c9361a7f40cf69413df05
Successfully built math_cpython
Installing collected packages: math_cpython
Successfully installed math_cpython-1.0.0
$ python3 ./main.py 
Segmentation fault (core dumped) # ?
```

### 50. Another CPython Example

### 51. Pybind11
- pyproject.toml:
```
[build-system]
requires = [
    "setuptools>=42",
    "wheel",
    "pybind11>=2.9.0",
]
build-backend = "setuptools.build_meta"
```
- setup.py:
```py
from pybind11.setup_helpers import Pybind11Extension
from pybind11.setup_helpers import build_ext
from setuptools import setup
ext_modules = [
    Pybind11Extension("math_cpp_python", ["math_cpp_python/clip.cpp"]),
]
def main() -> None:
    setup(
        name="math_cpp_python",
        version="1.0.0",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )
if __name__ == "__main__":
    main()
```    
- math_cpp_python/clip.cpp:
```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;
void clip_vector(py::list in, double min_value, double max_value) {
    size_t idx = 0;
    auto it_in = in.begin();
    for (; it_in != in.end(); ++it_in, ++idx)
    {
        const double curr_val = it_in->cast<double>();
        if (curr_val < min_value)
        {
            in[idx] = min_value;
        }
        else if (curr_val > max_value)
        {
            in[idx] = max_value;
        }
    }
}
PYBIND11_MODULE(math_cpp_python, m) {
    m.def("clip_vector", &clip_vector, "doc...");
}
```
- main.py:
```py
import math_cpp_python
def main() -> None:
    lst = [float(i) for i in range(10)]
    min_value = 2.0
    max_value = 4.0
    math_cpp_python.clip_vector(lst, min_value, max_value)
    print(lst)
if __name__ == "__main__":
    main()
```
- Demo:
```bash
$ pip install --user -e .
Obtaining file:///home/hpjeon/hw/class/udemy_IntermedPython/pybind11
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Building wheels for collected packages: math_cpp_python
  Building editable for math_cpp_python (pyproject.toml) ... done
  Created wheel for math_cpp_python: filename=math_cpp_python-1.0.0-0.editable-cp313-cp313-linux_x86_64.whl size=2767 sha256=752365f86a22666eeb6663a22c180931f72c00c8f30db22d349140f37780f57d
  Stored in directory: /tmp/pip-ephem-wheel-cache-dxwdidzg/wheels/e8/d8/85/f51bc0cacc127a2758e7d34fac1171ce8dea4feca1c256a479
Successfully built math_cpp_python
Installing collected packages: math_cpp_python
Successfully installed math_cpp_python-1.0.0
$ python3 ./main.py
[2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
```

### 52. Benchmark
- cython > numba > cpython > pybind11
  - 1 > 1.51 > 3.97 > 4.61x slow in this section modeling

### Quiz 13: CPython APIs

## Section 12: Chapter 7: Threads, Processes, and Async

### 53. Threads, Processes and Async
- Threading(concurrently):
  - A new thread is spawned within the existing process
  - Starting a thread is faster than starting a process
  - Memory is shared b/w all threads
  - Mutexes often necessary to control access to shared data
- Multiprocessing (parallel):
  - A new process is started independently from the first process
  - Starting a process is slower than starting a thread
  - Memory is not shared b/w processes
- Async
  - Refers to the occurrence of events independent of the main program flow
  - Waiting foir an event from the outside of our program
  - Concurrent programming is the ability to release CPU during waiting periods so that other tasks can use it

### 54. Threads
```py
import time
from threading import Thread
def worker(sleep_time: float)-> None:
  print("Start worker")
  time.sleep(sleep_time)
  print("end worker")
t1 = Thread(target=worker, name="t1", args=(2.0,))  
print(f"Ident: {t1.ident}")
print(f"Alive: {t1.is_alive()}") # t1 is not running yet
print(f"Name: {t1.name}")
'''
Ident: None
Alive: False
Name: t1
'''
t1.start() # t1 thread starts
print(f"Ident: {t1.ident}")
print(f"Alive: {t1.is_alive()}")
print(f"Name: {t1.name}")
t1.join()
'''
Start worker
Ident: 131715386283712
Alive: True
Name: t1
end worker
'''
```

### 55. Global Interpreter Lock
- GIL gives a computational power into a single thread
- The code deosn't run parallel
```py
import math
numbers = [
    18014398777917439,
    18014398777917439,
    18014398777917439,
    18014398777917439,
]
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3, 5, 7, 11, 13, 17):
        return True
    if (
        n % 2 == 0
        or n % 3 == 0
        or n % 5 == 0
        or n % 7 == 0
        or n % 11 == 0
        or n % 13 == 0
        or n % 17 == 0
    ):
        return False
    upper_limit = int(math.sqrt(n)) + 1
    for i in range(19, upper_limit, 2):
        if n % i == 0:
            return False
    return True
start = time.perf_counter_ns()
for number in numbers:
    is_prime(number)
end = time.perf_counter_ns()
print(f"time: {(end - start) / 1e09} s")
# time: 12.185844922 s
threads = [Thread(target=is_prime, args=(number,)) for number in numbers]
start = time.perf_counter_ns()
[t.start() for t in threads]
[t.join() for t in threads]
end = time.perf_counter_ns()
print(f"time: {(end - start) / 1e09} s")
# time: 11.580428442 s
```

### 56. Thread Pool
```py
import math
import time
from concurrent.futures import ThreadPoolExecutor
NUMBERS = [
    18014398777917439,
    18014398777917439,
    18014398777917439,
    18014398777917439,
]
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in {2, 3, 5, 7, 11, 13, 17}:
        return True
    if (
        n % 2 == 0
        or n % 3 == 0
        or n % 5 == 0
        or n % 7 == 0
        or n % 11 == 0
        or n % 13 == 0
        or n % 17 == 0
    ):
        return False
    upper_limit = int(math.sqrt(n)) + 1
    return all(n % i != 0 for i in range(19, upper_limit, 2))
def main() -> None:
    start = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=len(NUMBERS)) as ex:
        for number, prime in zip(
            NUMBERS,
            ex.map(is_prime, NUMBERS),
            strict=False,
        ):
            print(f"{number} is prime: {prime}")
    end = time.perf_counter_ns()
    print(f"time: {(end - start) / 1e09} s")
if __name__ == "__main__":
    main()
```
- Took 12.406750981 s


### 57. Processes
```py
import math
import time
from multiprocessing import Pool
NUMBERS = [
    18014398777917439,
    18014398777917439,
    18014398777917439,
    18014398777917439,
]
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in {2, 3, 5, 7, 11, 13, 17}:
        return True
    if (
        n % 2 == 0
        or n % 3 == 0
        or n % 5 == 0
        or n % 7 == 0
        or n % 11 == 0
        or n % 13 == 0
        or n % 17 == 0
    ):
        return False
    upper_limit = int(math.sqrt(n)) + 1
    return all(n % i != 0 for i in range(19, upper_limit, 2))
def main() -> None:
    start = time.perf_counter_ns()
    with Pool() as pool:
        result = pool.map(is_prime, NUMBERS)
    print(result)
    end = time.perf_counter_ns()
    print(f"Took: {(end - start) / 1e09} s")
if __name__ == "__main__":
    main()
```
- Took 3.079548471 s

### 58. Process Pool
```py
import math
import time
from concurrent.futures import ProcessPoolExecutor
NUMBERS = [
    18014398777917439,
    18014398777917439,
    18014398777917439,
    18014398777917439,
]
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in {2, 3, 5, 7, 11, 13, 17}:
        return True
    if (
        n % 2 == 0
        or n % 3 == 0
        or n % 5 == 0
        or n % 7 == 0
        or n % 11 == 0
        or n % 13 == 0
        or n % 17 == 0
    ):
        return False
    upper_limit = int(math.sqrt(n)) + 1
    return all(n % i != 0 for i in range(19, upper_limit, 2))
def main() -> None:
    start = time.perf_counter_ns()
    with ProcessPoolExecutor() as ex:
        for number, prime in zip(
            NUMBERS,
            ex.map(is_prime, NUMBERS),
            strict=False,
        ):
            print(f"{number} is prime: {prime}")
    end = time.perf_counter_ns()
    print(f"time: {(end - start) / 1e09} s")
if __name__ == "__main__":
    main()
```
- Took 3.085005714 s

### 59. Threads vs. Process - Recap

### 60. Asyncio
```py
# type: ignore
import asyncio
import sys
async def foo():
    print("start foo")
    await asyncio.sleep(2.0)
    print("end foo")
    return 0
async def bar():
    print("start bar")
    await asyncio.sleep(4.0)
    print("end bar")
    return 0
async def main_await() -> int:
    print("before await foo")
    await foo()
    print("after await foo")
    return 0
async def main_task() -> int:
    task = asyncio.create_task(foo())
    i = 2
    j = i * 2
    print(j)
    await task
    return 0
async def main_future() -> int:
    task_foo = asyncio.create_task(foo())
    task_bar = asyncio.create_task(bar())
    ret_foo = await task_foo
    print(ret_foo) # until bar is done, 2sec is available. We may do some jobs here
    ret_bar = await task_bar
    print(ret_bar)
    return 0
def main() -> None:
    # code = asyncio.run(main_await())
    # code = asyncio.run(main_task())
    code = asyncio.run(main_future())
    sys.exit(code)
if __name__ == "__main__":
    main()
```    

### 61. Asyncio Gather
```py
import asyncio
async def f(name):
    await asyncio.sleep(2)
    print(f"Task {name}")
    return name
async def main1():
    L1 = await f("A")  # noqa: N806
    L2 = await f("B")  # noqa: N806
    L3 = await f("C")  # noqa: N806
    print(L1, L2, L3)
async def main2():
    L = await asyncio.gather(f("A"), f("B"), f("C"))  # noqa: N806
    my_tasks = [f("A"), f("B"), f("C")]
    L = await asyncio.gather(*my_tasks)  # noqa: N806
    print(L)
if __name__ == "__main__":
    # asyncio.run(main1())
    asyncio.run(main2())
```

### Quiz 14: Multi-Threads and -Processes

## Section 13: Chapter 8: Conclusion of the course

### 62. Course conclusion

### 63. Bonus lecture


