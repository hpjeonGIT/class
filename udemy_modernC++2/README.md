## Modern C++: From Intermediate to Advanced
- Instructor: James Raynard

## Section 1: Introduction

### 1. Introduction to the Course

### 2. Lecturer Introduction

### 3. Guide to Exercises and Source Code
- Ref: https://github.com/JamesRaynard/

## Section 2: Review of C++

### 4. Local Variables and Function Arguments
- Local variables exist inside a scope
- Pass by value: `void func0(int y)`
- Pass by address: `void func1(int *y)`
- Pass by reference: `void func1(int &y)`
- Pass by const reference: read-only to class objects
```cpp
#include<iostream>
void func0(int y){ y = 0;};
void func1(int *y) { *y = 111;};
void func2(int &y) { y = 123;};
//void func3(const int &y) { y= 456;}; // cannot be compiled
int main()
{
  int x = 3;
  func0(x); std::cout << x << std::endl; // prints 3
  func1(&x); std::cout << x << std::endl; // prints 111
  func2(x); std::cout << x << std::endl; // prints 123
  //func3(x); std::cout << x << std::endl;
}
```
### Assignment 1: Local Variables and Function Arguments

### 5. Reference and Value Semantics
- Reference semantics
  - When initialized or assigned, new memory is not allocated but the reference is given
  - Garbage collector is used to manage the lifetime of these objects
  - Avoids the overhead of copying the object data
  - Overhead for garbage collection
    - Memory is not released immediately
    - Cannot predict when/in what order, objects will be destroyed
- C++ uses **value semantics by default**
  - Arguments are passed by value
  - Initialization creates an entirely new object
  - Objects exist only within a scope
- In order to have reference semantics:
  - Arguments are passed by reference
  - Initialization may create an alias to an existing object
  - To exist beyond the scope, use heap allocated arrays
- OLD C++ didn't have garbage collection
  - Only destruction by scope 
  - Allocated memory must be managed by the programmer

### 6. Declaration and Initialization
- Universal initialization
```cpp
int x{7};
string s {"hello world"};
vector<int> v{1,2,3,4};
int x = 7.7; // works but warning
in x{7.7}; //illegal
vector<int> x (4); // {0,0,0,0}
vector<int> x (4,2); // {2,2,2,2}
vector<int> x{4}; // {4}
vector<int> x{4,2}; // {4,2}
myClass x(); // ambiguous - object creation or function declaration?
myClass x{}; // object creation. 
```
- Instead of `typedef std::vector<int> IntVec;`, use `using IntVec = std::vector<int>`
- nullptr
  - A literal represening a null pointer
    - Works with any pointer type such as int *, double *, ...
    - Cannot be mapped into int or double or actual data type
  - In C, NULL is a macro pointing 0
    - Datatype might be different by a compiler
```cpp
#include <iostream>
void func(int i) { std::cout << "func(int) called\n"; }
void func(int *i) { std::cout << "func(int*) called\n"; }
int main()
{
  func(NULL); // not working with g++. works in VC++ ?
  func(nullptr); // prints func(int*) called
}
```

### 7. Classes
- A class is a compound data structure
- The public members provide the interface of the class: "What it does"
- The private members provide the implementation of the class: "How it does"
- Member functions are global functions
- `myClass x;`
  - `x.func(1,"three");` is called as `myClass::func(&x, 1, "three");`
  - The pointer of the object is available as `this`
  - Dereferencing "this" gives access to members of the object:`this->i=1;`

### 8. Special Member Functions
- Constructor
  - The same name as the class
  - Initializes a newly created object using its arguments
- Copy constructor
  - Similar to the constructor but uses another object for initialization
  - Always takes one argument which is a reference to another object of the same class
- Assignment operator
  - Assigns an existing object from another object
  - Always takes one argument which is a reference to another object of the same class
  - Returns a reference to the assigned object
- Destructor

### 9. Pointers and Memory
- A pointer is a variable whose value is an address in memory
- `new` operator allocates memory on the heap and returns the address of the memory
  - Calls the default constructor for the class
  - For a built-in type, the data is uninitialized
  - To initialize: `int *p3 = new int {36}`
- Failing to release memory when it is no longer needed causes a **memory leak**

### 10. Array, String and Vector
- Array
  - An indexed block of contiguous memory
  - Inherited from C
  - Allocated on stack
  - The size must be known at compile time
    - No dynamic size
- Dynamic array
  - Must be allocated on the heap if
    - the size is unknown at compile time
    - May vary the size
  - Must be deallocated explicitly
- C-style string
  - An array of const char
  - Has an null extra character in the end
  - String literals are C-style strings
    - `const char *str= "hello";` is eauivalent to `const char str[] = {'h','e','l','l','o','\n'};`
- std::string
  - A class
  - Has a member which is a dynamic array
  - Behaves like a dynamic array
  - Contiguous block of memory allocated on heap
  - Released in destructor in the end of scope
- std::string interface
  - [] is supported
  - size() function to return the size
- std::vector
  - Can store data of any single type

### Assignment 2: Classes and Strings
```cpp
#include<iostream>
#include<string>
class myURL {
public:
 myURL(std::string &x, std::string &y) : protocol{x}, resource{y} {} // for myURL obj2(a,b);
 myURL(std::string &&x, std::string &&y) : protocol{x}, resource{y} {} // for myURL obj("http", "www.example.com/index.html");, using R-value reference
 void display() {
  std::cout << protocol << "//" << resource << std::endl;
 }
 ~myURL() {}
private:
 std::string protocol;
 std::string resource;
};
int main() 
{
  std::string  a = {"http"};
  std::string b = {"www.example.com/index.html"};
  myURL obj("http", "www.example.com/index.html");
  obj.display();
  myURL obj2(a,b); obj2.display();
}
```

### 11. Conway's Game of Life Overview

### 12. Two-Dimensional Arrays

### Assignment 3: Two-Dimensional Arrays
```cpp
#include<iostream>
int main() {
  int arr2d[2][3] = { {1,2,3}, {4,5,6}};
  std::cout<< "row1/col2 = " << arr2d[1][2] << std::endl;
  for(int i=0; i<2; i++) {
    for(int j=0; j<3; j++) {
      std::cout << i << j << " " << arr2d[i][j] << std::endl;
     }
  }
  int arr1d[6] = {1,2,3,4,5,6};
  for(int i=0; i<2; i++) {
    for(int j=0; j<3; j++) {
      std::cout << i << j << " " << arr1d[i*3+j] << std::endl;
    }
  }
  return 0;
}
```

### 13. Conway's Game of Life Practical

### 14. Conway's Game of Life Practical Continued

### 15. Numeric Types and Literals
- char: 8bits
- int: 16bits or more
- long: 32bits or more
- long long: 64bits or more
- Fixed-width integers
  - Since C++11
  - int8_t, int16_t, int32_t, int64_t
  - uint8_t, uint16_t, uint32_t, uint64_t
- Decimal as default
  - 0x for hexadecimal number: `int x = 0x2a;`
  - 0 for octal: `int oct = 052;`
  - 0b or 0B for binary: `int bin0 = 0b101010;`
- Floating point
  - float: usually 6 digits precision
  - double: usually 15 digits precision
  - long double: usually 20 digits precision
- Digit separator
  - `const int oneM = 1'000'000;`
  - `double pi = 3.141'593;`
  - "'" can be anywhere
- Suffice:
  - f for float: `auto f = 3.1415f;`
  - ULL for unsigned long long: `auto x = 123456780ULL;`
```cpp
#include<iostream>
int main() 
{
  auto i = 123'45'678'9ULL;
  auto x = 3.1'41'56d; // separator can be located anywhere
  auto k = 052; // prints 42 as this is octal: 5*8+2*1 = 42
  std::cout << i << " "<<x << " " <<k <<std::endl;
  return 0;
}
```

### 16. String Literals
- Some texts inside of quotes "..."s
  - Not std::string
- C-style string literal
  - An array of const char, terminated by a null character
  - Very limited range of operations
  - Faster than std::string
- From C++14, we can create std::string literals
  - String literals can be converted into string by adding "s"
    - This is an operator from std::string_literals namespace
    - `std::string myString = "hello"s;` is equivalent to `std::string_literals::operator""s("hello", 5)`
  - "sv" for string_view object
```cpp
#include<iostream>
#include<string>
using std::string_literals::operator""s;
int main() {
  std::string str1 = {"hello world1"};
  std::string str2 = "hello world2";
  std::string str3 = "hello world3"s;
  std::cout << str1 << std::endl;
  std::cout << str2 << std::endl;
  std::cout << str3 << std::endl;
  //std::cout << "hello" +"world!"  << std::endl; // compile error
  std::cout << "hello "s +"world!"s  << std::endl; // now works
  return 0;
}
```
- Raw string literals
  - To avoid backslashitis
  - Inside of `R"(...)"`
    - Or `R"x(...)x"` or `R"!(...)!"`... Use your own delimiter
  - `string url = R"x(<a href="file">C:\Program Files\</a>\n)x"`

### 17. Casting
- Explicit conversion
  - Different datatype
  - Const expression into non-const equivalent
  - pointer to base class object to pointer of derived
- static_cast: `static_cast<char>(c)`
- const_cast: makes const into non-const
```cpp
#include <iostream>
void print(char *str) { std::cout << str << std::endl;}
int main() {
   const char *msg = "hello world";
   //print(msg); //error: invalid conversion from ‘const char*’ to ‘char*
   print(const_cast<char *> (msg));
   return 0;
}
```
  - In this code, if print() changes (mutates) str variable, undefined behavior may occur
- reinterpret_cast: mainly used in low-level work like HW communication, OS, ...
- dynamic_cast: converts a pointer to a base class object to a pointer to a derived class object
  - This is done at runtime

### 18. Iterator Introduction
- std::string has two member functions which return iterators
  - begin(): an iterator to the first element
  - end(): an iterator corresponding to **the element after the last element**
    - This is an invalid iterator and must not be dereferenced
```cpp
std::string::iterator it = str.begin();
while (it!=str.end()) {
  std::cout << *it << ",";
  ++it;
}
```
- For C-style string:
```c
const char str[] = {'H','e','l','l','o'};
const char *p = str;
const char *pEnd = str + 5;
while (p != pEnd) {
  std::cout << *p << ",";
  ++p;
}
```

### 19. The auto keyword
- `auto str1="Hello";` is equivalent to `const char str1[]="Hello"`
- `auto str2 = "Hello"s;` is equivalent to `std::string str2 {"Hello"s}`
- With qualifiers?
  - auto will give the underlying type
  - const, reference, etc are ignored
```cpp
const int& x{6};
auto y=x;
++y; // not const
```
- In order to enforce qualifiers, add explicitly
```cpp
const auto &y = x;
```
- When to use auto
  - When the type doesn't matter
  - When the type does not provide useful info.
  - When the code is clearer without the type: Ex) iterator
  - When the type is difficult to discover: Ex) template metaprogramming
  - When the type is unknown: Ex) compiler generated class
- When not to use auto
  - When we want a particular type
  - When it makes the code less clear

### 20. Loops and Iterators
- Const iterator: prevents the loop from modifying the data
  - cbegin(), cend()
- Reverse iterator: iterates backwards
  - rbegin(), rend()
  - crbegin(), crend()
- Global begin(), end() by C++11
  - This works for built-in arrays as well
  - C++14 adds cbegin(), cend(), rbegin(), rend(), crbegin(), crend()
```cpp
#include<iostream>
#include<string>
int main() {
  std::string msg = "hello";
  for (auto it=std::begin(msg); it !=std::end(msg);++it) std::cout << *it ;
  std::cout << std::endl;
  for (auto it=std::crbegin(msg); it !=std::crend(msg);++it) std::cout << *it ;
  std::cout << std::endl;
  return 0;
}
```
- Range for loops
  - Special concise syntax for iterating over containers
  - `for (auto el:vec){...}` is equivalent to `for(auto it=std::begin(vec); it!=std::end(vec);++it){...}`
  - To modify element, use a reference
```cpp
for (auto& el: vec)   el+=2; // adds 2 into each element
```
  - Suitable if we visit each element once, in order, without adding or removing elements
  - Otherwise, use a traditional loop

### Assignment 4: Loops and Iterators
```cpp
#include<iostream>
#include<vector>
int main() {
  std::vector<int> x {1,2,3};
  for(auto it=x.begin(); it!=x.end();it++) std::cout << *it <<std::endl;
  for(auto it=x.begin(); it!=x.end(); it++) *it *=3;
  for(auto it=x.begin(); it!=x.end();it++) std::cout << *it <<std::endl;
  return 0;
}
```

### 21. Iterator Arithmetic and Iterator Ranges
- Arithmetic on iterators
  - Similar to pointers
  - `auto second = std::begin(msg) + 1;`
  - `auto last = std::end(msg) - 1;`
  - `auto nsize = std::end(msg) - std::begin(msg);`
  - next(): returns the following iterator
  - prev(): returns the previous iterator
  - distance(): returns the number of steps needed b/w iterators. Ex) `std::distance(std::begin(msg),std::end(msg))`
  - advance(): moves an iterator by the given increment (2nd argument)
- Half-open range
  - Ex: `for(int i=0;i<10;++i) {...}`
  - This is written as [0,10)
    - i==10 is NOT allowed
    - `it != std::end(msg)` works the same way

### 22. If Statements and Switch in C++17
- Initializer in if statement
  - For the code `auto iter=std::begin(msg); if (iter != std::end(msg)) {...}`
  - C++17 allows an initializer in an if statement: `if (auto iter=std::begin(msg); iter!=std::end(msg)) {...}`
    - `iter` is local to the if statement
    - Includes else block
  - We don't need to declare a variable before the if statement. The used variable is local and the memory will be deallocated after the corresponding if statement block
```cpp    
#include<iostream>
#include<string>
int main() {
  std::string msg = "hello";
  if(auto iter = std::begin(msg); iter != std::end(msg))
    std::cout << *iter << std::endl;
  return 0;
}
```
- Initializer in switch statement
  - Since C++17:
```cpp
for (int i=0; i<10; i++) {
  switch(const char c=arr[i]; c) {
    case 'h':
    ...
  }
}
```
- Fallthrough attribute
  - In switch block, if `break;` is missing for a case, code will fall through to the next case
  - Since C++17, we may use `[[fallthrough]];` explicitly to help readability
```cpp
#include<iostream>
#include<string>
int main() {
  std::string msg {"hello"};
  for(auto el: msg) {
    switch(el) {
      case 'h':
        std::cout << "h";
      case 'e':
        std::cout << "e";
        [[fallthrough]];
      case 'l':
        std::cout << "l";
        break;
      case 'o':
        std::cout << "o";
        break;
   }
   std::cout << std::endl;
  }
}
```
- Results:
```bash
$ ./a.out 
hel
el
l
l
o
```

### 23. Templates Overview
- Generic programming
- Writes a code which is functionally the same but operates on different types of data
  - vector of int/float/double
- Instantiation of the template is done at compile-time, detecting the datatype
- Constructor Argument Deduction in C++17
  - `std::vector v{1,2,3};`: C++17 deduces as vector<int>
- `template<class T>` vs `template<typename T>`
  - typename was introduced by C++98

### 24. Namespaces
- Groups together logically related symbols
- For referring to the global namespace, use `::` as a prefix
- Name hiding
  - When a variable is defined in a namespace, it "hides" other variable outside the namespace with the same name
```cpp
#include<iostream>
int x = 12;
namespace abc {
    int x = 34;
  void func() {
    std::cout << "within abc " << x << std::endl; // 34
    std::cout << "global " << ::x << std::endl;   // 12
  } 
} 
int main() {
 abc::func();
 std::cout << x <<  std::endl;     // 12
 std::cout << abc::x << std::endl; // 34
 return 0;
}
```
- Using declaration
  - `using` will bring a particular namespace into global

### 25. Function Pointer
- A function's executable code is stored in memory
- Can get a pointer whose address is the start of this exe
```cpp
void func(int,int);
auto func_ptr = &func;
// Equivalent to void(* func_ptr)(int,int) = &func;
```
- We may use a type alias for a function pointer's type
  - `using pfunc = void(*)(int,int)`
    - void comes from the function return type
  - This is necessary to define the function pointer type for another function's argument
- A function pointer is a **callable object**
  - Behaves like a variable
  - Can be called like a function
- We can call the function by dereferencing
  - `(*func_ptr)(1,2);`
- A function pointer is a **first class object**
  - We can pass a function pointer as argument to another function
    - `void func2(int,int,pfunc) {}`
  - We can return a function pointer from a call to another function
```cpp
#include<iostream>
int func1(int x, int y) { return x+y;}
using pfunc1 = int(*)(int,int); // function pointer type
void func2(int x,int y, pfunc1 fptr) { std::cout <<(*fptr)(x,y) << std::endl;} // receives function pointer as an argument and executes
pfunc1 call_func1() { return &func1; } // returns a function pointer
int main() {
  auto my_ptr = call_func1();
  func2(1,2,my_ptr);
 return 0;
}
```
- Pros and Cons of function pointers
  - Inherited from C
  - Useful for writing callbacks
    - OS, GUI's, event-driven code
  - Raw pointer
  - Syntax is ugly

## Section 3: C++ String Interface

### 26. Basic String Operations
- Mixture of own interface + STL
- Assignment: s1=s2;
- Appending: s1+=s2;
- Concatenation: s3 = s1+s2;
- Comparison: s1 == != < > <= >= s2;
- C-style string
  - std::str has a member function c_str()
    - This returns a copy of string's data as a C-style string
    - Array of char terminated by null character
    - `const char *pChar = msg.c_str();`
- substr()
  - Returns partial data as given from arguments
    - `s2 = msg.substr(6,2);` returns 6-7th data
- Constructor
  - Default: `string msg;`
  - Constructor with a string literal: `string msg {"Hello"};`
  - Constructor with a count and an initial value: `string msg(3,'x');` yields "xxx"
  - Constructor with an initializer list: `string msg{'H','e','l','l','o'};`
  - From a substring

### 27. Searching Strings
- find(): case sensitive search
  - The argument can be a char, std::string, or C-style string
  - Returns the index of the first search
  - When no match, returns std::string::npos (this is the largest number in size_t)
- rfind(): finds the last occurrence
- find_first_of()/find_last_of(); first or last occurrence of any character from the argument string
- find_first_not_of()/find_last_not_of(): first or last occurrence of any character not in the argument

### 28. Adding Elements to Strings
- append()
  - msg.append("world"s);
  - += operator works as well
- insert()
  - Adds characters before a specified position
  - Can take an iterator: Ex) `msg.insert(std::end(msg)-1,"world");`
    - Note that the iterator of the old string will not work on the new string

### 29. Removing Elements from Strings
- erase()
  - `msg.erase(3,1);`
  - Can use iterators: `msg.erase(std::begin(msg)+1, std::end(msg)-1);`
- replace()
  - `msg.replace(std::begin(msg),std::begin(msg)+3, "wow");`
- assign(): replaces entire strings with the argument  

### 30. Converting between Strings and Numbers
- C-style: atoi(), atod(), ...
- C++11 provides `to_string()`
  - `std::string pi {to_string(3.14159)};`
- stoi(): string to int
  - Leading whitespace is ignored
  - Error handling
    - 2nd argument gets the number of characters which were processed
    - When successful, it must be equal to the string's size
    - When partially successful, this gives the index of the first non-numeric character
    - When conversion fails completely, it throws an exception

### 31. Miscellaneous String Operations
- data() member function
  - Returns a pointer to the container's internal memory buffer
  - For std::string, this is null-terminated (equivalent to c_str())
  - Useful for working with APIs written in C
- swapping
  - `msg1.swap(msg2);`
  - `std::swap(msg1,msg2);`: works for other datatypes
- Default implementation of swap()
  - Using a temporary object
  - But this might be inefficent for string as it has to allocate many memories
    - Instead of using temporary object, it just exchanges the headers of strings
      - Memory address and the size of string

### 32. Character Functions
- Inherited from C
- Defined in <cctype>
- isupper()
- islower()
- ispunct()
- isspace()
- isdigit()
- toupper()
- tolower()
- How to do case-insenstivie string comparison
  - No direct support from C++ standard
  - Compiler dependent functions
  - The simplest solution is to convert all strings into upper or lower cases then compare

### Assignment 5: Character Functions
```cpp
#include<iostream>
#include<string>
std::string exclaim(std::string& s1) {
  std::string s2 {};
  for (auto el: s1) {
    if (ispunct(el)) s2 += "!";
    else s2+= el;    
  }
  return s2;
}
int main() {
  std::string s1 {"To be, or not to be, that is the question:"};
  std::cout << s1 << std::endl;
  std::cout << exclaim(s1) << std::endl;
  return 0;
}
```

## Section4: Files and Streams

### 33. Files and Streams
- File interactions are represented by fstream objects
  - Similar to iostream
- fstream objects always access files "sequentially"
  - A sequence of bytes
  - In order
  - Of unknown length
  - With no structure
- fstream do not understand the file formats
- C++ fsream operations
  - Open
  - Read
  - Write
  - Close
  - For each operation, fstream object will call a function in OS API
  - The program stops and waits until the operation is performed
  - When OS completes the task, API call will return
  - The program then resumes
- When C++ program terminates, the runtime will make close all open files
  - But it is good practice to close all files when done

### 34. File Streams
- iostream
  - ostream (cout)
  - istrem (cin)
- fstream
  - ofstream
  - ifstream
- How to open a file for reading
  - Pass file name into the fstream constructor
    - By C++11, string can be used
  - Check the stream's state before using it
- Reading from a file
  - The same way as std::cin
```cpp
while (ifile >> text)  
  std::cout << text;
```
  - One word at a time
- Reading a complete line: getline()
```cpp
while(getline(ifile,text)) std::cout << text << std::endl;
```
  - Will return false in the end
- How to open a file for writing
  - `std::ofstream ofile("text_out.txt");`
  - Check the stream state before using it
- fstream destructor
  - When called, the file is automatically closed
  - But it is good practice to close when done

### 35. Streams and Buffering
- C++ streams use buffering to minimize calls to the OS
- Flushing: empties buffer when the buffer is full
- For ostream, stream buffer flushing depends on the terminal configuration
- ofstream is flushed when the buffer is full
- There is no direct way to flush ifstream
- std::flush
  - Enforces flushing
  - Affects the performance
    - Use when necessary only

### 36. Unbuffered Input and Output
- When stream buffering is not suitable
- Ex) network application
  - Data must be trasmitted in packets of a specified size
  - Data may need to be transmitted at specific times
- streams have member functions for reading/writing a single character: get(), put()
- For many characters: read(), write()  
  - Buffer must be large enough
- gcount(): returns the number of characters that were actually received
  - Then Allocate array to process the data

### 37. File Modes
- Options for opening a file
- By default, text mode
- By default, output files are opened in truncate mode
  - Overwriting on existing data
- To open an ofstream in append mode: `ofile.open("out.txt", std::fstream::app);`
- Binary mode
  - Identical to the data in memory
  - Low level and error prone. Could be portability issues
  - Should be avoided wherever possible
  - Work with file formats (image/media which are well-known)
- Other modes
  - trunc: truncate mode
  - in: for input
  - out: output mode + truncate mode
  - ate: similar to append but output can be written anywhere
- Combining modes
  - `file.open("my.txt", std::fstream::out | std::fstream::app);`
- File mode restrictions
  - out: fstream and ofstream only
  - in: fstream and ifstream only
  - trunc: only in output mode
  - app: cannot combile with trunc. Only in output mode  

### 38. Stream Member Functions and State
- open()
- is_open(): check if the file is open or not
- Stream state member functions
  - good(): returns true when the input was read successfully
  - fail(): returns true when there was a recoverable error such as wrong kind of data
  - bad(): returns true when there was an unrecoverable error such as media failure
  - clear(): restores the stream's state
  - eof(): returns true when the end of file has been reached  
- When std::cin is sent to a wrong data type or buffer size error occurs, next std::cin will not work as buffer is not emptied
  - No flush in input stream
  - ignore(): will ignore the buffer - may use maximum limit to ignore very long input

### 39 Stream Manipulators and Formatting
- Afftects the stream's behavior
- Such as std::flush and std::endl
- std::boolalpha: prints true/fase instead of 1/0
- std::setw(): width of data output field
  - Non-sticky manipulator. Doesn't affect sequential streaming
- std::setfill(): set padding with a given character
```cpp
#include<iostream>
#include<iomanip>
int main() {
  std::cout << std::setfill('#');
  std::cout << std::left << std::setw(15) << "Hello " << 5 << "\n";
  std::cout << std::right;
  std::cout << std::setfill('_');
  std::cout << std::setw(15) << "Pola bears " << std::endl;
  return 0;
}
```
- Result:
```bash
$ ./a.out 
Hello #########5
____Pola bears 
```

### 40. Floating-point Output Formats
- Scientific notation
  - mantissa and exponent
  - `std::cout << std::scientific << x << std::endl;`
- Fixed output
  - `std::fixed`
- Restoring floating-point defaults
  - All manipulators except setw() are all sticky - permanently change the behavior of the stream
  - `std::setprecision()`

### 41. Stringstreams
- The basic C++ stream is represented by std::ios
- iostream
  - ostream
  - istream
- fstream
  - ofstream: file stream for writing
  - ifstream: file stream for reading
- stringstream
  - ostringstream: string stream for writing
  - istringstream: string stream for reading
- stringstreams are defined in `<sstream>`
  - Wrapper around std::string
- ostringstream
  - An empty std::string object
  - Can store data in this string
  - str() member function will return a copy of the stream's string
    - May convert any input (int/double/...) into string data using str(): `outss << input; return outss.str();`
- istringstream
  - Uses a copy of an existing string
  - This string is passed to the istringstream's constructor
  - Can manipulate a line data into words or pieces
- Applications of stringstreams
  - ostringstreams are useful when interfacing to systems that expect strings in particular formats
    - Also may convert non-string datatype into string automatically with concatenation
  - istringstreams can be used with getline() to process input more easily than the >> operator
    - When we extract words from line data, iss is much easier than other operations  

### Assignment 6: Files Workshop Part One
```cpp
#include<iostream>
#include<iomanip>
#include<fstream>
#include<vector>
#include<string>
struct language {
  std::string lang;
  std::string designer;
  int date;
}; 
int main() {
  std::vector<language> x;
  std::ifstream inp("language.txt");
  if (inp.is_open()) {
    std::string oneline;
    while(std::getline(inp, oneline)) {
      std::istringstream iss(oneline);
      language tmp;
      iss>> tmp.lang >> tmp.designer >> tmp.date; // will date (year) be converted into interger?
      x.push_back(tmp);
    }
  }
  inp.close();
  for (auto el: x) std::cout << el.lang <<" " << el.designer << " "<< el.date <<
 std::endl;
  return 0;
}
```

### Assignment 7: Files Workshop Part Two
```cpp
#include<iostream>
#include<iomanip>
#include<fstream>
#include<vector>
#include<string>
struct language {
  std::string lang;
  std::string designer;
  int date;
}; 
int main() {
  std::vector<language> x;
  std::ifstream inp("language2.txt");
  if (inp.is_open()) {
    std::string oneline;
    while(std::getline(inp, oneline)) {
      std::istringstream iss(oneline);
      language tmp;
      iss >> tmp.lang; // one word for lang
      std::string oneword,lastword,sumword;
      while(iss >> oneword) { sumword += lastword + " "; lastword = oneword;}
      tmp.designer = sumword;
      tmp.date = std::stoi(lastword);
      x.push_back(tmp);
    }
  }
  inp.close();
  for (auto el: x) std::cout << el.lang <<" " << el.designer << " "<< el.date <<
 std::endl;
  return 0;
}
```

### 42. Resource Management
- Resources
  - Heap memory
  - files
  - Database connections
  - GUI windows
- They must be managed:
  - Must allocate, open or acquire the resource before use
  - Must release or close the resource after use
  - Be careful when copying the resource - copying could be expensive
  - Must think about error handling
- fstream and resources
  - We use the fstream interface, without knowledge of how to handle files directly
- Classes which manage resources
  - Common idiom
    - Resource is stored as a private member
    - Constructor acquires the resource
    - Public member functions control the access to the resource
    - Destructor releases the resource
  - This is known as RAII - Resource Acquisition Is Initialization
  - When an object is copied or assigned to, the target object acquires its own version of the resource

### 43. Random Access to Streams
- Stream position marker
  - This keeps track of where the next read/write operation will be performed
  - We may alter its position
    - fstream not opened in "app" mode
    - stringstream
- `seek` member function: sets the current position
- `tell` member function: tells the current position
- Supported seek and tell operations
  - `seekg` and `tellg` for input streams ("get operations")
  - `seekp` and `tellp` for output streams ("put operations")
  - Do not use them for iostreams
  - For fstream with app mode, seepk has no effect
    - always the end of the file
  - tellg and tellp
    - Return a `pos_type` object
    - When operation fail, it returns -1
  - seekg and seekp
    - takes a `pos_type` argument
    - May use `std::io_base::beg/end/cur` as base positions
- File modification
  - The best way to modify a file is usually
    - Read it into a istringstream
    - Ge the bound string and make the changes
    - When ready, overwrite the original file
  - Seek and tell operations can be used to modify a file in-place

### 44. Stream Iterators
- STL provides iterators on streams
  - Defined in `<iterator>`
  - They must be instantiated with the types of the data
- Stream iterators have a very limited inteface
  - Assigning to an ostream_iterator will put an object on the stream
  - Dereferencing an istream_iterator will get the object at the current position in the stream
  - Incrementing an istream_iterator will move the stream's position marker to the next object
- ostream_iterator example:
```cpp
#include<iostream>
#include<iterator>
int main(){
  std::ostream_iterator<int> oi(std::cout,"\n"); // "\n" is inserted after every data
  for(int i=0;i<3; i++) { *oi = i; ++oi;}
  return 0;
}
```
- Results:
```bash
$ ./a.out 
0
1
2
```
- istream_iterator
  - We may use an empty iterator to find out the end of input
```cpp
#include<iostream>
#include<iterator>
#include<vector>
int main(){
  std::istream_iterator<int> ii(std::cin);
  std::istream_iterator<int> eof;
  std::vector<int> vi;
  while (ii != eof) {
    vi.push_back(*ii);
    ++ii;
  }
  for (auto x: vi) std::cout << x << std::endl;
  return 0;
}
```
- Result:
```bash
$ ./a.out 
1 2
3 11 123  // Enter ctrl-d here
1
2
3
11
123
```

### 45. Binary Files
- Binary mode to work with: `ofile.open('image.bmp', std::fstream::binary);`
- `<<` and `>>` operators are not suitable
- Must use write() and read()
- Memory alignment
  - Modern HW is optimized for accessing data which are word-aligned
  - On 32bit system, addresses are multiples of 4
    - If not, accessing performance will be very poor
  - Padding
    - If a struct is not word-aligned, compilers will add extra bytes
    - `#pragma pack()` can employ different offsets but this is non-standard
  - alignas
    - Since C++11
    - Enforce compiler to align element
```cpp
#include <iostream>
#include <fstream>
#include <cstdint>
using namespace std;
//#pragma pack(push, 1)
struct point {
	char c;
	int32_t x;
	int32_t y;
};
//#pragma pack(pop)
int main() {
	point p{'a', 1, 2};
	ofstream ofile("file.bin", fstream::binary);
	if (ofile.is_open()) {
		ofile.write(reinterpret_cast<char *>(&p), sizeof(point));
		ofile.close();
	}
	ifstream ifile("file.bin", fstream::binary);
	point p2;
	if (ifile.is_open()) {
		ifile.read(reinterpret_cast<char *>(&p2), sizeof(point));
		ifile.close();
		cout << "Read " << ifile.gcount() << " bytes\n";
		
		cout << "Read x = " << p2.x << ", y = " << p2.y << endl;
	}
}
```

### 46. Binary File Practical
- We use a bitmap file as a demo
- Bitmap file format
  - File header
```cpp
#pragma pack(push, 2)                    // The elements must start on 16-bit intervals
struct bitmap_file_header {
    char header[2] { 'B', 'M' };
    int32_t file_size;
    int32_t reserved { 0 };
    int32_t data_offset;
};
#pragma pack(pop)              
```
  - Info header
```cpp
 struct bitmap_info_header {
    int32_t header_size{40};
    int32_t width;
    int32_t height;
    int16_t planes{1};
    int16_t bits_per_pixel{24};
    int32_t compression{0};
    int32_t data_size{0};
    int32_t horizontal_resolution{2400};
    int32_t vertical_resolution{2400};
    int32_t colours{0};
    int32_t important_colours{0};
};
``` 
  - Image data
    - Pixels
```cpp    
#pragma pack(push, 2)                    // The elements must start on 16-bit intervals
struct pixel {
	uint8_t blue;
	uint8_t green;
	uint8_t red;
};
#pragma pack(pop)                  
```
    - Pixel position dependes on its coordinate
- Bitmap class
```cpp
class bitmap {
private:
	int width{800};
	int height{600};
	std::string filename;                          // The name of the bitmap file
	std::vector<pixel> pixels;                     // Vector containing the image data
public:
	// Constructor
	bitmap(std::string filename) : filename(filename), pixels(width*height) {}
	void set_pixel(int x, int y, pixel p);         // Set the pixel at (x, y)
	void set_row(int rownum, pixel p);             // Set all the pixels in an entire row
	void set_all(pixel p);                         // Set all the pixels in the image
	bool write();                                  // Save the image data to file
};
#endif /* BITMAP_H_ */
```
- Writing the bitmap file
```cpp
  // Write the File Header
	ofile.write(reinterpret_cast<char *>(&file_header), sizeof(bitmap_file_header));
	// Write the Info Header
	ofile.write(reinterpret_cast<char *>(&info_header), sizeof(bitmap_info_header));
	// The first argument to write is an array containing the image data
	// The second argument is the size of the data
	ofile.write(reinterpret_cast<char *>(pixels.data()), pixels.size() * sizeof(pixel));
```

## Section 5: Special Member Functions and Operator Overloading

### 47. Constructors in Modern C++
- If member data are not initialized properly, random results might be produced
  - Assign default value or initialize when constructed

### 48. Copy Constructor Overview
- Takes only one argument - an object of the same class
  - Or using "=": `Test test3 = test1;`
- The argument is passed by reference: `Test(const Test& other);`
- If we do not implement a copy constructor for our class, the compiler will **synthesize** one for us
  - This will copy all the data members
  - Usually this is good enough
```cpp
#include<iostream>
class myTest {
public:
  myTest(int a, float b): x(a), y(b){ std::cout<< "default constructor\n";}
  myTest(const myTest& abc): x(abc.x), y(abc.y) {std::cout << "copy constructor\n";}
private:
  int x;
  float y;
};
int main() {
  myTest alpha(123, 0.12f);
  myTest beta = alpha;
  return 0;
}
```

### 49. Assignment Operator Overview
- "operator=" is a member function
- `y=z;` is equivalent to `y.operator=(z);`
  - `Test y = z;` calls copy constructor
  - `y = z;` calls operator=
  - Members of y will have the same value as those of z
  - The assignment operator takes its argument by **const reference**
- If we don't implement an assignment operator, then compiler will **synthesize** one for us
- Signature will be: `Test& operator=(const Test& arg)`
  - Unlike constructor, the declaration must return the class object (`return *this;`)
```cpp
#include<iostream>
class myTest {
public:
  myTest(int a, float b): x(a), y(b){ std::cout<< "default constructor\n";}
  myTest(const myTest& abc): x(abc.x), y(abc.y) {std::cout << "copy constructor\
n";}
  myTest& operator=(const myTest& ex) { std::cout <<"operator=\n"; x= ex.x; y = ex.y; return *this; } // cannot use list initializer
private:
  int x;
  float y;
};
int main() {
  myTest alpha(123, 0.12f), gamma(456, 7.3f); // default constructor
  myTest beta = alpha; // copy constructor
  gamma = beta; // operator=
  return 0;
}
```

### 50. Synthesized Member Functions
- Compilers may synthesize for us if they don't exist
  - Constructor: no arguments
  - Copy constructor: member data will be copied
  - Assignment oprator
  - Move operations
  - Destructor
- Drawbacks of synthesized functions
  - Built-in type members are default initialized
  - Pointer members are "shadow copied"
    - When deep copy is required, implement copy constructor and assignment operator

### 51. Shallow and Deep Copying
- Let's have a class allocating heap memory
  - RAII idiom
  - Make sure the resource is allocated before being used: memory allocation at constructor
    - When there is no argument, make default constructor initializes the memory related variables
  - Make sure the resource is released when it is no longer needed: memory deallocation at destructor
  - Make sure that copying resource is handled correctly: implement copy constructor
  - Make sure that any transfer of the resource to another object of the class is handled correctly: implement operator=
    - Check if the argument is not same as self: `if(&arg !=this){...}`
    - Deallocate the self heap memory then allocate again using the size from argument object
    - Then copy data

### 52. Copy Elision
- Compiler is allowed to skip over a call to the copy constructor in some cases
  - A kind of optimization
- This happens when copying temporary variables during function calls
- Modern compilers will apply copy elision if they can

### 53. Conversion Operators
- A class can define a conversion operator as a member function
- When certain datatype is required, the corresponding conversion operator is called
- Ex:
  - `Test test; int x = test + 5;`
  - First, the compiler tries to find an exact match
    - Check operator+
  - Second, the compiler tries to convert the arguments
    - Check conversion operator into int type
    - `int x = test + 5;` into `int x = test.operator int() + 5;`
- This is called implicit conversion
  - May yield undesired results
- In mordern C++, explicit conversion is required
  - When use, appropriate cast is required as well
```cpp
#include<iostream>
class myClass {
private:
  int i{123};
  float x;
public:
  explicit operator int() const { return i;} // explicit conversion
};
int main() {
   myClass q;
   int x = static_cast<int>(q) + 5; // casting is required
   std::cout << x << std::endl;
   return 0;
}
```
- Explicit constructor
  - When a single argument is given
```cpp
class Test { 
  int i;
public:
  explicit Test(int i): i(i) {}
};
...
Test test = 4; //Error
Test test = Test(4); // Explicitly creates an object
```

### 54. Default and Delete Keywords
- Compiler may synthesize special member functions
```cpp
class Test {
  public:
   Test() = default;             // default constructor
   Test(const Test&) = default;  // copy constructor
   Test& operator= (const Test&) = default; // assignment operator
...   
};
```
- Making a class uncopyable
  - In traditional C++, we may copy constructor and assignment operator private, preventing objects from being copied
  - In modern C++, we define them as delete then the same effect
```cpp
class myClass {
public:
  myClass() = default;
  myClass(const myClass& arg) = delete;
  myClass& operator=(const myClass& arg) = delete;
};
int main() {
  myClass a,c;
  //myClass b=a; // not allowed
  //c = a;       // not allowed
  return 0;
}
```
- Deleted functions
  - Defined but cannot be called
  - Mostly used for copy operators and default constructor
  - If a compiler cannot synthesize with default (when =default is requested), it will synthesize as =delete
    - This may happen when member data don't allow automatic initialization

### 55. Operators and Overloading
- Operators for built-in types
  - `c=a+b`: calls operator +(a,b) and stores the return into c
  - `a==b`: calls a.operator==(b)
- Unary and binary operators
- Ternary operator: `a?b:c` = `?(a,b,c)`
- Operators take either one or two arguments
- User defined operators
  - Operator overloading

### 56. Which Operators to Overload
- In general programming, the most useful operators are
  - =
  - ==
  - !=
  - <
  - Function call operator ()
- Operators which should not be overloaded
  - AND/OR (&&/||)
  - Address-of(&) and comman (,)
  - Scope(::), dot(.), * operator and ternary
    - C++ doesn't allow such overload
- Recommendation
  - Think carefully when adding operators to class
  - Return value must correspond to the C++ equivalent operator
    - Bool for logical/relational operators
    - The class type (by value) for arithmetic operators

### 57. The Friend Keyword
- Using non-member functions in a class
  - A class can introduce a global function to its method
  - This is a non-member function
  - The non-member function cannot access private data
```cpp
#include<iostream>
class myClass {
 private:
  int i{123};
 public:
  void print_global(const myClass& arg);
};
void print_global(const myClass& arg) { std::cout << "global function called\n";}
int main() {
  myClass myc;
  print_global(myc);
  return 0;
}
```
- Such non-member function cannot access private data
  - Use friend keyword to enable the access
```cpp
class myClass {
 private:
  int i{123};
 public:
  friend void print_global(const myClass& arg);
};
```
- If we add friend to another class, then the member functions of another class can access to the public/private members of the class
- How to avoid using friend
  - Using friend may hurt encapsulation of OOP
  - May define the non-member function to call the class's member function

### 58. Member and Non-member Operators
- Sometimes we need to use non-member functions
  - Some operators cannot be implemented as member functions
- When to use member operations?
  - Compound assignment +=, -=, *=, /=
  - Increment ++, decrement --
  - Assignment=
  - subscript[]
  - function call()
  - arrow->
- When to use non-member operators?
  - Arithmetic operators
  - Equality and relational operators
  - Bitwise operators
  - Input/output operators << and >>

### 59. Addition Operators
- Plus operator summary
  - Prototype: `T operator+(const T&lhs, const T& rhs);`
  - How to invoke: `a+b;`
  - Called as: `operator+(a,b);`
  - Return value: the sum of the two objects, returned by value
  - This is defined as a non-member function
    - Allows for symmetry in implicit type conversion and increasing encapsulation
- += operator summary
  - Prototype: `T& operator +=(const T&rhs);`
  - How to invoke: `a+=b;`
  - Called as: `a.operator+=(b);`
  - Returns the modified first object by reference
  - This is defined as a member function
```cpp
#include <iostream>
using namespace std;
// Avoid potential confusion with std::complex
class Complex {
private:
	double real{0.0};
	double imag{0.0};
public:
	Complex(double r, double i): real(r), imag(i) {}
	Complex(double r): real(r) {}
	// We define the += operator as a member function
	// This adds the real and imaginary parts separately and returns the modified object
	Complex& operator +=(const Complex& rhs) {
		real += rhs.real;                     // Assign new value of real member
		imag += rhs.imag;                     // Assign new value of imag member
		return *this;                         // Return modified object by value
	}
	void print() {
		cout << "(" << real << ", " << imag << ")" << endl;
	}
};
// Note that addtion is defined as non-member function
Complex operator + (const Complex& lhs, const Complex& rhs) {
	Complex temp{lhs};                       // Make a copy of the lhs argument
	temp += rhs;                              // Add the rhs argument to it
	return temp;                              // Return the modified copy of the lhs argument
}
int main() {
	Complex c1(1, 2);
	Complex c2(3, 4);
	cout << "c1: ";
	c1.print();
	cout << "c2: ";
	c2.print();
	Complex c3 = c1 + c2;
	cout << "c3: ";
	c3.print();
	c1 += c2;
	cout << "c1: ";
	c1.print();
	Complex c4 = 1 + c2;                      // Type conversion (int -> double -> Complex)
	cout << "c4: ";
	c4.print();
}
```

### 60. Equality and Inequality Operators
- Equality operator summary
  - Prototype: `bool operator==(const T&lhs, const T& rhs);`
  - How to invoke: `a==b;`
  - Called as: `operator==(a,b);`
  - Return value
    - true when the two objects are equal
    - Otherwise false
- Inequality operator summary
  - Prototype: `bool operator!=(const T&lhs, const T& rhs);`
  - How to invoke: `a!=b;`
  - Called as: `operator!=(a,b);`
  - Return value
    - false when the two objects are equal
    - Otherwise true
```cpp    
#include<iostream>
#include<string>
class student {
private:
  std::string name;
  int id;
public:
  student(std::string aname, int aid): name(aname), id(aid) {}
  friend bool operator==(const student& lhs, const student& rhs);
};

bool operator==(const student& lhs, const student& rhs) {
  if ((lhs.name == rhs.name) && (lhs.id == lhs.id)) return true;
  else return false;
}
int main() {
  student st1("John", 123);
  student st2("John", 123);
  std::cout << (st1 == st2) << std::endl;
  return 0;
}
```

### 61. Less-than Operator
- Less-than operator summary
  - Prototype: `bool operator<(const T&lhs, const T& rhs);`
  - How to invoke: `a<b;`
  - Called as: `operator<(a,b);`
  - Return value
    - true when the left object is less than the object on the right
    - Otherwise fals
- STL uses the less-than operator for sorting and ordering
  - When we sort a container, if the elements don't have implementation of `< operator` then code will not be compiled!!!
  - Same issue when inserting elements into a sorted container
- < operator can handle other comparison operations
  - `a==b` => `!(a<b) && !(b<a)`
  - `a >=b` => `!(a<b)`
  - `a>b` => `!(a<b)&&!(a==b)`
- Sorting a vector
  - C++ defines a sort() in <algorithm>
  - This will sort the elements in ascending order
  - < operator of elements will be used to determine the order
  - std::string  will be sorted in alphabetical order
- Sample implementation fo student class above
```cpp
bool operator <(const student& lhs, const student& rhs) {
	return (lhs.id < rhs.id);                          // Order by ID (numerical sort)
}
```

### 62. Prefix and Postfix Operators
- ++p or p++
- Overloading prefix ++operator
```cpp
Test& Test::operator++(){
  ++member;
  return *this;
}
```
- Overloading postfix ++operator
  - The postfix operator make a copy of the object, performs the increment and returns the unmodified object
  - The postfix operator takes a dummy argument, to give it a different signature from the prefix operator
```cpp
Test& Test::operator++(int t){ // t is a dummy
  Test temp(*this); // backup of the current object
  ++member;
  return *temp;  // return the backup
}
```

### 63. Function Call Operator
- Procedural vs functional programming
- Callable objects are used to implement functional programming in C++
- Function pointer
  - C allows us to create callable objects
    - `void func(int, int);`
    - `void (*func_ptr)(int,int)=func;`
    - `func_ptr(1,2);`
- Functors
  - C++ classes can define a function call operator
  - An object of the class is a data variable
  - The object can be called like a function
  - A c++ class which implements a function call operator is called a "functor"
- Functor call operator summary
  - Prototype: `some_type operator()(...);` Arguments in the second parentheses
  - How to invoke it: `test(...);`
  - Called as: `test_operator()(...);
```cpp
#include<iostream>
class myClass {
public:
 int operator()(int a, int b) {
   std::cout << "arguments " << a << " and " << b << std::endl;
   return a+b;
 }
};
int main() {
  myClass abc;
  std::cout << abc(1,2) << std::endl;
  return 0;
}
```
- Functors with state
  - Member data might be stored, and this may be used as a state (?)
  - Regular functions do not have such state feature

### 64. Printing Out Class Member Data
- How output works for builtin types
  - Operator << overloads for all the builtin and library types
  - Binary operator
  - Invoked as: `operator<<(std::out,i);`
- Nested calls of operator <<
  -  `std::cout << i << j;` => `operator << (operator << (std::cout,i),j);`
  - Prototype must be: `ostream& operator<<(ostream&,int);`
- Sample code:
```cpp
#include<iostream>
class myClass {
  int a,b; 
public:
 myClass(int x, int y): a(x), b(y) {}
 friend std::ostream& operator<<(std::ostream& os, const myClass& arg);
};
std::ostream& operator<<(std::ostream& os, const myClass& arg) {
  os << "a = " << arg.a << " b = " << arg.b << std::endl;
  return os; // std::ostream is returned
}
int main() {
  myClass abc(123, 456);
  std::cout << abc;
  return 0;
}
```

## Section 6: Algorithms Introduction and Lambda Expressions

### 65. Algorithms Overview
- Many useful methods in `<algorithm>`
- Code re-use
- Shorter/clearer code
- Searching algorithm
  - std::string::find() returns an index
  - std::find() returns an iterator
    - Some containers may not have index

### 66. Algorithms with Predicates
- predicate: provides state at comparison
  - By conditions, yields true or false
- Ex)
  - std::sort(): uses element's < operator
  - Using predicate, sort() may use different callable object(predicate) for comparison conditions
  - Predicate may be implemented as a regular function or a functor or a lambda expression

### 67. Algorithms with _if Versions
- std::find(): finds an element matching with value
- std::find_if(): finds an element matching with predicate
  - A predicate is provided as a callable object - a function or a functor or a lambda expression
```cpp
#include<iostream>
#include<algorithm>
struct is_odd { bool operator() (const int n) const { return (n%2 == 1) ; }};
int main() {
 int vec[] = {2,3,4,5};
 auto odd_it = std::find_if(std::cbegin(vec), std::cend(vec),is_odd());
 if (odd_it != std::cend(vec)) std::cout << "First odd number is "<< *odd_it << 
std::endl;
 return 0;
}
```
### 68. Lambda Expressions Introduction
- Anonymous local functions
- Similar to closures in other languages
- When the compiler encounters a lambda expression, it will synthesize a code that defines a functor
  - A unique name by the compiler
- Lambda expression syntax
  - Anonymous and defined inline
  - Begins with []: `[] (int n) {return (n%2 == 1);}`
- Let's replace the above written functor predicate with a lambda expression
```cpp
#include<iostream>
#include<algorithm>
int main() {
 int vec[] = {2,3,4,5};
 auto odd_it = std::find_if(std::cbegin(vec), std::cend(vec),[](int n) { return (n%2 == 1); }); // Using a lambda
 if (odd_it != std::cend(vec)) std::cout << "First odd number is "<< *odd_it << 
std::endl;
 return 0;
}
```
- Return type can be explicity addressed such as `[] (int n) -> bool {...}` but this is not required by C++17

### Assignment 8: Algorithm with Lambda Expression
```cpp
#include<iostream>
#include<algorithm>
#include<vector>
void print_v(std::vector<int> & x) {
  for (auto& el : x) std::cout << el << " ";
  std::cout << std::endl;
}
int main() {
  std::vector<int> q{3,1,9,6,2};
  print_v(q);
  std::sort(q.begin(), q.end(), [](int a, int b){ return a>b;});
  print_v(q);
  return 0;
}
```

### 69. Lambda Expressions Practical
- equal() algorithm
  - Compares elements of two vectors
  - 5th argument is a predicate (replaces == for element-wise)
- Case-insensitive comparison
```cpp
bool equal_string(const string& lhs, const string& rhs) {
  return equal(std::cbegin(lhs),std::cend(lhs),
               std::cbegin(rhs),std::cend(rhs), 
               [](char lc, char rc) 
                 { return std::touppoer(lc) == std::toupper(rc);}
              );
}
```

### 70. Lambda Expressions and Capture
- A lambda expression can access local variables which are references and where initialized with a constant expression
- Can read but not modify local variables which are integers/enums and were initialized with a constant expression
- Compilers may not implement this properly
```cppp
#include<iostream>
int global{99};
int main() {
 static int local1 {123};
 const int one{1};
 const int& r_one{one};
 []() { std::cout << global << " " << local1 << std::endl;
        std::cout << one << " " <<  std::endl;
        //std::cout << r_one << std::endl; // compile error
  };
 return 0;
}
```
- When a lambda expression needs full access to local variables, **capture** them inside of []: `[x,y](int arg) {return x*arg+y;}`
- A lambda with capture is synthesized as a **functor with state**
- Captured variables are passed into the functor's constructor by value (capture by value)
  - By default, member data are const
  - Using `mutable` keyword, member data can be changed

### 71. Lambda Expressions and Capture Continued
- In order to change the capture variables, we need to capture them by reference
- & before the captured varaible: `[&x,&y](int arg) {x++; y++; return x*arg+y;}`
```cpp
#include<iostream>
int main() {
  int a{1},b{5};
  auto myL = [&a,&b](int arg) { a++; b++; return a*arg+b; };
  std::cout << a << " " << b  << std::endl;
  std::cout << myL(2) << std::endl;
  std::cout << a << " " << b  << std::endl;
  return 0;
}
```
- Captured variables by reference are passed into the synthesized functor by reference
- Implicit capture
  - [=]: captures all variables by value in scope
  - [&]: captures all variables by reference in scope
  - [=,&x]: x by reference, all others by value
  - [&,=a,=b]: a and b by value while all others by reference
- How to capture **this** in a class
  - [this] or [&] or [=]
  - [=this] or [&this] are not allowed
  - [this] captures the objct by reference
    - Can access member data and member functions
- In C++17, [*this] is available
  - Capture by value
  - Lambda expression becomes immutable

### Assignment 9: Mutable Lambda
```cpp
#include<iostream>
int main() {
  int x{42}, y{99},z{0};
  auto lam = [=,&z]() mutable {++x; ++y; z = x +y;};
  lam();
  lam();
  std::cout << x << " " << y << " "<< z << std::endl;
  return 0;
}
```
- Yields 42 99 145
- As lam is synthesized as a functor, x/y are state variables. Briefly, internal change are managed, without affecting local variables of main()
- x/y are not changed as they are captured by value
- However, x/y in the synthesized functor are changed by ++x; ++y;
- `mutable` keyword stores member data in the synthesized functor as non-const, allowing `++x;++y`. 

### 72. Lambda Expressions and Partial Evaluation
- Storing lambdas
  - Lambda expressions are **first class objects**
  - Can store them in variables and pass them to a function: `auto lam = [max](const string& str) {return str.size() > max;};`
    - Must use auto as we do not know what type the compiler generated functor will be
  - Also returning a lambda expression from a function is allowed
```cpp
#include <iostream>
#include <string>
using namespace std;
// Function which returns a lambda function
auto greeter(const string& salutation) {
	return [salutation](const string& name) { return salutation + ", "s + name; };       // The lambda function
}
int main() {
	// Store the lambda function in a variable
	auto greet = greeter("Hello"s);
	// Call the lambda function
	cout << "Greeting: " << greet("James") << endl;
	cout << "Greeting: " << greet("students") << endl;
}
```
- Partial evaluation
  - data are processed in stages
  - Reduces computation
- Partial evaluation using Lambda
  - ?

### 73. Lambda Expressions in C++14
- Now we can use `auto` in the argument type in the lambda expression: `auto lam = [](auto x, auto y) { return x+y;};`
- Generic lambda implementation
  - When a functor is synthesized, template will be used for argument type
- Capture by move is implemented

### Assignment 10.:Generalized capture with initialization
```cpp
#include<iostream>
int main() {
  int y = 1;
  auto lam = [y=y+1](int x) {return x+y;};
  std::cout << lam(5) << std::endl;     // 7
  std::cout << "y="  << y << std::endl; // 1
  return 0;
}
```

### 74. Pair Type
- std::pair from `<utility>`
  - Two public data members
  - Accessed as `first` and `second`
- `std::pair<int,float> numpair{123, 3.14f};`
- Using make_pair(): `auto wordpair {make_pair("hello"s, "there"s)};`
- In C++17, `std::pair wordpair{"hello"s, "there"s};`

### 75. Insert Iterators
- An output stream iterator inserts data into an output stream
  - std::back_insert_iterator adds an element at the back
    - push_back() will be called
  - std::front_insert_iterator adds an element at the front
    - push_front() is called
    - Not working with std::vector or std::string
  - std::insert_iterator for any given position
- Insert functions
  - back_inserter()
  - front_inserter()
  - inserter()
```cpp
#include <iostream>
#include <vector>
using namespace std;
int main() {
	vector<int> vec = {1, 2, 3};                  // Create a vector
	// Print out vector elements
	cout << "Vector: ";
	for (auto v: vec)
		cout << v << ", ";
	cout << endl;	
	auto el2 = next(begin(vec));                  // Get an iterator to the second element	
	auto it = inserter(vec, el2);                 // Get an insert iterator for vec
	// Assign to this iterator
	*it = 99;                                     // Calls vec.insert(el2, 99)	
	// vec  now contains {1, 99, 2, 3}
	// Print out vector elements
	cout << "Vector after insert: ";
	for (auto v: vec)
		cout << v << ", ";
	cout << endl;	
	cout << "Data at el2: ";
	cout << *el2 << endl;
}
```

### 76. Library Function Objects
- C++ library provides some function objects
- Generic operators for arithmetic, logical and relational operations
- Implemented as templated functors
- std::plus
- std::minus
- std::multiplies
- std::divides
- std::modulus
- std::negate
- std::equal_to
- std::not_equal_to
- std::greater
- std::less
- std::greater_equal
- std::less_equal
- std::logical_and
- std::logical_or
- std::logical_not
- std::bit_and
- std::bit_or
- std::bit_exor
- std::bit_not

## Section 7: Algorithms Continued

### 77. Searching Algorithms
- std::string has a member function find_first_of()
- std::find_first_of() returns an iterator corresponding to the found element
```cpp
#include <string>
#include <iostream>
using namespace std;
int main() {
	string str {"Hello world"};
	cout << "String to search: " << str << endl;
	string vowels {"aeiou"};
	cout << "First vowel is at index " << str.find_first_of(vowels) << endl;
	cout << "Last vowel is at index " << str.find_last_of(vowels) << endl;
	cout << "First non-vowel is at index " << str.find_first_not_of(vowels) << endl;
	cout << "Last non-vowel is at index " << str.find_last_not_of(vowels) << endl;
}
```
- std::adjacent_find() looks for two neighboring elements that have the same value
  - `auto pos = std::adjacent_find(std::cbegin(str1), std::cend(str1));`
- std::search_n() looks for a sequence of n successive elements which have the same given value
  - `auto pos = std::search_n(std::cbegin(vec), std::cend(vec), 2,3);`
- std::search() takes two iterator ranges
  - `auto pos = std::search(std::cbegin(str1), std::cend(str1), std::cbegin(str2),std::cend(str2));`

### 78. Searching Algorithms Continued
- mismatch() takes two iterator ranges and looks for differences b/w the two ranges
  - Returns std::pair of each element found 
- all_of()/any_of()/none_of() take an iterator range and a predicate
- binary_search() is similar to find() but assumes that the iterator range is already sorted
- includes() checks if the second range is included in the first range

### 79. Numeric Algorithms
- From `<numeric>`
- std::iota(): populates or fills the elements of a vector 
- std::accumulate() returns the sum of all elements
  - Instead of addition, a callable object can be passed to perform a different operation
  - Cannot be parallelized
- std::reduce()
  - By C++17
  - Parallel version of std::accumulate()

### 80. Write-only Algorithms
- std::fill() assigns a given value to elements in an iterator range: `std::fill(begin(vec),end(vec), 123);`
- std::fill_n(): starting from the first iterator, length of variables to be filled and filling value are provided
- std::generate() uses the value from a callable object to assign elements
  - Random initialization of a vector or an array?

### 81. for_each Algorithm
- Calls a function on every element in an iterator range
- `for_each(std::cbegin(str1),std::cend(str1),[](const char c) {std::cout << c << ","; });`
- When arguments in the lamba expression are referenced, value can be changed
- Can be written as range-for loop
  - Loop is recommended
- Might be useful for sub-range of elements

### 82. Copying Algorithms
- Copying elements into another range
- `std::copy(std::cbegin(vec1),std::cend(vec1),std::begin(vec2));`
- copy_n()
- copy_if(): copies when a predicate is true

### 83. Write Algorithms
- std::replace() replaces a given value with another
- std::replace_if() changes elements when a predicate returns true
- Base algorithms overwrite
- _copy() version writes to another iterator range

### 84. Removing Algorithms
- remove(): removed elements move to the back of the iterator range. The values of those removed elements are not defined. Still size() returns the same number
- erase(): physically removes these elements from a iterator range. Size() changes

### 85. Removing Algorithms Continued
- std::remove_if() to use predicate
- _copy() version
- std::unique(): removes duplicate adjacent elements
  - Elements in the iterator must be sorted
  - std::unique() uses == operator. Can couple with predicate

### 86. Transform Algorithm
- transform() calls a given function on every element in the range
  - results are pushed into another iterator range

### 87. Merging Algorithms
- merge() combines two sorted iterator ranges into a destination
- set_intersection() combines two sorted iterator ranges into a destination
- set_union() combines two sorted iterator ranges into a destination

### 88. Reordering Algorithms
- Re-arranges the elements in iterator range
  - Not sorting
- reverse()
- rotate()
- rotate_copy()

### 89. Partitioning Algorithms
- Splits a container into two groups
  - They are still in a single container
- paritition()
  - Two groups by a predicate
  - Might not be ordered
- stable_partition()
  - Same as partition but the order is maintained
- is_partitioned()

### 90. Sorting Algorithms
- By default, element's < operator is used
- std::sort() orders the elements in ascending order
  - Predicate can change the order
- is_sorted()
- is_sorted_until()

### 91. Sorting Algorithms Continued
- partial_sort() 
- partial_sort_copy()
- nth_element(): after partial sort, find the corresponding element

### 92. Permutation Algorithms
- Every possible arrangement of elements
- next_permutation() takes an iterator range
```cpp
#include<iostream>
#include<algorithm>
#include<string>
int main() {
  std::string msg{"abc"};
  do {
   std::cout << msg << std::endl;
  } while(std::next_permutation(msg.begin(), msg.end()));
  return 0;
}
```
- Result
```bash
$ ./a.out 
abc
acb
bac
bca
cab
cba
```
- prev_permutation() reorders the elements to give the previous permutation
- is_permutation() takes two ranges of iterators

### 93. Min and Max Algorithms
- By default, the element's < operator is used
- Can be coupled with a predicate
- max_element()/min_element() take a range of iterator

### 94. Further Numeric Algorithms
- From `<numeric>`
  - partial_sum() yields {a, a+b, a+b+c, ...} from {a,b,c, ...}
    - Uses element's + operator
    - `std::partial_sum(v1.cbegin(),v2.cbegin(), v1.begin())
  - adjacent_difference() yields {a, b-a, b-c, ...} from {a,b,c, ...}
  - inner_product() multiplies the elements of two containers and sum them
    - Scalar product
```cpp
#include<iostream>
#include<vector>
#include<numeric>
int main() {
  std::vector<int> x = {1,2,3,4};
  std::vector<int> y;
  std::partial_sum(x.cbegin(), x.cend(), std::back_inserter(y));
  for (auto& el: y) std::cout << el << std::endl;
  return 0;
}
```

### 95. Further Numeric Algorithms Continued
- inner_product() is equivalent to transform() followed by accumulate()
- Using lambda or predicate, multiples/sum operations can be overridden with different operations
- As std::accumuate() is not parallel processed, we may replace with std::transfrom_reduce() (since C++17) for better performance

### 96. Introduction to Random Numbers
- Pseudo-random number
  - "Seed" will initialize
    - Same seed for the same sequence
- HW derived random number
  - True random number generators
  - Thermal noise, radioactive decay
  - Specialized devices

### 97. Random Numbers in Older C++
- rand() is inherited from C and declared in `<cstdlib>`
  - Returns a number b/w 0 and RAND_MAX
- srand() to seed the generator
  - Use time(0) to return the current time
  - `srand(time(0));`
- Disadvantages of rand()
  - Not very random
  - Rescaling may yield bias
  - Poor cryptographic security
    - May be able to guess the next number
```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;
int main() {
	srand(time(0));  // Use current time as seed
	// Print out a pseudo-random floating-point number with value between 0 and 1
	cout << 1.0*rand()/RAND_MAX << endl;            // Convert the result to double!
	// Print out ten pseudo-random integers with value between 1 and 100
	for (int i = 0; i < 10; ++i)
		cout << (99*rand()/RAND_MAX + 1) << endl;
}
```

### 98. Random Numbers in Modern C++
- From `<random>`
- A random number engine is implemented as a functor
- A distribution is also implemented as a functor
  - The constructor takes the range as arguments
- Two main random number engines are:
  - default_random_engine
    - May be a wrapper around rand()
  - mt19937
    - Veru fast generating number
    - Almost crypto secure
    - Has a lot of state
    - Usually the best choice for most requirements
- Distribution types
  - `uniform_int_distgribution<T>(m,n)` produces  uniformly distributed integer of type T with the value b/w m and n inclusive
  - `uniform_real_distribution<T>(m,n)`
- random_device() produces a HW generated random number from system entropy data
  - However, it will produce a pseudo random number if
    - The system does not provide entropy data
    - The compiler is GNU C++
- Recommendations
  - Use mt19937
  - Check documentation before using random_device
  - Make engine and distribution objects static
    - Quite expensive to create
    - Every time an engine is initialized, the sequence starts again
```cpp
#include <iostream>
#include <random>
using namespace std;
int main() {
	random_device rd;                                 // Random device (or maybe not!)   
	mt19937 mt(rd());                                 // Seed engine with number from random device
	uniform_int_distribution<int> idist(0, 100);      // We want ints in the range 0 to 100
	cout << "Five random integers between 0 and 10:\n";
	for (int i = 0; i < 5; ++i ) {
		cout << idist(mt) << ", ";                    // Call the functor to get the next number
	}
	cout << endl << endl;
	uniform_real_distribution<double> fdist(0, 1);    // Doubles in the range 0 to 1
	cout << "Five random floating-point numbers between 0 and 10:\n";
	for (int i = 0; i < 5; ++i ) {
		cout << fdist(mt) << ", ";
	}	
	cout << endl;
}
```

### 99. Random Number Algorithms
- `std::shuffle()` rearranges an iterator range in a random order
```cpp
std::vector<int> vec {3,1,4,2,9};
static mt19937 mt;
std::shuffle(vec.begin(), vec.end(), mt);
```
- `std::bernoulli_distribution()`

### 100. Palindrome Checker Practical
- A palindrome is an expression which reads the same backwards
  - Madam I'm Adam
  - Space characters, punctuation, and captialization are ignored
- Write a program to check if the given text is a palinedrome or not
```cpp
#include <iostream>
#include <algorithm>
#include <string>
using namespace std;
// Return a copy of the argument string
// with non-alphabetical characters removed, converted to lower case
string normalize(const string& s) {
	string retval{""};
	copy_if(cbegin(s), cend(s), back_inserter(retval),
				[](char c) { return isalpha(c); }
	);
	transform(begin(retval), end(retval), begin(retval), 
				[] (char c) { return tolower(c); }
	);
	return retval;
}
int main() {
	string s{""};
	cout << "Please enter your palindrome: ";
	getline(cin, s);
	// Input string with punctuation and spaces removed
	string pal{normalize(s)};
	// Call mismatch to compare the string to its reverse
	// Use a reverse iterator
	auto p = mismatch(cbegin(pal), cend(pal), crbegin(pal));
	// The return value from mismatch() is a pair of iterators
	// These point to the first mismatched element in each range
	if (p.first == cend(pal) && p.second == crend(pal)) {
		// No mismatch found - the string is the same in both directions
		cout << "\"" << s << "\" is a palindrome\n";
	}
	else {
		// There is a mismatch
		// The character at p.first does not match the character at p.second
		cout << "\"" << s << "\"" << " is not a palindrome\n";
		// Make a copy of the string, up to the mismatch in the reversed string
		string outstr;
		copy(cbegin(pal), p.second.base(), back_inserter(outstr));
		cout << "'" << *(p.first) << "'" << " does not match " << "\'" << *(p.second) << "\'";
		cout << " at \"" << outstr << "\"" << endl;
	}
}
```

### 101. Random Walk Practical
- Random walk occurs when a moving physical object changes direction at random
  - Brownian motion in physics
- 1D random walk code
```cpp
#include <iostream>
#include <random>
#include <thread>
#include <string>
using namespace std;
int main() {
	int x{0};
	int vx{0};
	const int width{40};
	string blank_line(width, ' ');
	mt19937 mt;
	bernoulli_distribution bd;
	while (true) {
		if (bd(mt)) {
			vx = 1;
			if (x == width)
				vx = -1;
		}
		else {
			vx = -1;
			if (x == 0)
				vx = 1;
		}
		x += vx;
		cout << "\r" << blank_line;
		string position(x, ' ');
		cout << "\r" << position << '.' << flush;
		this_thread::sleep_for(100ms);
	}
}
```

### Assignment 11: Algorithms and Iterators Workshop
- Ref: https://en.cppreference.com/w/cpp/algorithm
1) Fill a vector with 10 random integers between 0 and 1000
2) (For each exercise, display the result)
3) Find the maximum element in this vector
4) Find the index of this maximum element
5) Sum the elements of the vector
6) Count the number of odd numbers in the vector
7) Normalize the vector (divide all the elements by the largest.) Put
the normalized elements into a vector of doubles, without setting
the size of the output vector first
8) Make a sorted copy of the vector. Without using a functor or a
lambda (or equivalent), find the first element greater than 455 and
the number of elements > 455
9) Copy all the odd numbers to a vector of doubles, without setting
the size of the output vector first
10) Sort the vector in descending order
11) Randomly shuffle all but the first and the last element of the vector
12) Remove all the odd numbers from the vector
13) Write the remaining elements to a text file on a single line as a
comma separated list, without a trailing comma
14) Read the file "words.txt". Display each distinct word once. Ignore
punctuation and capitalization
Hint: look into std::istreambuf_iterator
15) Count the total number of words in the file
16) Count the number of lines in the file
17) Count the number of characters in the file
18) Read "words.txt" and "words2.txt". Display the words which are
common to both files
19) Calculate the factorial of 6 (6 x 5 x 4 x ... x 1)
```cpp
#include<iostream>
#include<fstream>
#include<algorithm>
#include<random>
#include<vector>
int main() {
  std::vector<int> vec(10);
  std::random_device rd;
  std::mt19937 mt(rd());
  // 1) 10 random element vector
  std::uniform_int_distribution<int> idist(0,1000);
  for (auto&el : vec) el = idist(mt); 
  for (auto&el :vec) std::cout << " " << el ;
  // 2)
  std::cout << std::endl; 
  auto it = std::max_element(vec.cbegin(),vec.cend());
  // 3+4) Find max and its index
  int maxv = 0;
  if (it != vec.cend()) {
    std::cout << "max value = " << *it << std::endl; 
    std::cout << "max index = " << std::distance(vec.cbegin(),it) <<std::e
ndl;
    maxv = *it;
  }
  // 5) sum of elements
  std::cout <<"sum of vector = "<< std::accumulate(vec.cbegin(), vec.cend(
), 0) << std::endl;
  // 6) Count the odd numbers
  auto odd_counts = std::accumulate(vec.cbegin(), vec.cend(),0, [](int acc
, int v){ if (v%2==0) { return acc; }  else {return acc+1 ;} });
  std::cout << "odd num count = " << odd_counts << std::endl; 
  // 7) normalize vector
  std::vector<double> xvec;
  std::transform(vec.cbegin(), vec.cend(), std::back_inserter(xvec), [maxv
](int n) { return 1.0*n/maxv; });
  for (auto&el :xvec) std::cout << " " << el ; std::cout << std::endl;
  // 8) sorted copy of the vector
  std::vector<int> sorted_vec(vec);
  std::sort(sorted_vec.begin(), sorted_vec.end());
  std::cout << "Now sorted: ";
  for (auto&el :sorted_vec) std::cout << " " << el ; std::cout << std::end
l;
  int first_el = -1, partial_sum = 0;
  for (auto &el : sorted_vec) {
     if (el > 455) {
        if (first_el < 0)  first_el = el;
        partial_sum += el;
     }
  } 
  std::cout << "partial sum = "<<  partial_sum << std::endl;
  // 9) extract odd numbers from a vector to a new vector
  std::vector<double> odds_only;
  std::copy_if(vec.cbegin(), vec.cend(), std::back_inserter(odds_only), []
(int n) { return (n%2 !=0) ; });
  std::cout << "odds only  = " ;
  for (auto&el :odds_only) std::cout << " " << el ; std::cout << std::endl
;
  //10) sort in descending order
  std::sort(vec.begin(), vec.end(), [](int a, int b) { return a>b; });
  std::cout << "descending order " ;
  for (auto&el :vec) std::cout << " " << el ; std::cout << std::endl;
  // 11) shuffle all but the first/last element
  std::shuffle(vec.begin()+1, vec.end()-1, mt);
  std::cout << "Shuffled : ";
  for (auto&el :vec) std::cout << " " << el ; std::cout << std::endl;
  // 12) delete all odd numbers from the vector
  auto new_end = std::remove_if(vec.begin(), vec.end(), [](int n) { return
 (n%2 !=0) ; });
  vec.erase(new_end, vec.end());
  std::cout << "odds numbers are removed:  ";
  for (auto&el :vec) std::cout << " " << el ; std::cout << std::endl;
  // 13) writing the remaining elements into a textg file
  std::ofstream of("vec.txt", std::fstream::out); 
  for (auto&el :vec) of << el << ",";
  of.close();
  return 0;
}
```
```cpp
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<cctype>
#include<vector>
#include<algorithm>
std::vector<std::string> summary_file(std::string fname) {
  std::cout << "Reading "<< fname << std::endl;
  std::ifstream ifile(fname);
  std::istreambuf_iterator<char> eof;
  std::istreambuf_iterator<char> it(ifile);
  std::vector<std::string> all_words;
  std::string word;
  int num_lines = 0;
  while(it != eof) {
    if (*it == '\n') num_lines ++;
    if (std::ispunct(*it) || std::isspace(*it)) {
      if (word.size() >0) all_words.push_back(word);
      word = ""; // initialization
     }
    else { word +=std::toupper(*it); }
    ++it;
  } 
  int num_words = all_words.size(); // 15)
  int num_chars = 0;
  for(auto &el: all_words) num_chars += el.size(); // 17)
  std::cout << "Num lines = " << num_lines << std::endl;
  std::cout << "Num of words = " << num_words << std::endl;
  std::cout << "Num of chars = " << num_chars << std::endl;
  // 
  std::vector<std::string> uniq_words(all_words);
  std::sort(uniq_words.begin(), uniq_words.end());
  auto new_end = std::unique(uniq_words.begin(), uniq_words.end());
  uniq_words.erase(new_end, uniq_words.end());
  return uniq_words;
}
int main() {
  auto f1_uniq_words = summary_file("words.txt");
  auto f2_uniq_words = summary_file("words2.txt");
  std::vector<std::string> overlap_words;
  std::set_intersection(f1_uniq_words.cbegin(), f1_uniq_words.cend(),
                        f2_uniq_words.cbegin(), f2_uniq_words.cend(),
                        std::back_inserter(overlap_words));
  std::cout << "Overlapping words are: "; // 18
  for (auto &el : overlap_words) std::cout << el << std::endl;
/*
  std::stringstream buffer;
  buffer << ifile.rdbuf();
  ifile.close();
  std::string word;
  while (buffer >> word) std::cout << word << std::endl;
*/
  return 0;
}
```
```cpp
// 19)
#include<iostream>
int main()  {
  int vec[] = {1,2,3,4,5,6};
  int factorial = 1;
  for (auto &el : vec)  factorial *= el;
  std::cout << "6! = " << factorial << std::endl;
  return 0;
}
```

## Section 8: Containers

### 102. Container Introduction
- Sequential containers
  - std::string and std::vector
  - Can access data by its position
- Associative containers
  - Key vs value
  - Sets and maps
  - do not support push_back() and pus_front()
  - Use insert() and erase()
- Container adaptors
  - Queues and stacks

### 103. Standard Library Array
- Built-in array
  - Static array inherited from C
  - Faster than std::vector and compatible with C code
  - Some disadvantages
- Mondern C++ provides std::array
- std::array
  - Defined in `<array>`
  - A templated type
  - Similar interface to STL container while retaining the speed of built-in arrays
    - Supports iterators
    - Contains the size info
- std::array object creation
  - Created on stack
  - The number of elements must be known at compile time
  - A contiguous block of memory is allocated to store the elements
  - `std::array<int, 5> arr{1,2,3,4,5};`
- std::array interface
  - sizse()
  - empty()
  - operator[]
  - at()  // access element by index with bounds checking
  - front()
  - back() 
  - data() // returns a built-in array container
```cpp
#include <iostream>
#include <array>
using namespace std;
int main() {
	// std::array can be initialized the same way as a vector
	std::array<int, 5> arr {1, 2, 3, 4, 5};
	cout << "Iterator loop: ";
	for (auto it = begin(arr); it != end(arr); ++it)         // Explicit iterator loop
		cout << *it << ", ";
	cout << endl;
	cout << "Range-for loop: ";
	for (auto el: arr)                                       // Range-for loop
		cout << el << ", ";
	cout << endl;
	cout << "Indexed loop: ";
	for (size_t i = 0; i < arr.size(); ++i)                  // Indexed loop
		cout << arr[i] << ", ";
	cout << endl;
	// Arrays of the same type and size can be assigned
	std::array<int, 5> five_ints;
	five_ints = arr;
	cout << "Elements of five_ints: ";
	for (auto el: five_ints)                                       // Range-for loop
		cout << el << ", ";
	cout << endl;
}
```

### 104. Forward List
- In a list, each element has its own memory allocation ("node")
  - No contiguous memory block over nodes
- Each node contains the element value and a pointer (linked list)
- std::forward_list only supports forward iterators
  - Use insert_after() or erase_after()
```cpp
#include <iostream>
#include <forward_list>
using namespace std;
int main() {
	forward_list<int> l{4, 3, 1};                 // Create a list object
	cout << "Initial elements in list" << endl;
	for (auto el: l)                              // Use a range-for loop to display all the elements
	    cout << el << ", ";
	cout << endl;
	auto second = l.begin();
	advance(second, 1);                           // i is an iterator to the second element
	l.insert_after(second, 2);                    // Insert a new element after the second element
	cout << "Elements in list after inserting 2" << endl;
	for (auto el: l)
	   cout << el << ", ";
	cout << endl;
	l.erase_after(second);                        // Remove this element
	cout << "Elements in list after erasing 2" << endl;
	for (auto node: l)
	   cout << node << ", ";
	cout << endl;
}
```

### 105. List
- std::list
  - Double-linked list
- Pros and cons
  - Adding or removing elements from the middle of a list is faster than vector
  - No indexing
  - Accessing an element is slower than vector
  - More memory than vector
```cpp
#include <iostream>
#include <list>
using namespace std;
int main() {
	list<int> l{4, 3, 1};                       // Create a list object
	cout << "Initial elements in list" << endl;
	for (auto el: l)                            // Use a range-for loop to display all the elements
	    cout << el << ", ";
	cout << endl;
	auto last = end(l);
	advance(last, -1);                          // i is an iterator to the second element
	auto two = l.insert(last, 2);               // Insert a new element before the last element
	cout << "Elements in list after inserting 2" << endl;
	for (auto el: l)
	   cout << el << ", ";
	cout << endl;
	l.erase(two);                               // Remove this element
	cout << "Elements in list after erasing 2" << endl;
	for (auto node: l)
	   cout << node << ", ";
	cout << endl;
}
```

### 106. List Operations
- List operations
  - We can use push_back() and push_front()
  - No random access using an index
- Generic sort() doesn't work as it uses random access
  - Use the member function: Ex) `l.sort();`
- Instead of generic functions, use member functions when available
  - Faster than generic 
  - `l.erase();`
```cpp
#include <iostream>
#include <list>
#include <algorithm>                       // For std::sort()
using namespace std;
int main() {
	list<int> l{4, 3, 1};                  // Create a list object
	cout << "Initial elements in list" << endl;
	for (auto el: l)                       // Use a range-for loop to display all the elements
		cout << el << ", ";
	cout << endl;
	// sort(begin(l), end(l));             // Does not compile
	l.sort();                              // Sort the list
	cout << "Elements in list after sorting" << endl;
	for (auto el: l)
		cout << el << ", ";
	cout << endl;
	l.remove(3);                          // Remove element with value 3
	cout << "Elements in list after removing 3" << endl;
	for (auto el: l)
		cout << el << ", ";
	cout << endl;
}
```  
- Operations which move elements
  - reverse()
  - remove()
  - remove_if()
  - unique()
- merge()
  - Before merge, sort each list first
- splice()
  - Behavior of splice() at std::list and std::forward_list is different
  
### 107. Deque
- std::deque
  - Double-ended queue
  - Defined in <deque>
  - Similar to vector but elements can also be added at front
    - Supports push_front()
```cpp
#include <iostream>
#include <deque>
using namespace std;
int main() {
	deque<int> dq;              // Create an empty container
	dq.push_back(5);
	dq.push_back(1);
	dq.push_front(3);           // Add element with value 3 before the other elements
	dq.push_front(2);
	dq.push_front(4);
	for (auto it: dq) {
		cout << it << ", ";
	}
	cout << endl;
}
```
- Deque is slightly slower than vector for most operations
- List is much slower than vector and deque for most operations, and uses more memory
- Vector must be the default choice

### Assignment 12: Sequential Containers
```cpp
#include <vector>
#include <list>
#include <deque>
#include <string>
#include <iostream>
int main() {
    std::vector<std::string> words;
    //std::list<std::string> words;
    //std::deque<std::string> words;
    for (std::string word; std::cin >> word;) {
        words.push_back(word);
        //words.push_front(word);    // Stores the strings in reverse order
    }
	for (std::vector<std::string>::iterator it = words.begin(); it != words.end(); ++it)
    //for (std::list<std::string>::iterator it = words.begin(); it != words.end(); ++it)
    //for (std::deque<std::string>::iterator it = words.begin(); it != words.end(); ++it)
    	std::cout << *it << "\n";
}
```

### Assignment 13: Sequential Containers Part Two
```cpp
#include<list>
#include<string>
#include<iostream>
class URL {
    std::string protocol;
    std::string resource;
public:
    URL(const std::string& prot, const std::string& res) {      protocol =
 prot;	resource = res;    }
    void print() const {std::cout << protocol << "://" << resource << "\n"
;}
    std::string get_protocol() {return protocol;}
    std::string get_resource() {return resource;}
    bool isSame(URL& arg) { if (arg.get_protocol() == protocol && arg.get_
resource() == resource) return true;
    else return false; }
    
};
class URLdbase {
  std::list<URL> dbase;
public:
  URLdbase() = default;
  size_t addURL(URL& arg) {
     for(auto &el: dbase) {
      if (el.isSame(arg)) return dbase.size();
     }
    arg.print(); dbase.push_front(arg);
    return dbase.size();
  }
};
int main() {
  URLdbase myDB;
  URL a1("http", "www.cnn.com");
  URL a2("http", "www.aws.com");
  URL a3("http", "www.azure.com");
  std::cout << myDB.addURL(a1) << std::endl;
  std::cout << myDB.addURL(a2) << std::endl;
  std::cout << myDB.addURL(a3) << std::endl;
  std::cout << myDB.addURL(a2) << std::endl;
  return 0;
}
```

### 108. Tree Data Structure
- C++ associative containers are implemented using a "tree"
- A node has a "key" for the element and two pointers, "left" and "right"
- Adding, removing, finding elements are very fast
- Tree balancing
  - Inserting/erasing elements may cause the tree "unbalanced"
  - Becomes less efficient
  - May need to "rebalance" the tree
    - Red-black tree
    - AVL tree

### 109. Sets
- std::set in `<set>`
- Unstructured collection of elements
- Associative container
  - Unique keys only
  - Elements are ordered using < operator of the key
- Implemented as a tree
  - Usually red-black tree
- insert()
  - Returns std::pair
    - When succesful: (iterator to the added element, true)
    - When failed: (iterator to the existing element, false)
- s.find(k) returns an iterator to the element with key k
  - or end() when k is not found
- Elements of std::set are const
  - Cannot be re-ordered
  - Many algorithms will not work
- Pros and Cons
  - Very fast accessing an arbitrary element
  - Insertion/deletion are usually very fast
  - Useful for checking membership/non-membership
  - Useful when unique data are handled
```cpp
#include <iostream>
#include <set>
#include <algorithm>
using namespace std;
void print(const set<int>& s) {
	cout << "Elements of set: ";
	for (auto el: s)                    // Use a range-for loop to display all the elements
		cout << el << ",";
	cout << endl << endl;
}
int main() {
	set<int> s;                         // Create an empty std::set
	s.insert(6);                        // Add some elements to it
	s.insert(7);                        // The argument to insert() is the key of the element
	s.insert(4);
	s.insert(5);
	s.insert(3);
	print(s);
	cout << "Calling find(7)\n";
	auto it = find_if(cbegin(s), cend(s), [](int n) {return n == 7; });
	if (it != s.end())
		cout << "The set has an element with key " << *it << endl;
	else
		cout << "The set has no element with key 7\n";
	cout << "\nCalling count(7)\n";
	auto n = count_if(cbegin(s), cend(s), [](int n) { return n == 7; });
	if (n == 1)
		cout << "The set has 1 element with key 7\n";
	else
		cout << "The set has 0 elements with key 7\n";
}
```

### 110. Map
- std::map in `<map>`
- Associative container
- Each element is of std::pair
  - First member is key
    - A key must be unique
  - Second member is value
- Elements are ordered using < operator of the key
- std::map is implemented as a tree, usually a red-black tree
- Adding elements
  - Using make_pair(): `m.insert(make_pair(k,v));`
  - In C++11, a list initializer: `m.insert({k,v});`
  - insert() will fail if there is a same key
  - insert() returns std::pair
    - When successful, (iterator to the added element, true)
    - When failed: (iterator to the existing element, false)
```cpp
#include <iostream>
#include <map>
#include <string>
using namespace std;
void print(const map<string, int>& scores) {
	cout << "Map elements:" << endl;
	for (auto it: scores)                    // Use a range-for loop to display all the elements
		cout << it.first << " has a score of " << it.second << endl;
	cout << endl;
}
int main() {
	map<string, int> scores;                                              // Create an empty std::map
	scores.insert(make_pair("Maybelline", 86));                           // Add some elements to it
	scores.insert( {"Graham", 78} );
	print(scores);
	cout << "Adding element with key Graham\n";
	auto ret = scores.insert( {"Graham", 66} );
	if (ret.second)
		cout << "Added element with key Graham to map\n";
	else {
		auto it = ret.first;                                            // Iterator to existing element
		cout << "Map already contains an element with key " << it->first;
		cout << " which has value " << it->second << endl;
	}
	print(scores);
	cout << "Removing element with key Graham\n";
	scores.erase("Graham");
	print(scores);
	auto ret2 = scores.insert( {"Graham", 66} );
	if (ret2.second)
		cout << "Added element with key Graham to map\n";
	else {
		auto it = ret2.first;                                            // Iterator to existing element
		cout << "Map already contains an element with key " << it->first;
		cout << " which has value " << it->second << endl;
	}
	print(scores);
}
```
- Map subscripting
  - map supports subscripting
  - If the element doesn't exist, it is created
  - If the element exists, overwrites the value
- Pros and cons
  - Very fast for accessing an arbitrary element
  - Insertion and deletion are usually very fast
  - Very useful for indexed data
  - Useful for key-value pairs like JSON, XML, etc

### Assignment 14: Maps
```cpp
#include<iostream>
#include<vector>
#include<map>
#include<string>
#include<algorithm>
int main() {
   std::string msg;
   std::cout << "Enter some string: " << std::endl;
   // vector
   std::vector<std::pair<std::string,size_t>> vec;
   std::cin >> msg;
   vec.push_back({msg,msg.size()});
   std::cin >> msg;
   vec.push_back({msg,msg.size()});
   for(auto& el: vec) std::cout << el.first << " " << el.second << std::endl;
   return 0;
}
```
```cpp
#include<iostream>
#include<vector>
#include<map>
#include<string>
#include<algorithm>
int main() {
   std::string msg;
   std::cout << "Enter some string: " << std::endl;
   // vector
   std::map<std::string,size_t> mm;
   std::cin >> msg;
   mm[msg] = msg.size();
   std::cin >> msg;
   mm.insert({msg, msg.size()});
   for(auto& el: mm) std::cout << el.first << " " << el.second << std::endl;
   return 0;
}
```

### 111. Maps and Insertion
- Insertion methods in Map
  - Operator[]
    - Insert or Assign functionality
    - New insertion or overwriting
  - Insert member function
    - insert()
    - When failed (when the key exists), value must be updated manually

### 112. Maps in C++17
- C++17 structured binding
  - In a single statement,
    - Declare local variables
    - Bind them to members of a compound data structure
    - The types are deduced by the compiler
- Structuerd bindings
```cpp
std::pair p (1, 3.14);
auto [i,d] = p;
```
  - i is declared as int, having 1
  - d is declared as double, having 3.14
- Loops and structured bindings
  - C++11/14:
```cpp
for (auto el: scores) {
  std::cout << el.first << "has a score of " << el.second << std::endl;
}
```
  - C++17:
```cpp
for (auto [name,score]: scores) {
  std::cout << name << "has a score of " << score << std::endl;
}
```
- map.insert() returns std::pair, containing iterator + true/false of operation
  - Structured binding can split this return value into each
  - `auto [it,success] = m.insert({"Graham"s, 66});`
- insert_or_assign()
  - Takes two arguments of key/value
  - Returns std::pair value
    - When insert works, (new element iterator, true)
    - When assign works, (old element iterator, false)
  - Element iterator will point the key/value of map data
- Initializer in if statement
  - Can use this structured binding in a if() statement, using the returning true/false:
```cpp  
	if (auto [iter, success] = scores.insert_or_assign("Graham"s, 66); success) {
		// new element was inserted
	}
	else {
		// An existing element was updated
	}
```

### 113. Multiset and Multimap
- Similar to set and map but allow duplicate keys
- No subscripting
- insert()
  - Always works for multimap and multiset
- erase()
  - Erase all elements matching the key
  - To erase a single element, iterator must be used
    - c.find(k) will return an iterator to the first element matching key k
      - maps are ordered and same key k elements are contiguous
    - c.count(k) returns the number of elements with key k
    - Looping over the iterator and counts, erase the corresponding it: `if (*it.second == 123) m.erase(it);`
```cpp
#include<map>
#include<iostream>
#include<string>
int main() {
  std::multimap<std::string, int> m;
  m.insert({"abc",1});
  m.insert({"abc",2});
  m.insert({"abc",3});
  m.insert({"xyz",4});
  for (auto [txt,num]: m) std::cout << txt <<" "<< num << std::endl;
  // let's erase abc+2
  auto it = m.find("abc");
  auto count = m.count("abc");
  for (int i=0; i<count; i++) {
     if (it != m.end()) {
       if ((*it).second == 2) m.erase(it); // {"abc",2} is erased
     } 
   ++it;
  }
  for (auto [txt,num]: m) std::cout << txt <<" "<< num << std::endl;
  return 0;
}
```

### 114. Searching Multimaps
- find() and count() to find elements in a multiset or multimap
- c.upper_bound(k) and c.lower_bound(k) return an iterator to the first element whose key is lower/greater than or equal to k
- equal_range() is equivalent to calling lower_bound() followed by upper_bound()
```cpp
#include <iostream>
#include <map>
#include <string>
#include <algorithm>
#include <vector>
using namespace std;
void print(const pair<string, int>& score) {
	cout << "(\"" << score.first << "\", " << score.second << "), ";
}
int main() {
	multimap<string, int> scores;                   // Create an instance of std::multimap
	scores.insert( {"Graham", 78} );                // Add some elements to it
	scores.insert( {"Grace", 66} );
	scores.insert( {"Graham", 66} );
	scores.insert( {"Hareesh", 77} );
	scores.insert( {"Graham", 66} );
	cout << "Multimap elements: " << endl;
	for (auto score : scores)
		print(score);
	cout << endl;
	vector<pair<string, int>> results;                         // vector to store search results
	auto gra_keys = scores.equal_range("Graham");      // Find elements with key "Graham"
	copy_if(gra_keys.first, gra_keys.second,
             back_inserter(results),
                    [](pair<string, int> p) { return p.second == 66;}
	);
	 for (auto result : results)
		 print(result);
	 cout << endl;
}
```

### 115. Unordered Associative Containers
- Associative containers store their elements in an order which depends on the key
- std::set, std::map use a tree
- C++11 introduced "unsorted" associative containers
  - These use a "hash table"
- Buckets
  - A hash table is an array of "buckets"
- Unordered container implementation
  - Buckets are linked lists of pointers to the container's elements
  - The index of the array is calculated from the element
  - Hash function generates a number (hash) based on the key
  - The hash of the key is used as an index into the array
- Hash collisions
  - For maximum efficiency, each element must have its own bucket
    - Perfect hashing
  - In practice, different keys sometimes give the same hash number - hash collision

### 116. Unordered Associative Containers Continued
- C++ unordered containers
  - unordered_set
  - unordered_multiset
  - unordered_map
  - unordered_multimap
- Operations
  - insert(), find(), and erase()
  - Usually faster than sorted containers
  - But can be slower when hash-collisions
- Iterators
  - Allows only forward iteration
    - rbegin()/rend() and crbegin()/crend() are not supported
- Unordered multiset/multimap
  - No support for lower_bound and upper_bound()
  - Still can use equal_range(), find(), count()
- Sorting an unsorted container
  - Copy elements of an unsorted container into a sorted one
    - `copy(umap.begin(),umap.end(), inserter(map, map.end()));`
  
### 117. Associative Containers and Custom Types
- Custom types can be used with all the associative containers
  - Key as string
  - Value as an object of a class
- Ordered associative containers use < operator of their keys to sort their elements
  - When a class object is used as a key, **< operator must be implemented**
- Unordred associative containers use the hash value of their key
  - **==operataor must be implemented**
- Identity
- Equality
- Equivalence
- book_index.h
```cpp
#ifndef BOOK_INDEX_H
#define BOOK_INDEX_H
#include <string>
#include <iostream>
class book_idx {
	std::string author;
	std::string title;
	public:
	book_idx(const std::string& author, const std::string& title) : author(author), title(title) {}
	bool operator < (const book_idx& other) const {
		// If the author is the same, order by title
		if (author == other.author)
			return title < other.title;
		// Otherwise, order by author
		return author < other.author;
	}
	friend std::ostream& operator <<(std::ostream& os, const book_idx& bkx) {
		os << bkx.author << ", " << bkx.title;
		return os;
	}
};
#endif //BOOK_INDEX_H
```
- book_index.cc:
```cpp
#include <iostream>
#include <map>
#include "book_index.h"
using namespace std;
// Class with book details
class book {
private:
    string publisher;
	string edition;
	int date;
public:
    book(string publisher, string edition, int date) : publisher(publisher), edition(edition), date(date) {}
    friend ostream& operator << (ostream& os, const book& bk) {
		os << "(" << bk.publisher << ", " << bk.edition << ", " << bk.date << ")";
		return os;
	}
};
int main() {
	multimap<book_idx, book> library;        // Key is a book_idx object, value is a book object
	// Add some books to the library
	book prog_princs("Addison-Wesley", "2nd Edition", 2014);
	book_idx prog_princs_idx{"Stroustrup, Bjarne", "Programming Principles and Practice"};
	library.insert( make_pair(prog_princs_idx, prog_princs) );
	book cpp_primer("Addison-Wesley", "5th Edition", 2013);
	book_idx cpp_primer_idx{"Lippman, Stanley B.", "C++ Primer"};
	library.insert( make_pair(cpp_primer_idx, cpp_primer) );
	book cpp_prog("Addison-Wesley", "4th Edition", 2013);
	book_idx cpp_prog_idx{"Stroustrup, Bjarne", "The C++ Programming Language"};
	library.insert( make_pair(cpp_prog_idx, cpp_prog) );
	book cpp_tour("Addison-Wesley", "1st Edition", 2018);
	book_idx cpp_tour_idx{"Stroustrup, Bjarne", "A Tour of C++"};
	library.insert( make_pair(cpp_tour_idx, cpp_tour) );
	// Print out all the books
	for (auto b: library)
	   cout << b.first << ", " << b.second << endl;
}
```
- book.cc:
```cpp
#include <iostream>
#include <map>
using namespace std;
// Class with book details
class book {
private:
	string title;
	string publisher;
public:
	book(string title, string publisher): title(title), publisher(publisher) {}
	friend ostream& operator << (ostream& os, const book& bk) {
		os << "(" << bk.title << ", " << bk.publisher << ")";
		return os;
	}
};
int main() {
	multimap<string, book> library;        // Key is the author's name, value is a book object
	// Add some books to the library
	book prog_princs("Programming Principles and Practice", "Addison-Wesley");
	library.insert({"Stroustrup, Bjarne", prog_princs});
	book cpp_primer("C++ Primer", "Addison-Wesley");
	library.insert({"Lippman, Stanley B.", cpp_primer});
	book cpp_prog("The C++ Programming Language", "Addison-Wesley");
	library.insert({"Stroustrup, Bjarne", cpp_prog});
	book cpp_tour("A Tour of C++", "Addison-Wesley");
	library.insert({"Stroustrup, Bjarne", cpp_tour});
	// Print out all the books
	for (auto b: library)
		cout << b.first << ", " << b.second << endl;
}
```

### 118. Nested Maps
- Map inside map
  - `std::map<int, std::map<int,float>> mm;`

### 119. Queues
- std::queue
  - Stored in the order in which they are inserted
  - Elements can be removed from the front only
  - Elements are added from back
  - FIFO
  - Mostly implemented as deque
- Queue operations
  - push(): adds an element to the back of the queue
  - pop(): removes the element from the front of the queue
  - front(): returns the element at the front of the queue
  - back(): returns the element at the back of the queue
  - empty(): returns true when there is no element in the queue
  - size(): returns the number of elements in the queue
- Queue applications
  - Mainly used for temporarily stgoring data in the order it arrived
    - Network data packets waiting for CPU time
    - Must be processed in sequence
- Queue pros and cons
  - Useful for processing events in the order they occur
  - Can only access the front element
  - No provision for queue jumping
```cpp
#include <iostream>
#include <queue>
using namespace std;
void print(const queue<int>& q) {
	cout << "The queue is " << (q.empty() ? "" : "not") << " empty\n";
	cout << "The queue contains " << q.size() << " elements\n";
	cout << "The first element is "<< q.front() << endl;
	cout << "The last element is "<< q.back() << endl;
}
int main() {
	queue<int> q;                                  // Create a queue object
	q.push(4);                                     // Add some elements to it
	q.push(3);
	q.push(5);
	q.push(1);
	print(q);
	// Insert a new element at the end of the queue
	cout << "\nAdding element with value 2\n";
	q.push(2);
	print(q);
	// Remove the first element
	cout << "\nRemoving first element\n";
	q.pop();
	print(q);
}
```

### 120. Priority Queues
- With a standard queue, elements are arranged strictly in arrival order
- When we need to process some elements out of sequence
- Priority queue
  - Orders its elements with the most important at the front
- C++ provides a priority queue
  - < operator of elements is used
- std::priority_queue can be implemented as a vector or a deque
- Interface is similar to std::queue
  - push()
  - pop()
  - top(): returns the element with the highest priority
  - empty()
  - size()
- Applications
  - For processing data with priority
    - OS schedulers
    - Out-of-band communications
      - Command to drop the connection immediately
    - Bug report management system
- Cons and Pros
  - We can only access "top" element
  - Elements with the same priority are not guaranteed to be in arrival order
  - If ordering by arrival time is important:
    - Use a nested map
      - The outer map's key is priority
      - The inner map's key is arrival time, value is data
    - For a class object, redefine < operator    
```cpp
#include <iostream>
#include <queue>
using namespace std;
void print(const priority_queue<int>& pq) {
	cout << "The queue is " << (pq.empty() ? "" : "not") << " empty\n";
	cout << "The queue contains " << pq.size() << " elements\n";
	cout << "The highest priority element is "<< pq.top() << endl;
}
int main() {
	priority_queue<int> pq;                        // Create a queue object
	pq.push(4);                                     // Add some elements to it
	pq.push(3);
	pq.push(5);
	pq.push(1);
	print(pq);
	// Insert a new element in the queue
	cout << "\nAdding element with value 2\n";
	pq.push(2);
	print(pq);
	// Remove the top element
	cout << "\nRemoving top element\n";
	pq.pop();
	print(pq);
}
```

### 121. Stack
- A pile of plates
  - One plate at a time
- Stack
  - A data structure in which elements are stored in the order in which they are inserted
  - When new elements are added to the stack, they are inserted at the top
  - Only the element at the top can be accessed
  - As a stack is processed, the element at the top is removed and the element below it now becomes the top
  - Elements are removed in the reverse order they were added 
  - Last In, First Out (LIFO)
- C++ stack is implemented using deque
- std::stack has a similar interface to priority_queue
  - push()
  - pop()
  - top()
  - empty()
  - size()
- Applications
  - Parsing expressions in compilers
  - Checking unbalanced parentheses in the code
  - Implementing "undo" functionality
  - Storing history for back/forward buttons in a browser
```cpp
#include <iostream>
#include <stack>                                 // Header file for std::stack
using namespace std;
int main() {
	stack<int> s;                                // Create a stack object
	s.push(1);                                   // Add some elements to it
	s.push(2);
	s.push(5);
	cout << "The stack contains " << s.size() << " elements\n";
	cout << "The top element is "<< s.top() << endl;
	cout << "The stack is " << (s.empty() ? "" : "not") << " empty\n";
	// Add a new element to the stack
	cout << "\nAdding element with value 3\n";
	s.push(3);
	cout << "The stack contains " << s.size() << " elements\n";
	cout << "The top element is now "<< s.top() << endl;
	// Remove the top element
	cout << "\nRemoving the top element\n";
	s.pop();                                        // Remove the top element
	cout << "The stack contains " << s.size() << " elements\n";
	cout << "The top element is now "<< s.top() << endl;
}
```

### 122. Emplacement
- push_back()/insert() require an existing object
  - The container copies the existing object into new element
  - If there is no existing object, we need to create one
  - Ex) `std::vector<myClass> vec,b{...}; vec.push_back(b);`
- With emplacement, the container creates the object in the new element, instead of copying it
- Drawback of insert
  - insert() copies its argument into the newly created element
    - `myClass a1(123, 3.14, true); vec.insert(vec.begin(),a1);` Need to create one class object. Then copy a1 into vec element. a1 is not used anymore
    - `vec.insert(vec.end(), myClass(123, 3.14,true));` myClass(123, 3.14,true) is created as a temporary object and copy it to the new leement
- emplace()
  - Pass arguments into the constructor
  - **Avoids any copy**
  - `vec.emplace(vec.begin(), 123,3.14,true);`
```cpp
#include <iostream>
#include <vector>
using namespace std;
class refrigerator {
	// Data members
	int temperature;
	bool door_open;
	bool power_on;
public:
	refrigerator(int temp, bool open, bool power): temperature(temp), door_open(open), power_on(power) {}
	void print() {
		cout << "Temperature = " << temperature;
		cout << boolalpha;
		cout << ", door_open = " << door_open;
		cout << ", power_on = " << power_on;
	}
};
int main() {
	vector<refrigerator> vec;
	refrigerator fridge(2, false, true);                 // Create a refrigerator object
	vec.insert(begin(vec), fridge);                      // Add an element and copy fridge into it
	vec.insert(end(vec), refrigerator(3, false, true));
	vec.emplace(end(vec), 5, false, false);              // Add an element and create an object in it
	cout << "Vector elements:\n";
	for (auto el : vec) {
		el.print();
		cout << "\n";
	}
}
```
- For containers that support push_back(), emplace_back() is available
- For containers that support push_front(), emplace_front() is available
- emplace() for push()
- Constainers with unique keys
  - std::set and std::map
  - emplace() creates a temporary object
  - C++17 provides try_emplace()
    - If the same key exists, it does nothing
```cpp
// Requires a C++17 compiler
#include <iostream>
#include <map>
#include <string>
using namespace std;
class refrigerator {
	// Data members
	int temperature;
	bool door_open;
	bool power_on;
public:
	refrigerator(int temp, bool open, bool power): temperature(temp), door_open(open), power_on(power) {}
	void print() {
		cout << "Temperature = " << temperature;
		cout << boolalpha;
		cout << ", door_open = " << door_open;
		cout << ", power_on = " << power_on;
	}
};
int main() {
	map<string, refrigerator> fridges;
	refrigerator meat_fridge(2, false, true);                      // Create a refrigerator object
	fridges.insert_or_assign("Meat fridge"s, meat_fridge);         // Add an element and copy fridge into it
	fridges.insert_or_assign("Dairy fridge"s, refrigerator(3, false, true));

	auto [iter, success] = fridges.try_emplace("Not in use"s, 5, false, false);
	if (success) {
		cout << "Inserted the new element\n";
	}
	else {
		auto [name, fridge] = *iter;                 // Get the members of the element pair
		cout << "Insert failed: ";
		cout << "existing element with name " << name << " and data ";
		fridge.print();
	}
	cout << "Refrigerators:\n";
	for (auto [name, fridge] : fridges) {
		cout << name << ": ";
		fridge.print();
		cout << "\n";
	}
}
```
- Pros and Cons
  - When an object cannot be copied, emplace() is the only way to insert
  - But still a temporary object can be created if:
    - The container does not allow duplicates (std::set/std::map)
    - The implementation uses assignment than copying
    - A type conversion is required
  - Move semantics can avoid copying temporary objects

### 123. Mastermind Game Practical

### 124. Containers Workshop

## Section 9: Inheritance and Polymorphism

### 125. Class Hierarchies and Inheritance
- Class hierarchy
  - Expresses the relationships
  - Makes it easier for relatd class to re-use code
- Base class
  - The most general or basic version
- Derived class
  - Inherit from or derived from the base class
- Inheritance
  - Relationship b/w classes at different levels in the hierarchy

### 126. Base and Derived Classes
- Deriving a class
  - Put a colon after its name then public keyword and the name of base class
  - The derived class will be able to call the non-private member functions of the base class
  - Can also acces any non-private data members
  - Optionally the derived class can override the base class member functions
- Memory layout
  - Base class object + Derived class object
  - When a derived class object is created, the base class's constructor is called first then the derived class's constructor
  - When destroyed, the derived class's destructur is called before the base class's
- Calling base class constructor in the derived class constructor
  - After colon, the base class constructor:
  - Looks like list initialization but it is a constructor
```cpp
class baseC {
  baseC() {}
  ...
}  
class dervC : public baseC {
  dervC(): baseC() {}
  ...
}
```

### 127. Member Functions and Inheritance
- A child class inherits all the non-private member functions of its parent class
- A child class can rewrite the same function, overriding the original function
  - May call parent function inside of it
- Public access
  - A child class will have access to all its parent's public member data
  - Cannot access parent's private member data
- Protected access
  - Members of a parent class and available to its children but not to other code
    - Maintains encapsulation
  - Use protected access specifier
  - Parent cannot use them

### 128. Overloading Member Functions
- Not only the contents, but we can change the signature
  - Different arguments or const vs non-const
  - Calling previous function signature will not work
- Hidden member functions
  - When we overload an inherited function in the child class, it will hide all the other inherited member functions with that name
  - Those cannot be called on the child class
  - Inconsistent OO design principles
  - Older solution
    - Redefine another function with matching signature
      - Inside of it, just call base class's function
  - Modern solution
    - `using baseC::func;`
    - Makes the hidden function available in the child class
```cpp
#include<iostream>
class vehicle {
public: 
  vehicle()=default;
  void accelerate() { std::cout << "base acceleration\n";}
  void accelerate(double x) { std::cout << "base acceleration at "<< x << std::endl;}
};
class car : public vehicle {
public:
  car()=default;
  using vehicle::accelerate; // allow's parent functions
  void accelerate(int n) {std::cout << " gear " << n << std::endl;}
};
int main() {
  car myCar;
  myCar.accelerate(); // from Base class
  myCar.accelerate(0.12); // from Base class
  myCar.accelerate(7); // overridden at Child class
  return 0;
}
```

### 129. Pointers, References and Inheritance
- A pointer to base class can point to any object in the same hierarchy
  - But not allowed in the other direction due to memory layout: parent + child
```cpp
class Shape {};
class Circle: public Shape {};
...
Circle circle;
Shape *p = &circle; // allowed
Shape shape;
Circle *pc = &shape; // NOT allowed
```
- Base class might be used for vector declaration then can add child class objects
  - From those vector elements, to access child function, `static_cast<>()` might be necessary
```cpp
#include <iostream>
#include <vector>
using namespace std;
class Shape {
public:
	void draw() { cout << "Drawing a generic shape...\n"; }
};
class Circle: public Shape {
public:
	void draw() { cout << "Drawing a circle...\n"; }
};
int main() {
	vector<Shape *> shapes;      // Vector of pointers to Shape instances
	// Create a pointer to a child class of Shape and append it to the vector 
	shapes.push_back(new Circle);
	for (auto s: shapes)
		static_cast<Circle *>(s)->draw();              // Calls Shape::draw()
	for (auto s : shapes)       // Release allocated memory
		delete s;
}
```

### 130. Static and Dynamic Type
- C++ variables have two different kinds of type
  - Static type
  - Dynamic type
- C++ almost always uses static typing
  - Less runtime overhead
  - Better optimization
- The dynamic type is only used for a pointer or reference to a base class
  - So this is why virtual keyword is introduced (?)
```cpp
#include<iostream>
class Shape {
public:
  Shape() = default;
  virtual void draw() { std::cout << "Shape drawing\n";}
};
class Circle: public Shape {
public:
  Circle() = default;
  void draw() { std::cout << "Circle drawing\n";}
};
int main() {
  Circle cobject;
  Shape *sp = &cobject;
  sp->draw();
  return 0;
}
```
  - *sp is a Shape object. But as draw() is virtual in Shape class, it will try to find derived class function
  - Using Shape object, the original draw() still can be called. `virtual` keyword is for polymorphism only

### 131. Virtual Functions
- Static binding
  - Function call is determined at compile time
- Dynamic binding
  - Fuction call is determined at run time
  - Requirements:
    - A member function is called through a pointer or a reference to a base class
    - The member function was declared virtual in the base class
```cpp
#include<iostream>
using namespace std;
class Shape {
public:
  virtual void draw() const { cout << "Drawing a generic shape...\n"; }
};
class Circle: public Shape {
public:
  void draw() const { cout << "Drawing a circle...\n"; }
};
class Triangle: public Shape {
public:
  void draw() const { cout << "Drawing a triangle...\n"; }
};
void draw_shape(const Shape& s) {  // Argument is a reference to a Shape
  s.draw();                        // Calls draw member function of derived class
}
int main() {
  Shape shape;
  Circle circle; 
  Triangle tri;
  draw_shape(shape);  // Base draw()
  draw_shape(circle); // Circle draw()
  draw_shape(tri);    // Triangle draw()
  return 0
}
```

### 132. Virtual Functions in C++11
- When a child class define a member function with the same name as a virtual member function in the parent class
  - When the same signature (same arguments) is found
    - Override
    - Child class function is called from dynamic binding
  - When the function signature (arguments) is different
    - Overloading
    - Prevents dynamic binding
    - May hide parent's member function
- It is easy to overload by mistake
  - When function signature changes
  - Compiler will not say anything
  - How to avoid?
    - Use `override` keyword to confirm that the function is actually overriding
```cpp
class Circle : public Shape {
public:
	void draw() const override {...}
...
};  
```
- If draw() function doesn't override, due to mismatching of signatures, compiler error occurs
  - In such cases, use `using Shape::draw;` to enable the same signature function
- `final` keyword to prevent inheritance. No more child allowed

### Assignment 15: Virtual Functions
```cpp
#include<iostream>
class Base {
 int i;
public:
  Base() {}
  Base(int n) : i{n} {}
  virtual void print() { std::cout << i << " Base\n"; }
};
class Child : public Base {
int i;
public:
  Child() {}
  Child(int n) : Base(n), i{n} {}
  void print() { std::cout << i << " Child\n"; }
};
class Grandchild : public Child {
int i;
public:
  Grandchild() {}
  Grandchild(int n) : i {n} {}
  void print() { std::cout << i << " Grandchild\n"; }
};
void print_class(Base& p) {
  p.print();
}
int main() {
  Base b(1); Child c(2); Grandchild g(3);
  print_class(b); print_class(c); print_class(g); 
  return 0;
}
```

### 133. Virtual Destructor
- Destructor of the base class must be virtual
  - Child class is destroyed, it will call base destructor
- The default destructor synthesized by compiler will NOT be virtual
  - May cuase memory/resource leak or undefined behavior
- Manual implementation of virtual destructor
  - `virtual ~Shape() {}` for all C++
  - `virtual ~Shape = default;` for C++11
- In general, if a class has virtual functions, the destructor must be virtual

### 134. Interfaces and Virtual Functions
- Base class is the interface to the hierarchy
- Pure virtual function
  - Virtual and abstract
```cpp
class Shape {
  public:
   virtual void draw() = 0; // a pure virtual function
};
```  
  - A class with a pure virtual member function cannot be instantiated - abstract base class
  - When a derived class is made, all pure virtual functions must be overriden!!!
    - Otherwise, the derived class will be another abstract class
- An abstract base class cannot be passed by value
- Only can be passed by reference or by address
- In the function body, dynamic binding will be used

### 135. Virtual Function Implementation
- Member functions are not stored in class
  - Handled as global functions
- vtable
  - Class's virtual member function tables
- Virtual function overhead
  - Take 25-50% mroe time than non-virtual functions
  - The class will need vtable
  - A separate pointer for each virtual member function
  - Use virtual member functions only when the extra flexibilty is required

### 136. Polymorphism
- Means "many forms"
- Means different types but the same interface
  - std::vector vs std::string
  - Parametric polymorphism
- Subtype polymorphism
  - The classes in an inheritance hierarchy have the same interface
  - An object of a type can be replaced by an object of its subtype
  - Liskov substitution principle
- Benefits
  - Avoids duplicated code
  - Can save time
  - Ensures correct behavior
- In C++, subtype polymorphism is implemented using pointers or reference to the base class and calling virtual functions on them
  - Dynamic binding
- Polymorphism in C++
  - Subtype polymorphism
    - Run-time overhead
    - May require memory management
    - No control over child classes
    - Can result in large, unmaintainable inheritance hierarchies
  - Parametric polymorphism
    - Compile-time binding
    - No run-time overhead
    - No memory allocation
    - More control over which types are instantiated
- When to use inheritance
  - Often over-used
  - Many problems are better solved by composition
  - The trend in C++ is away from subtype polymorphism towards parametric polymorphism
  - Only use inheritance if you need an "is-a" relationship

## Section 10: Error Handling and Exceptions

### 137. Error Handling
- Issues from runtime environment
- Error communication
  - Tell user what the problem is and what they can do about it
  - Give opportunity to retry, resolve, or ignore as appropriate
- Higher level error handling
  - A better approach is to have that errors handled at a higher level
    - Error needs to be passed from the place where the error occurred to the code that handles it

### 138. Error codes and Exceptions
- In C, error code:
  - The function returns a coded number corresponding to the error
  - The caller checks the return value
  - The caller handles the error itself
  - Or return the error code to its own caller
  - Or return a different error code
- Disadvantages of error codes
  - Makes the code more complicated
  - Large switch statement
  - Adding a new error code requires many changes
  - Error codes do not work well with callback functions
  - C++ constructors cannot return an error code  
- Exception
  - C++ provides exceptions
    - Code which could cause a run-time error is put inside its own scope
    - When an error occurs, creates an exception object and exits the current scope
    - Finds a handler for the exception, depending on the exception's type
    - The handler processes the exception object
  - The programmer specifies the type of the exception object and provides a handler for it
  - The programmer specifies when to create the exception object
  - The compiler will generate the code to create the exception object and invoke the correct handler for it
- Advantages of exceptions
  - Avoids a lot of tedious and error-prone coding
  - When there is no suitable handler, the program terminates immediately
  - An exception object can convey more information
- Disadvantages of exceptions
  - Handling exceptions requires extra code to be executed at run-time
  - Throwing an exception takes much longer time thant returning an error-code
  - The overhead from exception could be significant
  
### 139. Exceptions Introduction
- std::exception
  - Any type can be used for an exceptionobject, including built-in types
  - Has a number of sub-classes: std::out_of_range, etc
  - Has a `what()` virtual member function
    - Returns a C-style string describing the error
- Try block
  - Exception mechanism requires some code to manage exceptions at run time
```cpp
std::vector<int> vec;
try {
  std::cout << vec.at(2) <<std::endl;
}
```
- Catch block
  - The type of exception we are going to handle goes in brackets after the catch keyword
  - Put a reference to base class, dynamic binding will be used and all subclasses will be handled as well
```cpp
catch (cont exception& e)  {
  std::cout << e.what() << std::endl;
}
```
- Uncaught exceptions
  - If an exception is not handled by any of the catch blocks after the try block, the program will try to find one in the "enclosing scope"
  - If it does not find one, it will jump out of the current function and look for a hanlder in the function's caller
  - If it does still not find one, it jumps to that function's caller, and so on
  - If there is no suitable handler, the program terminates

### 140. Try and Catch Blocks
- Catch statements
  - Comes after a try block
  - More than one catch block for the same try block
    - Ordering may matter
    - The most derived class first
    - The base class (std::exception) last
- Writing an exception hanlder
  - Avoid allocating memory, creating new variables, calling functions
  - If possible, only use variables of built-in types
  - Keep the code simple
  - Don't do anything that may throw a new exception
- Nested try/catch blocks
```cpp
try {
  try {

  } catch (const std::bad_alloc& e) {...}
} catch (const stdd:: exception& e) {...}
```
- Might be too much overhead

### 141. Catch-all Handlers
- A catch handler with `...` allows any type of exceptions
```cpp
try {
  ... // some codes here
} catch (...) { // here ... is literal
  std::cout << "Some error "<< std::endl;
}
```
- Useful to put a catch-all handler after the other catch blocks
- Pros and cons
  - Good for testing but not helpful for debugging
  - No information about the error condition
  - Does not capture other events

### 142. Exception Mechanism
- When an exception is thrown:
  - The thrown object is copied into a special area of memory
  - This area is set up by the compiler
  - Every local variable in scope is destroyed, including the original thrown object
  - The program immediately leaves this scope
    - Stops executing any further instructions in the scope
- Stack unwinding
  - The process of repeatedly destroying local variables and exiting the current scope
- Rethrowing an exception
  - When a suitable handler is found, the program executes the code in it and contiues
  - Normally, it will proceed to the next instruction after try/catch block and continue executing from there
  - However, the handler can rethrow the exception
  - In this case, the exception will be handled again, but in a handler belonging to another try/catch block
  - To rethrow the same exception object, use throw with no argument
  - Since the exception is thrown outside the current try block, the program will regard this as a completely new exception
  - A fresh process of stack unwinding begins
```cpp
#include <iostream>
#include <vector>
using namespace std;
void func(const vector<int>& vec) {
	try {
		cout << vec.at(2) << endl;                         // Throws an exception
	}
	catch (const std::out_of_range& e) {
		cout << "Abandoning operation due to exception\n";
		cout << "Exception caught: " << e.what() << endl;  // Print out a description of the exception
		//throw std::exception(e);
		throw;
	}
}
int main() {
	vector<int> vec;
	try {
		func(vec);
	}
	catch (const std::exception& e) {
		cout << "Call to func failed\n";
		cout << "Exception caught: " << e.what() << endl;  // Print out a description of the exception
	}
}
```
- Applications of rethrowing
  - Log the exception at the point where it happens
  - Add extra information to the exception
  - Convert the exception to a higher level type
```cpp
catch(TcpHandShakeFailure& e) {
  logger << e.what() << std::endl; // logging of error
  e.remote_host = remote_host; // extra info of host
  throw NetworkConnectionError(e); // Rethrow high-level exception to UI
}
```

### 143. std::exception Hierarchy
- C++ defines std::exception class
  - Base class for an inheritance hierarchy
- std::exception hierarchy
  - bad_cast
  - bad_alloc
    - bad_array_new_length (C++11)
  - bad_weak_ptr (C++11)
  - bad function_call (C++11)
  - runtime_error
    - overflow_error
    - undeflow_error
    - range_error
    - system_error (C++11)
      - ios_base::failure  (C++11)
  - logic_error
    - out_of_range
    - domain_error
    - invalid_argument
    - length_error
    - future_error (C++11)
- std::exception hierarchy interface
  - constructor
  - copy constructor
  - assignment operator
  - virtual member function what(): returns a C-style string
  - virtual destructor
```cpp
#include <vector>
#include <iostream>
#include <string>
using namespace std;
int at(const vector<int>& vec, int pos) {
	// Check index corresponds to a valid element
	// If not, throw std::out_of_range with a suitable error message
	if (vec.size() < pos + 1) {
		string str{ "No element at index "s + to_string(pos) };
		throw std::out_of_range(str);
	}
	// Return the element
	return vec[pos];
}
int main() {
	//vector<int> vec;
	vector<int> vec{1, 2, 3};
	try {
		int el = at(vec, 2);
		cout << "vec[2] = " << el << endl;
	}
	catch (const std::exception& e) {                           // Will handle all subclasses of std::exception
		cout << "Exception caught: " << e.what() << endl;       // Print out a description of the exception
	}
}
```

### 144. Standard Exception Subclasses
- std::exception subclasses
  - bad_alloc: when memory allocation fails
  - bad_cast: when dynamic_cast fails
  - logic_error: Parent class for error conditions resulting from faulty logic
  - runtime_error: Parent class for error conditions beyond the program's control
  - out_of_range: attempting to access an element outside a defined range
  - invalid_argument: Not acceptable argument
  - domain_error: argument is outside the domain of applicable values
  - length_error: the length limit of an object is exceeded
  - overflow_error: Too large computation result
  - underflow_error: Too small computation result
  - range_error: A value cannot be represented by the result variable
- C++ Standard library and Exceptions
  - Due to efficiency issue, C++ standard library make little use of exceptions

### 145. Exceptions and Special Member Functions
- Destructors and exception
  - When an exception is thrown, destructor is called for all local variables
  - What happens if a destructor throws an exception?
    - Multiple exceptions and undefined behavior
  - Therefore, destructors should never throw exceptions
- Constructors and exception
  - An exception thrown in a constructor should be left for the caller to handle
```cpp
#include <iostream>
using namespace std;
class StudentGrade {
	int grade;
public:
	StudentGrade(int grade): grade(grade) {
		if (grade < 0 || grade > 100) {
			// Invalid grade - throw exception
			throw std::out_of_range("Invalid grade");
		}
	}
};
int main() {
	int result;
	cout << "Please enter the student's grade (between 0 and 100)" << endl;
	cin >> result;
	try {
		StudentGrade sgrade(result);
		cout << "sgrade created\n";   // If we get here, no exception was thrown - safe to use sgrade
	}
	catch (const std::out_of_range& e) {
		cout << "StudentGrade constructor threw an exception:\n" << e.what() << endl;
	}
}
```

### 146. Custom Exception Class
- We can write our own exception classes
- It is best to derive it from subclasses of std::exception
  - Existing interface
  - Inherit code instead of rewriting it
- We do not derive directly from std::exception
  - std::exception has a default constructor only
  - No provision for passinga custom error message
- Custom exception class requirements
  - Needs constructors which take a string, both of std::string and C-style
  - Needs a copy constructor
  - Can override what()
  - Can have data members to store information
- Custom exception class considerations
  - A custom exception object will be copied to stack memory
  - Must be lightweight
  - Perform minimal processing
  - Avoid anything which may throw a fresh exception
- Custom exception class example
  - invalid_student_grade Constructor
    - Derived from std::out_of_range
    - Implement for C-style string and std::string
    - Calls std::out_of_range constructor when the constructor is called
  - invalid_student_grade members
    - For Error string but already in std::out_of_range
    - No member data and no copy constructor is implemented
      - Let the compiler synthesize one
```cpp
#include <iostream>
#include <stdexcept>
#include <string>
using namespace std;
class bad_student_grade : public std::out_of_range {
  public:
   // Default constructor 
   bad_student_grade() : std::out_of_range("Invalid grade: please try again") {}
   // We need constructors which take a string, for consistency with std::exception
   bad_student_grade(const char *s) : std::out_of_range(s) {}
   bad_student_grade(const string& s) : std::out_of_range(s) {}
   // These default operators are good enough as we do not have any data members
   bad_student_grade(const bad_student_grade& other) = default;
   bad_student_grade& operator =(const bad_student_grade& other) = default;
   // Finally, we can override the virtual what() member function
   // const char* what() const noexcept override { /* ... */ }
};
class StudentGrade {
	int grade;
public:
	StudentGrade(int grade) : grade(grade) {
		if (grade < 0) {
			throw bad_student_grade("bad grade");
		}
		if (grade > 100) {
			throw bad_student_grade();
		}
	}
};
int main() {
	int result;
	cout << "Please enter the student's grade (between 0 and 100)" << endl;
	cin >> result;
	try {
		StudentGrade sgrade(result);
		// If we get here, no exception was thrown - safe to use sgrade
		cout << "Valid student grade entered: " << result << endl;
	}
	catch (bad_student_grade& e) { cout << e.what() << "\n"; }
}
```

### 147. Exception Safety
- Implies that the code behaves correctly when exceptions are thrown
- All your programs must be exception safe!
- 3 main ways to write exception-safe code
  - Basic exception guarantee
    - No resource is leaked when thrown
    - All opened files are closed
    - All allocated memory is released
    - All the operations and functions in the C++ standard library provide this basic guarantee
  - Strong exception guarantee
    - When an exception is thrown, the program reverts to its previous state
  - No-throw guarantee
    - The code does not throw any exceptions

### 148. The throw() Exception Specifier
- No-throw guarantee
  - Means that an operation will not throw an exception
  - None of functions and operators in the core C++ language and library throwo exceptions, except:
    - new
    - dynamic_cast
    - throw
- throw()
  - By C++09
  - List of exceptions can be given as list initialization
    - Compiler doesn't check the list. Incorrect list would terminate the program
  - Empty throw() means no-exception thrown    
- Removal of throw() from C++ 
  - Replaced by noexcept
  - throw() was deprecated in C++11
  - throw() with an argument list was removed in C++17
  - throw() with an empty argument list is removed in C++20

### 149. The noexcept keyword
- C++11 introduced noexcept keyword
  - Equivalent to throw() with an empty list
- `void func() noexcept {...}`
  - No exception thrown here
  - If an exception is thrown, the program terminates immediately
  - This is helpful when we write an exception-safe code
- Performance advantages of noexcept
  - More optimization from compiler
  - May use more optimized operators from library
- When to use noexcept
  - Wherever possible
- No overloading on noexcept
  - One of `void func();` or `void func() noexcept;`
- noexcept-ness is inherited through class hierarchy
- A child can add noexcept-ness but not remove it
- Destructors are implicitly noexcept
  - Compiler will assume the destructor is noexcept if:
    - All members of the class have a noexcept destructor
    - All parenet classes have a noexcept destructor
  - However, it is better to write "noexcept" explicitly

### 150. Swap Function
- std::swap() is declared as no-except
  - It copies its argument
- When we write a class, we may overload swap() for that class
  - If copying the object is too expensive, std::swap() may suffer 
```cpp
#include <iostream>
using namespace std;
class String {
private:
	int size;
	char *data;
public:
	String(int size): size(size), data(new char[size]) {}
	String(const String& arg): size(arg.size) {
		cout << "Calling copy constructor\n";
		data = new char[size];              // Allocate the heap memory for arg's data
		for (int i = 0; i < size; ++i)      // Populate the memory with arg's data
			data[i] = arg.data[i];
	}
	// Assignment Operator
	String& operator =(const String& arg) {
		cout << "Calling assignment operator\n";
		if (&arg != this) {                    // Check for self-assignment			
			cout << "Reallocating memory\n";
			delete[] data;                    // Release the original memory
			data = new char[arg.size];         // Allocate the data member
			size = arg.size;                   // Assign to the size member
			for (int i = 0; i < size; ++i)     // Populate the data
				data[i] = arg.data[i];
		}
		return *this;                                  // Return the assigned-to object
	}
	// Destructor
	~String() {
		cout << "Calling destructor: " << static_cast<void *>(data) << endl;
		delete[] data;                     // Release the heap memory for the data
	}
	// Declare overloaded swap() as a friend of this class
	friend void swap(String& l, String& r) noexcept;
	void print() {
		cout << "String with size = " << size;
		cout << ", data address " << static_cast<void *>(data) << endl;
	}
};
inline void swap(String& l, String& r) noexcept {
	cout << "\nIn String::swap\n";
	swap(l.size, r.size);
	swap(l.data, r.data);
}
int main() {
	String a(5), b(6);
	cout << "Before swapping\n";
	cout << "a: ";
	a.print();
	cout << "b: ";
	b.print();
	swap(a, b);
	cout << endl << "After swapping\n";
	cout << "a: ";
	a.print();
	cout << "b: ";
	b.print();
	cout << endl;
}
```

### 151. Exception-safe Class
- Consider our RAII class
- Is this class exception-safe?
- Constructor
  - When allocating array, this cannot be no-except
    - new() uses exception
  - Strong guarantee
- Destructor: no-exception
- Assignment operator: strong guarantee
- Exception safety and RAII
  - With RAII, the constructor and copy constructor automatically provide the strong exception guarantee
  - Destructor automatically provides the no-throw guarantee
  - Assignment operator may offer the strong guarantee if implemented carefully

### 152. Copy and Swap
- For a generic assignment operator:
```cpp
	String& operator =(const String& arg) {
    delete[] data;
    size = arg.size();
    data = new char[size];
    ...
  }
```
- copy-and-swap idiom
```cpp
	String& operator =(const String& arg) {
		String temp(arg);  // 1. using the copy constructor
		swap(*this, temp); // 2. using swap
		return *this;    
	} 
```
- Advantage
  - No need to check for self-assignment
  - Code reuse
  - Much shorter code with less scope for errors
  - Provides the strong exception guarantee
- Disadvantages
  - Always makes anew allocation
  - Creates an extra object, which increases memory consumption

### 153. Comparison with Java and C# Exceptions

## Section 11: Move Semantics

### 154. Move Semantics
- In swap(), we can save time by exchanging data instead of copying it
  - With move semantics, we can save time by moving data in and out of function calls
- Value semantics
  - C++ often use value semantics, in which data is copied
    - By default, function arguments are passed by value
    - By default, functions return by value
  - STL containers copy data into their elements
    - Using references is not allowed - to prevent dangling references
    - Pointers might be used but at the risk of dangle or memory leaks
  - Value semantics avoids the need for garbage collector
  - But requires copying objects
    - Affects the performance
- Move semantics
  - C++11 introduced
  - If the source object is an "rvalue", its data can be moved into the target instead of being copied
  - Similar to moving files b/w folders instead of copying them
- Pass-by-reference (Lvalue References) vs Move semantics (Rvalue References)
  - T& or const T& vs T&& (rvalue references) or std::move

### 155. Lvalues and Rvalues
- An object in C++ can either be an lvalue or an rvalue
- lvalue on LHS of an assignment
- rvalue on RHS of an assignment
- `x=2;`, `x=func()`;
- In C++, an lvalue represents a named memory location
  - It has a name
  - We can take its address using & operator
  - Ex) `x`
- Anything else is an rvalue
  - Ex) `2`, `func()`
- lvalues and rvalues can behave differently when passed as function arguments
- Pass by value
  - lvalues can be passed by value
    - `void func(int); int x=2; func(x);`
  - rvalues can be passed by value
    - `func(2);`
- Pass by address
  - lvalues can be passed by address
    - `void func(int *); int x=2; func(&x);`
  - rvalues cannot be passed by address
- Pass by reference
  - lvalues can be passed by reference
    - `void func(int &); int x=2; func(x);`
  - rvalues cannot be passed by reference
- Pass by address and pass by non-const reference are used when the function wants to modify the argument
  - This doesn't make sens for rvalues, and is not allowed
- Passing rvalue by const reference
  - An efficient way of passing large objects by value
```cpp     
int get() {return 2;} // return value from get() is an rvalue
...
void func(const int&); //
func(get());           // rvalue is passed by const reference
```
- Pass by move
  - In older C++, arguments are copied by default
    - `void func(Type obj);`
  - In C++11, the argument can be moved if
    - obj is an rvalue
    - and Type is "moveable"
    - True for all C++ library types (some exception)
  - obj will be copied if
    - obj is an lvalue
    - or Type is not moveable

### 156. Lvalue and Rvalue References
- Lvalue reference
  - Typically implemented as a pointer
  - Compiler adds code to dereference the pointer as needed
```cpp
int x; 
int &y=x;  // equivalent to int *y = &x;
y=3;       // equivalent to *y = 3;
```
- lvalue reference to rvalue
  - cannot bind an lvalue refernce to an rvalue
    - `int&x = 3;` // Error
  - But const reference works OK
    - `const int&x = 3;` // OK
- Rvalue reference
  - Since C++11
  - This is a syntactic device which indicates that a function argument must be a moveable rvalue
  - rvalue reference has `&&` after the type
```cpp
void func(int&& x);
func(2); // 2 is an rvalue. OK
int y{2};
func(y)  ; // Error. ys is an lvalue
```
- std::move
  - If we want to pass an lvalue to a function which take an rvalue reference, we have to cast it to rvaule
  - std::move() will cast its argument to rvalue
    - `int y{2}; func(std::move(y));`
  - This will move y's data into the function argument x
  - This must only be done if y's data is expendable
    - After calling func(), y's data may be empty or unusable
    - If we want to use y again, we must re-assign its data
```cpp
#include<iostream>
#include<string>
void test(std::string &s) { std::cout << "lvalue ref\n"; }
void test(std::string&& s) { std::cout <<"rvalue ref\n";}
int main() {
  std::string lval {"Hello"};
  std::string& lref = lval;
  test(lval);            // lvalue ref
  test(lref);            // lvalue ref
  test("Hello");         // rvalue ref
  test(std::move(lval)); // rvalue ref
  std::cout << lval << std::endl; // Prints "Hello". local data is still available
  return 0;
}
```
- Argument type summary
  - Lvalue
    - Can pass an lvalue then the function has a copy of the data
    - Can pass a moveable rvalue then the function now owns the passed object'd data
  - Lvalue reference
    - Can pass an lvalue then the function can modify the passed object's data through the reference
  - Const lvalue reference
    - Can pass an lvalue or an rvalue then the unction cannot modify the passed object's data 
  - Rvalue reference
    - Can pass a moveable rvalue but not an lvalue
    - The function now owns the passed object's data

### 157. Value Categories
- Characteristics of lvalues
  - lvalues represent persistent objects
  - Theses occupy memory which is accessible to the programmer
    - On stack or heap
  - They remain valid until they go out of scope or are deleted
    - Local variables
    - Global variables
    - Static variables
    - Function arguments
- Characteristics of rvalues
  - rvalues are stored in locations which are not accessible to the programmer, such as processor registers
  - Literals like 2 or `c`
  - Temporary objects
- C++ has 3 "value categories"
  - Literals
    - No name and cannot be referred to again
    - Pure rvalues or prvalues
  - Temporary objects
    - Represents an object and their data can be moved
    - xvalues
  - Lvalues
- Every expression has a type and a value category
- rvalue = xvalue + prvalue
- glvalue = lvalue + xvalue

### 158. Move Operators
- We can overload a functio so that it behaves differently depending on whether its argument is moved
  - `void func(const myClass& obj);` // pass an lvalue
  - `void func(myClass&& obj);` // pass an rvalue
  - Overload the copy constructor and assignment operator using the same way
- Move operators
  - C++11 added new special member functions for this purpose
    - Move constructor
    - Move assignment oeprator
  - These are called automatically when the passed object is a moveable
```cpp
Test(const Test& arg);     // Copy constructor
Test(Test&& arg) noexcept; // Move constructor
Test& operator=(const Test& arg);     // Copy assignment operator
Test& operator=(Test&& arg) noexcept; // Move assignment operator
```  
- The move operator argument CANNOT be const
- The move operators must not throw exceptions
  - No simple way to recover from a part-completed move
  - STL containers will only call an element's move oeprator if it is noexcept
- The move assignment operator returns the assigned-to object by lvalue reference
```cpp
#include <iostream>

using namespace std;

class MyClass {};

// Class with move constructor
class Test {
private:
	int i{0};
	MyClass m;
public:
	Test() = default;
	// Copy constructor 
	Test(const Test&arg) : i(arg.i), m(arg.m) {
		cout << "Copy constructor called" << endl;
	}
	// Move constructor
	Test(Test&& arg) noexcept : i(arg.i), m(std::move(arg.m)) { // local data are copied while class object is moved
		cout << "Move constructor called" << endl;
	}
	// Copy assignment operator
	Test& operator =(const Test& arg) {
		cout << "Copy assignment operator called" << endl;
		if (this != &arg) {
			i = arg.i;
			m = arg.m;
		}
		return *this;
	}
	// Move assignment operator
	Test& operator =(Test&& arg) noexcept {
		cout << "Move assignment operator called" << endl;
		if (this != &arg) {
			i = arg.i;                       
			m = std::move(arg.m);            // Force move assignment operator to be called
		}
		return *this;
	}
};
int main() {
	Test test;                                 // Call default constructor
	cout << "Copying: ";
	Test test2 = test;                         // Call copy constructor 
	cout << "\nMoving temporary: ";
	Test test3 = Test();                       // Call move constructor by using temporary object
	cout << "\nMoving rvalue: ";
	Test test4(std::move(test));               // Call move constructor by casting test to rvalue
	cout << endl;
	Test test5;
	cout << "\nAssigning: ";
	test5 = test2;                             // Call copy assignment operator
	Test test6;
	cout << "\nAssigning from temporary: ";
	test6 = Test();                            // Call move assignment operator
}
```
- Move operators for derived classes
  - Apply the corresponding operator for the base class when a move operator is used for a derived class
```cpp
// copy constructor
Derived(const Derived& arg): Base(arg) {...}
// Move constructor
Derived(Derived&& arg): Base(std::move(arg)) noexcept {...}
```
- MOVE sematics saves the planet !!!

### 159. RAII Class with Move Operators
- How to add a move constructor and move assignment operator to our RAII class:
```cpp
class String {
  private:
    int size;
    char *data;
  public:
  ...
};
```
- Move constructor and assignment operator
```cpp
	// Move constructor
	String(String&& arg) noexcept {
		data = arg.data;
		size = arg.size;
		// Anything else?
		// "arg" and "this" now have a pointer to the same allocated memory
		// We must make sure "arg" does not delete the pointer
		arg.data = nullptr;                 // Safe - deleting nullptr has no effect
		arg.size = 0;
	}
	// Move assignment operator
	String& operator=(String&& arg) noexcept {
		if (this != &arg) {
			delete[] data;
			data = arg.data;
			size = arg.size;
			arg.data = nullptr;
			arg.size = 0;
		}
		return *this;
	}
```
- Move assignment operator
  - May use move-and-swap
```cpp
String& operator=(String&& arg) noexcept {
  String temp(std::move(arg));
  swap(*this,temp);
  return *this
}
```

### 160. Move-only Types and RAII
- To make a move-only class, implement the move operators and make the copy operators deleted
```cpp
class Test{ 
  public:
    Test(const Test&) = delete;           // delete copy constructor
    Test& operator=(const Test&) = delete; // delete copy assignment operator
    Test(test&&) noexcept;                 // implement move constructor
    Test& operator=(Test&&) noexcept;      // implement move assignment operator
}
```
- Move-only types
  - C++ has some types which cannot be copied but can be moved
    - fstream, iostream
    - Smart pointers
  - These types follow the RAII idiom
    - Only one object can own a given resource instance at a time
    - The object acquires ownership of the resource in the constructor
    - The object releases ownership of the resource in the destructor
  - The onwership of the resource can be transferred from one object to another using move semantics
- fstream
  - Has a file handle as a data member
  - fstream constructor opens a file
  - fstream destructor closes the file
  -  an fstream object cannot be copied but can be moved
    - The moved-from object no longer owns the file handle. It has anull handle
    - The moved-to object becomes the owner of the file handle
- Lambda capture and move-only objects
  - C++11 lambda expressions are not good at capturing move-only objects
  - By C++14, generalized lambda capture
  - A variable in the outer scope can be moved into a lambda-local variable: `[lfs = std::move(fs)]`
```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;
int main() {
	vector<string> strings(5);
	cout << "Capture by reference" << endl;
	[&strings]() { cout << "Size in lambda = " << strings.size() << endl; }();
	cout << "After calling lambda, size in main = " << strings.size() << endl;
	cout << endl << "Capture by move" << endl;
	[vs = std::move(strings)] () { cout << "Size in lambda = " << vs.size() << endl; }();
	cout << "After calling lambda, size in main = " << strings.size() << endl << endl;
}
```

### 161. Special Member Functions in C++11
- Move constructor
- Move assignment operator
- Synthesized move operators if
  - The class does not define a copy constructor, assignment operator or destructor
  - Every data member is either:
    - Built-in type
    - User-defined type with move operators
    - Static data member (not moved)
- Synthesized move constructor will call the move constructor for each member
- Synthesized move assignment operator will call the move assignment operator for each member
- If a class defines a move operator, both the copy operators will be synthesized as "deleted"
  - The class will be move-only
  - If its own copy operators are required, define them

### 162. Using Special Member Functions in C++11
- Rule of zero
  - If a class does not need to declare a destructor (to release memory), it doesn't need to declare copy or move operators
- Rule of 3 (older C++)
  - When a class implements destructor (to release memory), implement copy constructor and copy assignment operator
- Rule of 5 (C++11)
  - When a class implements destructor, implement copy constructor, move constructor, copy assignment operator, and move assignment operator
- Move-only class
  - Ex) network managing class. destructor will close the connection
  - Ex) fstream. destructor will close the file
  - Make the class move-only
  - Provide:
    - Destructor
    - Move constructor
    - Move assignment operator
- Immovable class
  - A class which is immoveable and uncopyable
  - Delete the copy oerators
  - Compiler will not synthesize move operators
- Copy-only class
  - Declare the move operators as deleted
  - But not recommended

### 163. Function Arguments and Move Semantics
- Pass by value
  - Copy constructor called
  - Function has its own copy of the caller's object
  - Caller's object is unmodified
- Pass by const reference
  - Function has read-only access to caller's object
  - Caller's object is unmodified
- Pass by non-const reference and pass by address
  - Function has full access to caller's object
  - Caller's object may be modified
- Pass by move
  - Move constructor called
  - Function now owns the caller object's data
  - Caller's object is unusable (until re-assigned)

### 164. Forwarding References
- Nested references
```cpp
#include <iostream>
using namespace std;
void func(int& x) {
	cout << "func called with argument int&" << endl;
}
int main() {
	int i{42};
	//int& & ri = i;                       // Error
	using int_ref = int&;                  // or typedef int& int_ref;
	int_ref j{i};                          // j is a reference to int
	int_ref& rj{j};                        // rj is a reference to (reference to int)
	func(rj);
}
```
- Reference collapsing rules
```cpp
using lval_ref = int&;
using rval_ref = int&&;
```
  - lval_ref& => int&
  - lval_ref&& => int&
  - rval_ref& => int&
  - rval_ref&& => int&&
- && argument parameters
  - It can only be bound to an rvalue (xvalue or prvalue)
  - When used with Template, it means "forwarding" reference
```cpp
#include <iostream>
using namespace std;
class Test {};
template <class T>
void func(T&& x) {
	cout << "func called" << endl;
}
int main() {
	Test t;
	Test& rt{t};
	// T is Test& and x is T&& => Test&
	func(t);                  // Compiler instantiates func(Test& x)
	// T is Test& and x is T&& => Test&
	func(rt);                 // Compiler instantiates func(Test& x)
	// T is Test and x is T&& => Test&&
	func(std::move(t));       // Compiler instantiates func(Test&& x)
}
```
- Why are forwarding references useful?
  - We can write only one function, without overloading (?)

### 165. Perfect Forwarding
- Forwarding
  - A function that passes some or all of its argument to another function is called to "forward" them
  - `void f(Test x) { g(x);}` // f() forwards the argument x to g()
- With perfect forwarding, the properties of the passed objects are preserved
  - If x is modifiable in f(), it is modifiable in g()
  - If x is unmodifiable in f(), it is unmodifiable in g()
  - If x was moved into f()'s argument, it will be moved into g()'s argumen
  - Ex) `make_pair()`
```cpp
#include <iostream>
using namespace std;
class Test { };
void g(Test& x) {
    std::cout << "Modifiable version of g called\n";
}
void g(const Test& x) {
    std::cout << "Immutable version of g called\n";
}
void g(Test&& x) {
    std::cout << "Move version of g called\n";
}
template <class T>
void f(T&& x) {
    g(x); // will run lvalue. To run rvalue, see below
    // The solution is g(std::forward<T>(x));
}
int main() {
    Test x;
    const Test cx;
    cout << "Calling f() with lvalue argument\n";
    f(x);
    cout << "\nCalling f() with const lvalue argument\n";
    f(cx);
    cout << "\nCalling f() with rvalue argument\n";
    f(std::move(x));
}
```
- move() problem
  - The rvalue verson of f() does not call the rvalue version of g()
  - Inside of f(), x is an lvalue
  - Therefore, when call g(), it should be `g(std::move(x));`
- std::forward()
  - Casts its argument to rvalue reference
  - `std::forward<T>(x);`

### 166. Perfect Forwarding Practical
```cpp
#include <iostream>
using namespace std;
class Test { 
    string m_str;
public:
	Test(const string& str) : m_str(str) {
		cout << "const string& constructor called\n";
	}
	Test(string&& str) : m_str(std::move(str)) {
		cout << "string&& constructor called\n";
	}
};
void g(string& x) {
    std::cout << "Modifiable version of g called\n";
}
void g(string&& x) {
    std::cout << "Move version of g called\n";
}
template <class T>
Test make_test(T&& x) {
    g(std::forward<T>(x));
	return Test(std::forward<T>(x));
}
int main() {
    string hello{"Hello"};
    cout << "Calling make_test() with lvalue argument\n";
	Test t1 = make_test(hello);
    cout << "\nCalling make_test() with rvalue argument\n";
	Test t2 = make_test(std::move(hello));
}
```

## Section 12: Smart Pointers

### 167. Smart Pointers Introduction
- Drawbacks of traditional pointers
  - No concept of ownership
  - No destructor
- From C++11, 
  - Use references instwead of pointers for stack memory
  - Smart pointers for heap memory
  - Smart pointers or references for dynamic binding
- Smart pointers
  - Implemented using RAII
- std::auto_ptr
  - In C++98
  - Removed in C++14
- std::unique_ptr
  - In C++11
    - Cannot be copied or assigned to
    - Can be moved (std::move())
- std::shared_ptr
  - In C++11
  - Can share the memory allocation with other shared_ptr objects
  - Reference counting is used to manage memory
  - Similar to garbage-collected object
  - More overheads than std::unique_ptr

### 168. Unique Pointer
- std::unique_ptr
  - In most situations, the best replacement for traditional pointers which manage heap memory
  - No issue with shallow copying
  - Very lightweight
  - Defined in `<memory>`
  - A template class with a traditional poiner as member
    - `std::unique_ptr<int> p;`
- Initialization in C++11
  - `std::unique_ptr<int> p1 {new int(42)};` // p1's member points to the int on the heap
  - `std::unique_ptr<int[]> p1 {new int[6]};` // fixed size array. p2's member points to the first element
    - Better to use std::array or std::vector though
- C++14 initialization using make_unique:
  - `auto p1 {std::make_unique<int> (42)};`
  - `auto p2 {std::make_unique<int[]> (6)};`
  - std::make_unique() uses perfect forwards its argument to the constructor
- std::unique_ptr as function argument  
  - Using pass-by-move
``` cpp
void func(std::unique_ptr<Point> upp) {...}
auto ptr {std::make_unique<Point>(p)};
func(std::move(ptr)); // std::unique_ptr cannot be copied
```
- Returning std::unique_ptr from function
  - Returning std::unique_ptr object uses move, not copy
```cpp
std::unique_ptr<Poitn>  point_ptr (int x, int y) {
  Point p{x,y};
  auto ptr{std::make_unique<Point>(p)};
  return ptr; // move is used, not copy
}
auto upp {point_ptr(3,6)}
```

### 169. Unique Pointers and Polymorphism
- Vector of class objects
  - Each vector element was created by new, and must be deallocated in the end
  - Unless, memory leak
```cpp
#include <iostream>
#include <vector>
using namespace std;
class Shape {
public:
    virtual void draw() const = 0;
	virtual ~Shape() = default;
};
class Circle : public Shape {
public:
    void draw() const { cout << "Drawing a Circle...\n"; }
};
class Triangle : public Shape {
public:
    void draw() const { cout << "Drawing a Triangle...\n"; }
};
class Square : public Shape {
public:
    void draw() const { cout << "Drawing a Square...\n"; }
};
int main() {
	vector<Shape *> shapes;
	shapes.push_back(new Circle);
	shapes.push_back(new Triangle);
	shapes.push_back(new Square);
	for (auto& it : shapes)
		it->draw();
	for (auto& it : shapes)  // Memory leak if the pointers are not deleted
		delete it;
}
```
- Polymorphism with unique_ptr
  - `std::unique_ptr<Base> pbase {make_unique<Derived>()};`
  - The allocated memory is handled by unique_ptr
```cpp
...
int main() {
	vector<unique_ptr<Shape>> shapes;
	shapes.push_back(make_unique<Circle>());
	shapes.push_back(make_unique<Triangle>());
	shapes.push_back(make_unique<Square>());	
	for (auto& it : shapes)
		it->draw();
  // manual deallocation is not necessary
}
```  
- Factory pattern
  - Calls a function to create a new object
  - The function decides the type of the new object and returns a pointer to it
  - Easy to extend if new child classes are added
- Traditionally, the factory pattern was written with new()
  - Since C++11, unique_ptr
  - Return this unique_ptr from the function
  - Caller's unique_ptr will release the memory when done
```cpp
// Factory function
unique_ptr<Shape> create_shape(int sides) {
	// The returned unique_ptr will be moved into the function's return space
	if (sides == 1)
		return make_unique<Circle>();               // Create 1-sided shape
	else if (sides == 3)
		return make_unique<Triangle>();             // Create 3-sided shape
	else if (sides == 4)
		return make_unique<Square>();               // Create 4-sided shape
	else {
		cout << "Cannot create a shape with " << sides << " sides\n";
		return nullptr;
	}
}
int main() {
	// The returned unique_ptr will be moved into a variable in the caller
	auto pshape{create_shape(3)};
	if (pshape)
		pshape->draw();
	// Memory allocation is automatically released at end of scope
}
```

### 170. Unique Pointers and Custom Deleters
- How to create unique_ptr object
  - Using std::make_unique(), which calls new() internally
  - Call the constructor with a traditional pointer as argument. Unique_ptr takes the ownership
    - `myClass obj; std::unique_ptr<myClass> ptr(&obj);`
- Destructor of unique_ptr calls delete() internally
- We can provide a "deleter"
  - Can use Lambda expression
  - Using decltype
```cpp
#include <iostream>
#include <memory>
/// Code from C Networking API ///
struct destination { /* ... */};
struct connection { /* ... */ };
// Function to open a connection
connection connect(destination dest) { 
	std::cout << "Connecting\n";
	connection conn;
	return conn;
}
// Function to close a connection
void disconnect(connection conn) {
	std::cout << "Disconnecting\n"; 
}
/// End of code from C Networking API ///
// Custom deleter to close network connection
auto end_connection = [] (connection *conn) { disconnect(*conn); };
void get_data(const destination& d) {
    connection conn = connect(d);
	std::unique_ptr<connection, decltype(end_connection)> p(&conn, end_connection);
	std::cout << "Getting data...\n";
}
int main() {
	destination dest;
	get_data(dest);
}
```

### 171. The Handle-Body Pattern
- In OOP, we aim to separate the interface from the implementation
  - Clients do not need to know the inner detais of class
  - Clients do not need to modify their code
- Handle-body pattern
  - We split the class into two parts
  - The handle is an outer class that provides the interface to clients
  - The body is an inner class that provides the implementation
  - When a client creates a handle object, the handle creates a body object
  - When a client calls a member function on the handle object, the call is forwarded to the body

### 172. The pImpl Idiom
- In the pImpl idiom, the handle has a private member which is a pointer to a body object
- pImpl: pointer to Implementation
  - Known as compiler firewall
- The handle class is implemented using RAII
- Applications of pImpl
  - In many large projects like Qt
  - Simplifies product updates

### 173. Reference Counting
- A technique for allowing different objects to share the same resource
- Reference counter
  - An integer to store the number of sharing
- Constructor
  - Allocates the counter
- Destructor
  - Decrements the counter
  - If the counter is non-zero, we must not release the memory as it is still shared
- Copy constructor
- Assignment operator
- Move operator

### 174. Shared pointer
- std::shared_ptr
  - Defined in `<memory>`
  - Uses reference counting
  - When shared_ptr object is copied or assigned, there are no memory operations
  - Insted, the reference counter is incremented
- std::shared_ptr structure
  - Has a private data member which is a pointer to the allocated memory
  - Has another private data member which si a pointer to control block
    - The control block contains the reference counter
- How to create std::shared_ptr object
  - Pass a pointer to its constructor
  - make_shared()
  - Move a unique_ptr into shared_ptr
    - This transfers the ownership of the uniqe_ptr's allocated memory
    - Can be useful with factory functions
    - But we cannot convert shared_ptr into unique_ptr
- Initialization
  - Using make_shared: `auto p1 {std::make_shared<int>(42)};`
  - Calling new() will allocate the memory manually, and this may require the control block to allocate separately. Not recommended
- Assignment
  - `p3 = p1;`
  - p1 counter increments
  - p3 counter decrements
- shared_ptr has more overhead than unique_ptr

### 175. Weak Pointer
- Not a smart poiner
- A safe way of aliasing a shared_ptr
- Avoids the problem of dangling
- std::weak_ptr
  - Bound to a shared_ptr object
  - Doesn't affect the reference count
  - weak_ptr cannot access the shared memory directly
  - Has to be converted back to shared_ptr to access the shared memory
    - Only allowed when the shared_ptr is still valid
- weak_ptr to shared_ptr
  - `shared_ptr<int> sp1 = wptr.lock();`
  - `shared_ptr<int> sp2(wptr);`
  - Will throw std::bad_weak_ptr exception when not valid
  - .lock() is an atomic operation
- Weak pointer applications
  - Cache implementatin

### 176. Weak Pointer and Cycle Prevention
- Cyclic references
  - Objects which have shared_ptr to each other
  
## Section 13: Miscellaneous Features

### 177. Chrono Library Introduction
- Time and dates in C++
  - In herited some functions and types from C
    - `<ctime>`
    - Low level and poorly designed
  - C++11 added the chrono library
    - More precise but more complicated
  - C++20 extends chrono library to handle dates
- Time in traditional C++
  - clock() returns the number of clock ticks since the program started
    - Type of clock_t
    - Can be used for intervals of up to a few minutes
    - Precision is implementation-defined (usually 1microsecond)
  - time()
    - Type time_t by address
    - Number of seconds since 1970
    - Up to several decades
    - Precision is 1 second
- std::chrono
  - Defined in `<chrono>`
  - Three important concepts
    - Clock
      - Start date (epoch) and tick rate
      - C's clock started on Jan 01 1970 and ticks once per second        
    - Time point
      - The number of clock ticks since the epoch, at a given point in time
    - Duration
      - An interval b/w two time points, measured in clock ticks

### 178. Chrono Duration Types
- `<chrono>` defines integer types which represents units of duration
  - hours
  - minutes
  - seconds
  - milliseconds
  - microseconds
  - nanoseconds
- C++20 has days, weeks, months and years
- Duration initialization
  - Constructor takes a single argument
  - `hours a{5};`
- std::chrono literals
```cpp
using namespace std::literals;
auto a = 5h;
auto b = 10min;
```
- Duration type cannot be printed directly. Use .count() member function
```cpp
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;
using namespace std::literals;  // For suffixes
int main() {
	seconds s;                                  // Undefined value
	//cout << "Uninitialized seconds variable has value " << s << endl;
	cout << "Uninitialized seconds variable has value " << s.count() << endl;
	auto hour = 5h;                             // 5 hour interval
	auto mins = 10min;                          // 10 minute interval
	auto sec = 2s;                              // 2 second interval
	auto msec = 20ms;                           // 20 milliseconds
	auto usec = 50us;                           // 50 microseconds
	auto nsec = 80ns;                           // 80 nanoseconds
	cout << "hour = " << hour.count() << " hours " << endl;
	cout << "mins = " << mins.count() << " minutes " << endl;
	cout << "sec = " << sec.count() << " seconds " << endl;
	cout << "msec = " << msec.count() << " milliseconds " << endl;
	cout << "usec = " << usec.count() << " microseconds " << endl;
	cout << "nsec = " << nsec.count() << " nanoseconds " << endl;
}
```
- The construct is explicit
  - `seconds s = 2;` // error. Cannot convert from int
  - `seconds s = 2s;` // OK
  - `seconds s = 1h;` // OK. s= 3600seconds

### 179. Chrono Clocks and Time Points
- std::chrono provides three clocks
  - system_clock
    - Measures wall time using HW system's clock
    - Similar to C library clock
    - The best choice for interactive use
    - Not well suited for measuring time intervals
      - Change of system (like daylight saving) may yield wrong results
  - steady_clock
    - Idealized clock which only goes forward, one tick at a time (monotonic)
    - **The best clock for measuring time intervals**    
  - high_resolution_clock
    - Clock with the shortest tick period supported by the system
    - Implementation-defined. Usually an alias of system_clock or stead_clock
- now() function
  - All three clock types have a static function now()
  - Returns the clock's current time point
  - `system_clock::now()`
  - `stead_clock::now()`
- time_point class represents a point in time
  - now() returns a time_point object
  - dt = time_point_1 - time_point_0
```cpp
#include <iostream>
#include <chrono>
using namespace std::chrono;
long long fibonacci (long long n) {
	return (n < 2) ? n: fibonacci(n-1) + fibonacci(n-2);
}
int main() {
	auto start = steady_clock::now();
	long long n = fibonacci(45);
	std::cout << "Fibonacci number is " << n << std::endl;
	auto finish = steady_clock::now();
	auto elapsed = duration_cast<milliseconds>(finish - start).count();
	std::cout << "Time taken: " << elapsed << " milliseconds" << std::endl;
}
```
- sleep_for()
  - From `<thread>`
```cpp
include <thread>  
...
std::cout << "Wait\n";
this_thread::sleep_for(2s);
std::cout << "Done\n";
```
  - In a single threaded program, this will make the main thread sleep
  - The actual duration may be longer due to the scheduling
```cpp
#include<thread>
#include<iostream>
#include<chrono>
using namespace std::chrono_literals;
int main() {
  std::cout << "Wait 2s\n";
  auto start = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(2s);
  auto finish = std::chrono::steady_clock::now();
  std::cout << "Done\n";
  auto elapsed = std::chrono::duration_cast<std::chrono::millisecond
s>(finish - start).count();
  std::cout << "Time taken: " << elapsed << " milliseconds" << std::
endl;
  return 0;
}
```

### 180. Bitsets
- C has operators for bitwise manipulation of integers
  - Inherited to C++
- C++11 provides std::bitset in `<bitset>`
- Abstraction which represents groups of bits
- std::bitset is a templated type
```cpp
std::bitset<8> b1{"10101110"};  // from string
std::bitset<8> b2{Oxae};        // from integer
std::bitset<8> b3{0b1010'1110}; // from binary constant
```
- Can be used with streams: `std::cout << b1 << std::endl;`
- to_ulong()
- to_ullong()
- to_string()
- size()
- test() checks bounds: `b1.test(8);`
```cpp
#include <iostream>
#include <bitset>
using namespace std;
int main() {
	bitset<8> b1{"10101110"};              // Initialize from string literal - bitset with value 174
	bitset<8> b2{0xae};                    // Initialize from integer (decimal or hexadecimal)
	bitset<8> b3{0b1010'1110};             // Initialize from binary constant (C++14)
	cout << "b1 is " << b1 << endl;                          // Displays 10101110
	cout << "b2 in decimal is " << b2.to_ulong() << endl;    // Displays 174
	cout << "b2 as a string is " << b2.to_string() << endl;  // Displays 10101110
	cout << "b3 is " << b3 << endl;                          // Displays 10101010
	cout << "b1 has " << b1.size() << " bits\n";         //Displays 8
	// Display all the bits in b1
	cout << "The bits of b1 are: ";
	for (std::size_t i = 0; i < b1.size(); ++i)
		std::cout << b1[i] << ",";
	cout << endl;
	// Bit access with bounds checking
	cout << "Trying access to bit 8 of b1 with bounds checking\n";
	try {
		std::cout << "b1 bit "<< 8 << " = " << b1.test(8) << endl;
	}
	catch (std::exception& e) {
		cout << "Caught exception: " << e.what() << endl;
	}
}
```
- Bit operations  
  - set()
  - reset()
  - flip()
- Bitset checks
  - all()
  - any()
  - none()
  - count()

### 181. Tuples
- std::pair review
  - A compound type
  - Two members, first and second respectively
  - `std::pair<std::string,std::string> wordpair("hello"s, "there"s);`
  - `auto wordpair { std::make_pair("hello"S,"there"s)};`
- std::tuple
  - From `<tuple>`
  - Similar to std::pair but many members
  - Elements are accessed by index, not by member name
  - `std::tuple<double, int, std::string> number(1.1, 2, "three"s);`
  - `auto numbers {std::make_tuple(1.1,2,"three"s)};`
- Accessing tuple members
  - `std::get()` with element's index
  - `auto x = std::get<0>(numbers);`
  - `std::get<1>(numbers) = 3; `
  - C++14 allows us to use the type, if unique
  - `auto i = std::get<int>(numbers);`
- Unpacking a tuple
  - Use `std::tie()`
  - `double d; int i; std::string msg; std::tie(d,i,msg) = numbers;`
  - Cannot use auto with tie()
- When to use tuples
  - Ad-hoc dataype
  - Somewhere to store data in short-term
    - No member functions
    - No need to make a new type
    - Elements of different types
- When returning multiple values
  - Traditional solution is to use a struct
  - Tuple enables us to use much smaller code
```cpp
tuple<double,int,std::string> func() {
  ...
  return {1.1, 2, "three"s};
}
...
std::tie(d,i,msg) = func();
```

### 182. Tuples in C++17
- Constructor Template Argument Deduction (CTAD) works with tuples
  - In C++11/14, `std::tuple<int, double, string> tup{1, 1.1, "three"s};`
  - In C++17 with CTAD, `std::tuple tup {1, 1.1, "three"s};`
- Unpacking
  - `auto [d,i,msg]= func();`
- Unpacking tuple into function arguments
  - `std::apply()`
  - The first argument is the function name
  - The second argument is a tuple containing arguments
```cpp
void func(int i, double d, std::string s);
...
std::apply(func, std::tuple(1, 2.0, "three"s));
```
- Unpacking tuple into constructor arguments
  -  Use `std::make_from_tuple<T>()`
  - `auto test = std::make_from_tuple<Test>(tup);` is equivalent to `Test(1,2.0,"three"s);`

### 183. Unions
- Inherited from C
- A compound data structure
  - Each member has a distinct type
  - All the members are stored at the same address
  - Only one member can be in used at a time
- Union characteristics
  - All members are public by default
  - Can have member functions but not virtual
  - No base/derived class
  - Data member share the same memory space
    - This is why only one member at a time
- Union usage
  - If we assign to char member, that member is now "in use"
    - Only one member at a time
    - The other member is not reached, and yields undefined value
- Tagged union
  - Use of unions is higly error-prone
  - The programmer has to remember which type is in use
  - May add a "tag" to indicate which member is in use
```cpp
#include <iostream>
using namespace std;
union Token {
	// Members are public by default
	char c;
	int i;
	double d;
};
int main() {
	Token token;
	token.c = 'Z';               // char member is in use
	cout << token.d << endl;    // Double member not in use - has undefined value
}
```
- Why union vs tuple?
  - Union can achieve memory efficiency  or low-level interoperability

### 184. Unions Continued
- Wrapped tagged union
  - A tagged union requires the programmer to check before accessing a member
  - For safety, wrap the tagged union inside a class
    - The union is a private member 
    - It can only be accessed by the class's public member function
    - Performs the necessary setting and checking of the tag member
    - This enforces the correct use of the union
- C++17 std::variant
  - Similar to a wrapped tagged union but
    - Type safe
    - Can have different alternatives with the same type
    - Automatically calls constructors and destructors, when required
    - Simpler to use
  - Defined in `<variant>`
```cpp
std::variant<char, int, double> v;
v = 'Z';
std::get<char>(v);
std::get<0>(v);
```
- `std::holds_alternative()` to check whether an alternative is in use
```cpp
if (std::holds_alternative<double>(v))
  std::cout << std::get<double>(v) << std::enld;
else
  std::cout >> "Double-alternative is not in use\n";
```
- Applications of unions and std::variant
  - Parsing
  - Language implementation
  - Return values
  - Polymorphism

### 185. Mathematical Types
- std::valarray
  - Vector-like type based on Fortran arrays
  - Fixed size
  - Easier synatx for numerical operations
  - Not widely used
  - Not well optimized (?)
  - Better to use a third party library (eigen, blaze, armadillo)
- std::complex from `<complex>`
  - Templated type
  - float, double, or long double
  - For std::cin/cout, `(r,i)` works OK
  - ==operator is defined already
  - ++/-- not working
  - By C++14, literal suffix i is supported: `auto z = 3.0 + 4i;`

### 186. Bind
- `std::bind` from `<functional>`
- Takes a callable object as its first argument
- bind() performs a partial function call
- It makes a new callable, using the remaining arguments
- `auto match_cat = bind(match, "cat");` yields a new callable, `match_cat();`, which is equivalent to `match("cat");`
- When multiple arguments are required, use a place holder like `_1`, `_2`, from `std::placeholders`
  - `auto match_cat = bind(match,_1, "cat");` yields a new callable while `match_cat("dog);` is equivalent to `match("dog","cat");`
```cpp
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
using namespace std;
using namespace std::placeholders;
bool match(const string& animal, const string& species) {
	cout << R"(1st argument is ")" << animal << R"(", 2nd argument is ")" << species << "\"" << endl;
	return animal == species;
}
int main() {
	auto match_cat = bind(match, _1, "cat");                  // Captures "cat" as second argument
	if (match_cat("dog")) {                                   // Calls match("dog", "cat");
		cout << "Matched\n";
	}
	else {
		cout << "Not matched\n";
	}
}
```
- This is equivalent to a C++14 lambda which creates a local variable
- bind1st() and bind2nd() are deprecated in C++11 and removed in C++17

### 187. Callable Objects
- An object supports () operator
- Different callable objects
  - Pointer to non-member function
  - Functors
  - Lambda expressions
  - Objects returned from calling bind()
- Different callable objects which have the same signature
- std::function in `<functional>`
  - This class has a private member which stores a callable object
    - This can be any callable object which has a given signature
  - Template class
    - The object's signature is the parameter
      - `std::function<bool(const std::string&)> match_ptr;`
  - Implemented using inheritance
  - std::function performs type erasure
    - The object's original type cannot be recovered
- Uses of std::function
  - We can declare as function argument as type std::function
  - We can create a container whose elements are of type std::function
- Limitations of std::function
  - No support for overloading
  - Run-time overhead
  - Invoking the callable object involves indirection
  - May perform a memory allocation
- Alternatives to std::function
  - Must be used only when we need a polymorphic function object
  - For string a callable object in a variable, use auto
```cpp
#include <iostream>
#include <string>
#include <vector>
#include <functional>
using namespace std;
using namespace std::placeholders;
// Non-member function
bool match(const string& test) {
    return test == "cat";
}
// Functor
class functor_match {
	public:
	bool operator()(const string& test) {
		return test == "cat";
	}
};
bool bind_match(const string& animal, const string& species) {
    return animal == species;
}
int count_strings(vector<string> &texts, function<bool(const string&)> match_ptr) {
    int tally = 0;
    for(auto text : texts) {
        if(match_ptr(text)) {
            ++tally;
        }
    }
    return tally;
}
int main() {
	vector<string> animals = {"cat", "dog", "tiger", "lion", "bear", "cat", "giraffe"};
	int n;
	// Function pointer
	cout << "Calling count_strings with function pointer\n";
	n = count_strings(animals, match);
	cout << R"(The vector contains )" << n << R"( occurrences of the word "cat")" << endl;
	// Functor
	cout << "\nCalling count_strings with functor\n";
	n = count_strings(animals, functor_match());
	cout << R"(The vector contains )" << n << R"( occurrences of the word "cat")" << endl;
	// Lambda expression
	cout << "\nCalling count_strings with lambda expression\n";
	n = count_strings(animals,
		[](const string& test) { return test == "cat";}
		);
	cout << R"(The vector contains )" << n << R"( occurrences of the word "cat")" << endl;
	// bind() object
	cout << "\nCalling count_strings with bind() object\n";
	auto match_cat = bind(bind_match, _1, "cat");                    // Captures "cat" as second argument
	n = count_strings(animals, match_cat);
	cout << R"(The vector contains )" << n << R"( occurrences of the word "cat")" << endl;
}
```

### 188. Member Function Pointers
- Can define a pointer to a member function of a class
- More complicated tha with function pointers
  - Requires different syntax
  - May need to provide "this" pointer
- Member function pointers cannot be called directly  
  - Must be explicitly derefereneced
- For a class: `class Test { ... ; void func(int a, const std::string&b);};`
  - Define a function pointer: `void(Test::*pfunc) (int,const std::string&);`
  - Make a pfunc pointer: `pfunc = &Test::func;`
  - In C++11, `auto pfunc = &Test::func(int, const std::string&);`
  - May use a type alias: `using PfuncType = void (Test::*)(int, const string&); std::vector<PfuncType> ptrs;`
- std::mem_fn() in `<functional>`
  - Member function pointer
  - `auto f = std::mem_fn(pfunc);`
```cpp
#include <iostream>
#include <functional>
using namespace std;
class Test {
public:
	void func(int a, const string& b) {
		cout << "func called with arguments " << a << " and " << b << endl;
	}
};
int main() {
	// Define pfunc as a pointer to a member function of Test
	auto pfunc = &Test::func;
	// Generate a callable object from pds
	auto f = mem_fn(pfunc);
	// Invoke this callable object
	Test test;
	f(test, 42, "Hello"s);                      // Pass the "this" object as the first argument
}
```
- May use std::bind()
- Applications of member function pointers
  - Concurrency
  - Run-time decision making
  - Choose which object is used
  - Choose which member function is called
  -
### 189. Interfacing to C
- Many interfaces are written in C
  - OS API
  - Database API
  - Third party libraries and frameworks
  - Language with foreign function interfaces
- Source code compatibility
  - C is almost complete subset of C++
    - C99 introduced some incompatible features
  - The simplest solution is to compile the entire source code with C++ compiler
  - May not work
    - C code contains incompatibilities
    - When C source code is not available
- Binary compatibility
  - Object, static library or shared object
  - C and C++ compilers must be compatible
- Name mangling
  - C++ compiler manges the name while C doesn't
- Extern "C" directive
  - No name mangling in C
  - To make C++ binaries compatible with C, we use extern "C" directive
    - This tells C++ compiler not to mangle the function name
```cpp
extern "C" int ifunc(int x);
...
extern "C" {
  int func(int x);
  double dfunc(double x);  
}
...
extern "C" {#include "cstuff.h"}
```
- Exporting to C
  - When writing C++ function that will be called from C
    - Built in types only
    - Array/pointer in built-in type
    - Structs which have members of built-in types only
  - Put the function in the global namespace
- Conditional compilation
  - For C++ compiler, `__cplusplus` preprocess symbol
    - Use this like `#ifdef __cplusplus` to hide some code for C copmiler
- Adding C++ code into C
  - add.cc
```cpp
#include "add.h"
int add(int x, int y) {
	return x + y;
}
```
  - add.h
```cpp
#ifndef ADD_H
#define ADD_H
#ifdef __cplusplus
extern "C" int add(int x, int y);
#else
int add(int x, int y);
#endif
#endif //ADD_H
```
  - add_main.c
```c
#include <stdio.h>
#include "add.h"
int main() {
	printf("add(2, 3) returns %d\n", add(2, 3));
}
```
- Converting sequential containers to arrays
  - std::string and std::vectoir have member function .data()
- Using C function in the C++ code
  - array.h
```c
#ifndef ARRAY_H
#define ARRAY_H
int array_print(int arr[], int n);
#endif //ARRAY_H
```  
  - array.c
```c
#include <stdio.h>
#include "array.h"
int array_print(int arr[], int n) {
	int i = 0;
	for (i = 0; i < n; ++i)
		printf("%d,", arr[i]);
	printf("\n");
}
```  
  - array_main.cc
```cpp
#include <iostream>
#include <vector>
extern "C" {
	#include "array.h"
}
using namespace std;
int main() {
	vector<int> vec {3, 1, 4, 1, 5, 9};
	cout << "Vector elements: ";
	for (auto v : vec)
		cout << v << ", ";
	cout << endl;
	cout << "Calling array_print(): ";
	array_print(vec.data(), vec.size());
}
```  

### Assignment 16: Interfacing to C
- Using array.c/h above:
```cpp
#include <iostream>
#include <set>
extern "C" {
	#include "array.h"
}
int main() {
  std::set<int> aset {1,2,3};
  auto nsize = aset.size();
  int *carray = new int[nsize];
  int i = 0;
  for (auto& el: aset) { carray[i] = el; i++;}
  array_print(carray, nsize);
  delete [] carray;
  return 0;
} 
```

### 190. Run-time Type Information
- RTTI relates to the dynamic type of the object
  - typeid
  - type_info
  - dynamic_cast
- `std::typeid()` returns information about the dynamic type of its arguments
  - From `<typeinfo>`  
  - Mainly used in comparisons
```cpp
Circle d;
Shape *pShape = &d;
...
if (std::typeid(*pShape) == std::typeid(Circle)) {...}
```
- `std::type_info` from `<typeinfo>`
  - Contains the information about the dyanmic type of an object
  - `std::type_info::name()` returns C-style string
```cpp
#include <iostream>
#include <typeinfo>
using namespace std;
class Shape {
	public:
	virtual ~Shape() {}
};
class Circle : public Shape {};
class Triangle : public Shape {};
int main() {
	Circle circle;
	Triangle triangle;
	//Shape *pShape = &circle;
	Shape *pShape = &triangle;
	const type_info& tShape = typeid(*pShape);
	const type_info& tCircle = typeid(circle);
	cout << "Dynamic type of pShape is " << tShape.name() << endl;
	cout << "Dynamic type of circle is " << tCircle.name() << endl;
}
```
  - `std::type_info::hash_code()`
    - Can be used for comparison
- `dynamic_cast` converts a pointer to Base to a poiner to Derived
  - When failed, nullptr is returned

### 191. Multiple Inheritance
- A derived class has more than one parent
- Controversial
  - Adds complexity
  - Many programmers and OO designers dislike it
  - Some languages do not support it
- But useful for writing "mix-in" classes
- Memory layout
  - base1 + base2 + derived class
- The derived class inherits the members of all parents
- Class scope
  - When both parents have the same name of member functions
  - Call the function name with the namespace of the parent

### 192. Virtual Inheritance
- In multiple inheritance, both parents class may have the same base class
- Virtual inheritance
  - When parents inherit from the base, use virtual keyword
  - `class SalesEmployee: public virtual Employee {...};`
  - Resolves the issue of diamond inheritance
```cpp
#include <iostream>
using namespace std;
class Employee {
public:
	void address() { cout << "Employee address = " << static_cast<void *>(this) << endl; }
};
class SalesEmployee: public virtual Employee {
};
class Manager: public virtual Employee {
};
class SalesManager: public SalesEmployee, public Manager {
};
int main() {
	SalesManager sm;
	cout << "Cast to SalesEmployee\n";
	SalesEmployee* se{static_cast<SalesEmployee *>(&sm)};
	se->address();
	cout << "Cast to Manager\n";
	Manager* m{static_cast<Manager *>(&sm)};
	m->address();
}
```  

### 193. Inline Namespaces
- Compressing steps of namespaces
- Useful for multi-versioning
```cpp
namespace MyLibrary {
  namespace V1 { // Older version
    void old_function();
  }
  inline namespace V2 { // Newer, preferred version
    void new_function();
  }
}
```    
- MyLibrary::new_function(); // Automatically uses V2::new_function()
- MyLibrary::V1::old_function(); // Explicitly uses the older version

### 194. Attributes
- Vendor provided compiler directives
  - #pragma
  - __attribute
  - __delspec
- C++11 introduced attributes to provide a standard syntx
  - Gives extra information to the compiler
  - `[[ noreturn ]] void server();`
  - `[[gnu::always_inline]]`
- C++14 added `[[deprecated]]`
  - `[[ deprecated ("Use version 2.0 in new code) ]] void func();`
- C++17 added `[[nodiscard]]`
  - Compiler gives warning when the return value is not received

## Section 14: Compile-time Programming

### 195. Compile-time Programming Overview
- Compile time programming
  - Compiler executes the code at compile time
  - The reuslt of computation is available in the program
  - No run-time overhead
- Compile-time programming in C
  - Uses preprocessor macro functions
  - Copy/paste with substitution
  - No type information
  - Error prone
- Template programming features
  - It was intended for generic programming
  - Provides a Turing-complete programming language
- Template metaprogramming
  - Mainly by library developers
  - Making decisions at compile time for efficiency and/or portabilty
  - Domain specific programming
  - Expressing complex software patterns and concepts
  - Uses class templates rather than functions
  - Heavy use of template specialization
- Compile-time programming
  - Templates
    - Clumsy and verbose syntax
    - Logcial operations are complicated
    - Incomprehensible error messages
    - No debuggr support
  - C++11 Also  provides constexpr
    - The compiler executes normal C++ code
    - Understandable error messages

### 196. Constant Expressions
- A value that is evaluated at compile-time and which cannot change
  - A literal
  - A value computed from literals
  - A value computed from other constant expressions
- constexpr
  - In C++11
- constexpr vs const
  - const is used for function argument/return
  - constexpr variable is known at compile time
    - For constant values
    - Improves performance by doing computations at compile time
```cpp
 std::array<int,5> arr1; // works OK
 int i {5}; std::array<int,i> arr2; // compile error
 const int n {5}; std::array<int,n> arr3; // works OK
```
### 197. Constexpr Functions
- In C++11
  - Arguments are const expressions
  - Return a constant expression
  - Executed at compile time
  - `constexpr double miles_to_km(double miles) { return miles *1.602;}`
  - Must be pure - cannot modify the arguments
  - Implicitly defined as inline
    - Multiple definitions allowed
- Run-time constexpr functions
  - A constexpr function can be called with non-const arguments
  - The return value will not be a constant expression
  - The function will be evaluated at runtime
  - If it is received by constexpr variable then errors out
```cpp
// constexpr function
constexpr double miles_to_km(double miles) { return miles * 1.602; }
// The argument is a constant expression
// The return value is a constant expression
// The function is evaluated at compile time
const double dist1 = miles_to_km(40);
// The argument is not a constant expression
// The return value is not a constant expression
// The function is evaluated at run time
double arg{40};
double dist2 = miles_to_km(arg);
// The argument is not a constant expression
// The return value is not a constant expression
// The return value is required to be a constant expression
// Error!
//constexpr double dist4 = miles_to_km(arg);
int main() {}
```
- Applications
  - Useful for performing calculations at compile time
  - No runtime overhead
- constexpr member functions
  - A member function can be made constexpr
    - Takes constexpr arguments
    - Returns a constexpr
  - In C++11, constexpr member functions were also const - cannot modify "this"
  - In C++11, constexpr member functions are allowed to modify "this", unless also declared as const
- constexpr members
  - Initialized from constant expression
  - Cannot be modified
  - Must be declared static - no dynamic memory

### 198. Classes and Templates
- Template classes work very much like other classes
  - Friend functions and classes
  - Static members
  - Member function with default arguments
- A templated member functions of a class is called a member template
- A member template cannot be virtual
- Member template with different parameters
  - Func has to be a callable object which can compare two objects of type T
- C++20 has concepts
  - Concepts allow us to express the requirement as part of the template definition
  - Clearer code
  - Compiler errors will clearly state
```cpp
#include <iostream>
using namespace std;
// Templated class with parameter T
template <typename T>
class comparer {
	T t1, t2;
public:
	comparer(const T& t1, const T& t2): t1(t1), t2(t2) {}
	// Member template with parameter Func
	template <typename Func>
	bool compare(Func f) { return f(t1, t2); }
};
int main() {
	int x{1}, y{2};
	// Instantiate a comparer object where T is an int
	comparer<int> c(x, y);
	cout << "comparer class: t1 = " << x << ", t2 = " << y << endl;
	// Instantiate comparer::compare() where Func is a lambda expression
	auto b = c.compare([](int i1, int i2) { return i1 < i2; });
	cout << "Result of calling compare: " << boolalpha << b << endl;
}
```

### 199. Template Specialization
- Templates are generic
- We may want a different behavior for some types
  - Template specialization
  - Use `template <>` then add `<datatype>` next to the class name
```cpp
template <typename T> // generic definition
class Vector {
public:
	void identify() {
		cout << "Vector<T>\n";
	}
};
// Specialization of Vector class for bool type
template <>
class Vector<bool> {
public:
	void identify() {
		cout << "Vector<bool>\n";
	}
};
```
- Specialization must follow the generic one
- Partial specialization
```cpp
// generic
template <typename T>
class Vector {...};
// partial specialization for pointer
template <typename T>
class Vector<T*> {...};
```

### 200. Extern Templates
- Template bloat
  - One header file has `template <typename T> void func(arg<T>) {...}`
  - File_001.cc has `func(this_string);`
  - File_999.cc has `func(that_string);`
  - Every File_nnn.o will contain th ebinary code for func
  - But the linker will remove the duplicate definitions
  - Still serious problem in large projects as those object files have duplicated definitions
- extern keyword
  - extern declaration means that the defintion is in an else file
- extern template
  - In the header file:
```cpp
template<typename T> void func(arg<T>) {...}
extern template void func(string);
```
  - In many files, func(string) can be called but instantiated somewhere
  - There must be exactly one file where the function is instantiated: `template void func(string);`

### 201. Variadic Templates
- Variadic functions
  - Can take any number of arguments
  - Last parameter is `...`
  - Not type-safe
  - Only work properly with C-types
  - Arguments are processed at run time
- C++11 introduced variadic template functions
  - `template <typename... Args> void func(Args.. args);`
  - `...` lists are known as parameter packs
- Type deduction
  - `func("hello"s);` // Equivalent to template <typename T> func(T t);
  - `func(442, 0.0);` // Equivalent to template<typename T, typename U> func(T t , U u);
- Parameter packs
  - Available only at compile time
  - Three things are allowed
    - sizeof...() to get the number of elements
    - make_tuple() to store them in tuple
    - Iterate over the elements, using template recursion
```cpp
#include <iostream>
#include <string>
using namespace std;
template <typename... Args>                 // Args is a list of types (template parameter pack)
void func(Args... args) {                   // args is a list of arguments whose types match Args
	cout << "Compiler has deduced call with " << sizeof...(args) << " argument(s)\n";
}
int main() {
	int i{42}; double d{0.0}; string s{"text"};
	func(s);                                // Instantiated as func(string);
	func(i, d, s);                          // Instantiated as func(int, double, string);
}
```
- Template recursion
```cpp
template <typename T, typename U, typename V>
void func(T t, U u, V v) {
  func(u,v);      // recursive call
}
template<typename U, typename V>
void func(U u, V, v) {
  func(v);        // recursive call
}
template <typename V>
void func(V v) {} // End of recurssion
```
  - Pack processing
    - Each call removes the first element from the list
```cpp
template<typename T, typename... Args>
void func(T t , Args... args) {
  // do something
  func(args...);
}
```
  - Terminate the recursion
    - Write another template with a single argument
    - This must be declared above the variadic template
```cpp
#include <iostream>
#include <string>
using namespace std;
template <typename T>
void func(T t) {
	cout << "Non-variadic template\n";
	cout << "Processing argument " << t << endl;
}
template <typename T, typename... Args>      // Args is a list of types (template parameter pack)
void func(T t, Args... args) {               // args is a list of arguments whose types match Args
	size_t n_args = sizeof...(args) + 1;
	cout << "Variadic template: compiler has deduced variadic call with " << n_args << " argument(s)\n";
	cout << "Processing argument " << t << endl;
	func(args...);
}
int main() {
	int i{42}; double d{0.0}; string s{"text"};
	func(i, d, s);                           // Instantiated as func(int, double, string);
}
```

### 202. Miscellaneous Template Features
- assert()
  - Inherited from C
  - Defined in `<cassert>`
  - Checks its argument at runtime. If the argument is zero, it calls std::abort(). Otherwise, the program continues normal execution
  - Useufl for checking invariants: `assert(x==42);`
  - Can be disabed by `#define NDEBUG`
- static_assert()
  - In C++11
  - Takes a constant bool expression and a string literal
  - Bool expression is checked during compilation
  - Mainly used in template metaprogramming
- Default template parameters
  - `template <typename T=int> class number { ...};`
  - Same instantiation for other datatype
  - For default parameer, leave `<>` empty: `number <> myObject(2);`

### 203. Library-defined Operators
- Library-defined operator objects
  - C++11 defines some generic operator classes in `<functional>`
  - Arithmetic operators
  - Relational operators
  - Logical operators
  - Bitwise operators
```cpp
#include <iostream>
#include <functional>
using namespace std;
// Function template with parameter Func
// If caller does not provide a callable object, use T's operator <
template <typename T, typename Func = less<T>>
bool compare(const T& t1, const T& t2, Func f = Func()) { 
	return f(t1, t2); 
}
int main() {
	int x{1}, y{2};
	cout << "x = " << x << ", y = " << y << endl;
	auto b = compare(x, y, [] (int i1, int i2) { return i1 < i2; });
	cout << "Result of calling compare with lambda: " << boolalpha << b << endl;
	auto b2 = compare(x, y);
	cout << "Result of calling compare with default: " << boolalpha << b2 << endl;
}
```

### 204. Constexpr If Statement
- `constexpr if` was added in C++17
  - Allows conditionals to be evaluated at compile time
```cpp
if constexpr (a<b)  
...
else
...
```
  - Only the true branch is compiled
  - False granch is ignored
- constexpr if vs preprocessor directives
  - C++ has preprocessor directive like `#if` and `#ifdef`
    - Simple tex-based substitution
    - No understanding types or C++ syntax
    - Arguments are not evaluated
  - `constexpr if` is executed at compilation
    - Has access to all the compiler's internal data and type information
- Advantages of constexpr if
  - Template specialization
  - SFINAE and enable_if
    - Complex and obscure code
    - Hard to maintain
  - constexpr if
    - Single function with normal looking code

### 205. Constexpr If Examples
```cpp
// Requires C++17 compiler
#include <iostream>
#include <string>
using namespace std;
template <typename T, typename... Args>      // Args is a list of types (template parameter pack)
void func(T t, Args... args) {               // args is a list of arguments whose types match Args
	size_t n_args = sizeof...(args) + 1;
	cout << "Variadic template: compiler has deduced variadic call with " << n_args << " argument(s)\n";
	cout << "Processing argument " << t << endl;
	
	if constexpr(sizeof...(args) > 0)
		func(args...);
}
int main() {
	int i{42}; double d{0.0}; string s{"text"};
	func(i, d, s);                           // Instantiated as func(int, double, string);
}
```

### 206.The decltype Keyword
- decltype
  - Figures out the declaration type
  - Added in C++11
  - Used at compile time
  - Does not evaluate its argument
  - Not executed
  - Just the compiler relaces decltype with the type of the argument would return
- Unlike auto, decltype will retain const, reference, etc
- decltype with lvalues
  - int x;
  - decltype(x) // Gives "int"
  - decltype(x+y) // Gives "int&"
  - decltype((x)) // Gives "int&"
- decltype with rvalues
  - decltype(2) // Gives "int"
  - decltype(Test()) // Gives "Test&&"

## Section 15: Project: A Breakout Game Using Modern C++ with SFML

### 207. Project Breakout
- Inspired by a talk by Vittorio Romeo

### 208. SFML Introduction
- Simple and Fast Multimedia Library
  - Written in C++
  - Easy to use
  - Very well documented
  - www.sfml-dev.org
- At ubuntu: `sudo apt-get install libsfml-dev`

### 209. Compiler Configuration for SFML
- -I path to SFML include files
- -L path to SFML library files
- Compile command: `g++ main.cc -I/usr/include/ -L/usr/lib/x86_64-linux-gnu -lsfml-system -lsfml-window -lsfml-graphics`

### 210. Basic Window
- Defined from `<SFML/Graphiics.hpp>`
- namespace sf
- sf::Window
- sf::RenderWindow
  - clear()
  - draw()
  - display()
- sf::Event
- constants.h 
```cpp
#ifndef CONSTANTS_H
#define CONSTANTS_H
// Code for a "breakout" game
// Based on a talk by Vittorio Romeo
// Uses the SFML graphics library
struct constants {
  static constexpr int window_width{520};
  static constexpr int window_height{450};
};
#endif // CONSTANTS_H
```
- main.cc:
```cpp
#include <SFML/Graphics.hpp>
#include <string>
#include "constants.h"
using namespace std::literals;
// The main function for the program
int main() {
  // Create the game's window using an object of class RenderWindow
  // The constructor takes an SFML 2D vector with the window dimensions
  // and an std::string with the window title
  // The SFML code is in the sf namespace
  sf::RenderWindow game_window{{constants::window_width, constants::window_height},
			       "Simple Breakout Game Version 1"s,};
  // Limit the framerate
  // This allows other processes to run and reduces power consumption
  game_window.setFramerateLimit(60);      // Max rate is 60 frames per second
  // Game loop
  // Clear the screen
  // Check for new events
  // Calculate the updated graphics
  // Display the updated graphics
  while (game_window.isOpen()) {
    // Clear the screen
    game_window.clear(sf::Color::Black);
    // Check for any events since the last loop iteration
    sf::Event event;
    // If the user pressed "Escape", or clicked on "close", we close the window
    // This will terminate the program
    while (game_window.pollEvent(event)) {
	if (event.type == sf::Event::Closed)
	  game_window.close();
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape))
      game_window.close();
    // Calculate the updated graphics
    // This space left intentionally blank!
    // Display the updated graphics
    game_window.display();
  }
}
```

### 211. Random Walk Revisited
- sf::Vector2
  - 2D vector
  - Member x and y
  - Supports == and !=
  - Template class
- sf:Vector2f
  - Alias for `sf::Vector2<float>` instantiation
  - SFML uses float internally
- Texture
  - sf::Texture represents an image
  - sf::Shape represents a pre-defined image
    - An abstract base class
  - sf::CircleShape: derived from sf::Shape
```cpp
#include <SFML/Graphics.hpp>
#include <string>
#include <random>
#include "constants.h"
using namespace std;
using namespace std::literals;
// Class to represent the creature moving around the screen
class creature {
    // Static random number engine and Bernoulli distribution objects
    static mt19937 mt;
    static bernoulli_distribution bd;
    // Define some properties of the creature
    // SFML uses float internally
    float vx{4.0f};
    float vy{4.0f};
    sf::Vector2f velocity;
    // We use the SFML CircleShape
    sf::CircleShape circle;
public:
    // Interface of the class
    // Constructor
    // Arguments are the initial coordinates of the centre of the ball
    // SFML uses the computer graphics convention
    // (0, 0) is the top left corner of the screen
    // x increases to the right
    // y increases downwards
    creature(float x, float y) {
        // Set the initial position and velocity
        // Use (x, y) for the initial position
        velocity = {vx, vy};
        circle.setPosition(x, y);
        // Set the graphical properties
        circle.setRadius(5.0f);
        circle.setFillColor(sf::Color::Red);
    }
    void draw(sf::RenderWindow& window) {
        // Ask the window to draw the shape for us
        window.draw(circle);
    }
    // Compute the creature's new position
    void update() {
        // Change the direction of movement at random
        vx = bd(mt) ? vx : -vx;
        vy = bd(mt) ? vy : -vy;
        // Move the creature to its new position
        circle.move({vx, vy});
    }
};
mt19937 creature::mt;
bernoulli_distribution creature::bd;
// The main function for the program
int main() {
    // Create a creature object in the middle of the screen
    creature the_creature(constants::window_width/2.0,
        constants::window_height/2.0);

    // Create the game's window using an object of class RenderWindow
    // The constructor takes an SFML 2D vector with the window dimensions
    // and an std::string with the window title
    // The SFML code is in the sf namespace
    sf::RenderWindow window{{constants::window_width, constants::window_height},
        "Random Walk"s};
    // Limit the framerate
    // This allows other processes to run and reduces power consumption
    window.setFramerateLimit(60);      // Max rate is 60 frames per second
    // Game loop
    // Clear the screen
    // Check for user input
    // Calculate the updated graphics
    // Display the updated graphics
    while (true) {
        // Clear the screen
        window.clear(sf::Color::Black);
        // Check for any events
        sf::Event event;
        // If the user pressed "Escape", or clicked on "close", we close the window
        // This will terminate the program
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
                break;
            }
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) {
            window.close();
            break;
        }
        // Calculate the updated graphics
        the_creature.update();
        // Display the updated graphics
        the_creature.draw(window);
        window.display();
    }
}
```

### 212. Sprite
- sf::Sprite represents a texture associated with a rectangle
  - A group of pixels as a single graphical entity
  - This entity floats over th display
- Loading an image into the graphics card
  - sf::Texture texture;
  - texture.loadFromFile("aaa.jpg");
- Use setTexture to associate this texture with a sprite
  - sf::Sprite sprite;
  - sprite.setTexture(texture);
- entity class
  - Our game will require several sprites
  - An abstract base class
- background.h:
```cpp
#ifndef BACKGROUND_H
#define BACKGROUND_H
#include "constants.h"
#include "entity.h"
// Class to represent the background
// Inherits from entity
class background : public entity {
  // Private data members
  static sf::Texture texture;
 public:
  // Interface of the class
  // Constructor
  // Arguments are the initial coordinates of the centre of the background
  // SFML uses the computer graphics convention
  // (0, 0) is the top left corner of the screen
  // x increases to the right
  // y increases downwards
  background(float x, float y);
  // Implement the pure virtual functions
  void update() override;
  void draw(sf::RenderWindow& window) override;
};
#endif // BACKGROUND_H
```
- background.cc
```cpp
#include "background.h"
// Initialize static data
sf::Texture background::texture;
background::background(float x, float y) : entity() {
  // Load the texture
  texture.loadFromFile("background.jpg");
  sprite.setTexture(texture);
  // Set the initial position and velocity of the background
  // Use (x, y) for the initial position of the background
  sprite.setPosition(x, y);
}
// Compute the background's new position
void background::update() {
  // Nothing to do here
}
void background::draw(sf::RenderWindow& window) {
  // Ask the window to draw the sprite for us
  window.draw(sprite);
```

### 213. Ball
- Implement as a sprite which can move b/w updates
- moving entity
  - Inherits from entity
  - Derives ball class
- ball.h:
```cpp
#ifndef BALL_H
#define BALL_H
#include "constants.h"
#include "entity.h"
// Class to represent the bouncing ball
// Inherits from moving_entity
class ball : public moving_entity {
  // Private data members
  static sf::Texture texture;
 public:
  // Interface of the class
  // Constructor
  // Arguments are the initial coordinates of the centre of the ball
  // SFML uses the computer graphics convention
  // (0, 0) is the top left corner of the screen
  // x increases to the right
  // y increases downwards
  ball(float x, float y);
  // Implement the pure virtual functions
  void update() override;
  void draw(sf::RenderWindow& window) override;
};
#endif // BALL_H
```
- ball.cc:
```cpp
#include "ball.h"
// Initialize static data
sf::Texture ball::texture;
ball::ball(float x, float y) : moving_entity() {
  // Load the texture
  texture.loadFromFile("ball.png");
  sprite.setTexture(texture);
  // Set the initial position and velocity of the ball
  // Use (x, y) for the initial position of the ball
  sprite.setPosition(x, y);
  velocity = {constants::ball_speed, constants::ball_speed};
}
// Compute the ball's new position
void ball::update() {
  // Move the position of the ball
  sprite.move(velocity);
}
void ball::draw(sf::RenderWindow& window) {
  // Ask the window to draw the sprite for us
  window.draw(sprite);
}
```

### 214. Bouncing Ball
- Getter function to return
  - The current position of the ball: getPosition()
  - Its center:  get_centre()
  - Its bounding rectangle: get_bounding_box()

### 215. Paddle

### 216. Moving Paddle
- When a user presses left arrow key, it moves to the left
- Pressing right arrow key, it moves to the right
- paddle::update()
  - Check for key press
  - Change velocity.x
- paddle.h
```cpp
#ifndef PADDLE_H
#define PADDLE_H
#include "constants.h"
#include "entity.h"
// A class to represent a paddle
// Inherits from moving_entity
// The paddle moves across the bottom of the screen, under the user's control
// When the ball hits the paddle, it will bounce and change its direction
class paddle : public moving_entity {
  // Private data and member functions
  static sf::Texture texture;
  // Respond to input from the player
  void process_player_input();
 public:
  // Interface of the class
  // Constructor
  // Arguments are the initial coordinates of the centre of the paddle
  paddle(float x, float y);
  // Implement the pure virtual functions
  void update() override;
  void draw(sf::RenderWindow& window) override;
};
#endif // PADDLE_H
```
- paddle.cc
```cpp
#include "paddle.h"
// Initialize static data
sf::Texture paddle::texture;
paddle::paddle(float x, float y) : moving_entity() {
  // Load the texture
  texture.loadFromFile("paddle.png");
  sprite.setTexture(texture);
  // Set the initial position of the paddle
  sprite.setPosition(x, y);
  // Set the velocity of the paddle
  velocity = {constants::paddle_speed, 0.0f};    // The paddle can only move sideways
  // By default, operations are relative to the sprite's top lh corner
  // Make them relative to the sprite's centre
  sprite.setOrigin(get_centre());
}
// Compute the paddle's new position
void paddle::update() {
   // Respond to user input as this will affect how the paddle moves
  process_player_input();
  // Move the position of the paddle
  sprite.move(velocity);
}
void paddle::draw(sf::RenderWindow& window) {
  // Ask the window to draw the shape for us
  window.draw(sprite);
}
// Respond to input from the player
// If the player presses the left arrow key, move to the left (negative velocity)
// If the player presses the right arrow key, move to the right (positive velocity)
// Otherwise, do not move (zero velocity)
// Do not allow the paddle to move off the screen
void paddle::process_player_input() {
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left)) {
    // Left arrow key pressed - move to the left
    // Unless the paddle has gone past the left hand side
    if (x() >= 0)
      velocity.x = -constants::paddle_speed;
    else
      velocity.x = 0;
  }
  else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right)) {
    // Similarly for the right arrow
    if (x() <= constants::window_width)
      velocity.x = constants::paddle_speed;
    else
      velocity.x = 0;
  }
  else {
    // Some other key pressed, or no key at all
    velocity.x = 0;
  }
}
```

### 217. Ball-Paddle Interaction
- Detects and resovles collisions b/w entities
- `void handle_collisio(ball &b, const apddle&p);`
- `bool is_interacting(const entity& e1, const entity& e2);`

### 218. Bricks
- Implemented as a vector of sprites

### 219. Ball Interaction with Bricks
- When the ball hits a brick
  - The brick is destroyed
    - Removed from a vector
  - The ball bounces off
- handle_collision()
  - Collision can occur on all four sides

### 220. Game Manager
- We need to refactor the code
  - main() is too long
  - Duplicated code
- We move the code from main() to a new class
- game class
  - Data member
    - Game window
    - The ball
    - The background
    - The paddle
    - The vector of blocks
  - Member function
    - Constructor to intialize
    - run() to perform the game loop
    - reset() to re-initialize the entities
- Pausing the game
  - A class enum manages the state

### 221. Entity Manager Overview
- Refactoring the entity management into a separate class
  - Simplifies the game loop
  - Provides tbetter abstraction
- entity_manager
  - Will use polymorphism
    - Store entities as pointers to base class
    - Call virtual function through pointer to base
  - Will provide an interface for performing operations on entities
    - creat()
    - refresh()
    - clear()
    - get_all()
    - apply_all()
    - update()
    - draw()
  - Will create all entities in the game
  - all_entities
    - Stores all the entities in the game as an std::vector of std::unique_ptr<entity>
  - grouped_entities
    - Sotres all the entities in the game as an std::map
    - The key depends on the entity type

### 222. Entity Manager and Object Creation
- entity_manager::create()
  - Creates a new entity object
  - Adds it to all_entities
  - Adds an alias to the object to grouped_entities
  - Will use a variadic template
  - Perfect forwarding for efficiency

### 223. Entity Manager and Object Operations
- entity_manager::get_all()
  - Returns all the entities of a given type
  - Template for class type
- entity_manager::apply_all()
  - Applies a function to all entities of a given type

### 224. Brick Strength
- Requires several impacts before the brick is destroyed
- Add a strength member to the brick class
- sf::Color
  - red
  - green
  - blue
  - alpha: opacity. 0 is a completely transparent and 255 is opaque color. Default is 255
  - They are 8-bit unsigned intgers
  - sf::Color(255,0,0) => red

### 225. More Features
- Player lives
- Text in SFML
  - sf::Text represents graphical text
  - Associated wtih sf::Font

### 226. Conclusion
- A game code in 670 lines
  - Modular
  - Easy to understand except entity_manager
- Adding a new entity type
  - Derive a class from entity or moving_entity
  - Load its image file in the constructor
  - Override update()
  - Add data members and member functions
  - Create and register objects in game_manager::reset()
  - Add collision handling code to game loop if required
- How to improve?
  - Code refactoring
  - Randomize brick strengths
  - Sound effects

## Section 16: Resources

### 227. Recommended Books
- The recommended book for this course is "C++ Primer (5th Edition)" by Lippman, Lajoie and Moo. Thorough coverage of C++11, plus traditional features which are still widely used.
- Also worth reading is "A Tour of C++ (2nd Edition)" by Bjarne Stroustrup which gives a brief description of every important feature in Modern C++ up to C++17.

### 228. C++ "Cheat Sheet" Infographics
- A very useful resource for quickly looking up how to do things in C++. Contains some C++17 and C++20 features not covered in this course. https://hackingcpp.com/cpp/cheat_sheets.html

### 229. The "Awesome C++ Frameworks and Libraries" Github
- The "Awesome C++ Frameworks and Libraries" github is a massive collection of links to C++ libraries and frameworks: https://github.com/fffaraz/awesome-cpp

### 230. The "Awesome Modern C++ Resources" Github
- The "Awesome Modern C++ Resources" Github is a massive collection of links to C++ books, talks, blogs etc etc
- https://github.com/rigtorp/awesome-modern-cpp

### 231. "Classy Header-only Classes"
- A cross-platform collection of useful classes which fill many gaps in the C++ standard library. Header-only, so you just drop them into your project* without further ado!
- https://github.com/Tracktion/choc
- * except for HTTP and WebSocket Server class, which requires linking against boost::beast

### 232. Bonus Material
