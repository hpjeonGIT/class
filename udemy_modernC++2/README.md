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
- Some texts inside of quotes "..."
- C-style string literal
  - An array of const char, terminated by a null character
  - Very limited range of operations
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
### 54. Default and Delete Keywords
### 55. Operators and Overloading.
### 56. Which Operators to Overload
### 57. The Friend Keyword
### 58. Member and Non-member Operators
### 59. Addition Operators
### 60. Equality and Inequality Operators
### 61. Less-than Operator
### 62. Prefix and Postfix Operators
### 63. Function Call Operator
### 64. Printing Out Class Member Data

## Section 6: Algorithms Introduction and Lambda Expressions

    Algorithms Overview
    07:29
    Algorithms with Predicates
    05:32
    Algorithms with _if Versions
    05:49
    Lambda Expressions Introduction
    05:29
    Algorithm with Lambda Expression

1 question
Lambda Expressions Practical
03:45
Lambda Expressions and Capture
07:29
Lambda Expressions and Capture Continued
09:51
Mutable Lambda
3 questions
Lambda Expressions and Partial Evaluation
06:55
Lambda Expressions in C++14
06:06
Generalized capture with initialization

    4 questions
    Pair Type
    06:07
    Insert Iterators
    07:16
    Library Function Objects
    03:24

    Searching Algorithms
    06:14
    Searching Algorithms Continued
    05:16
    Numeric Algorithms
    06:18
    Write-only Algorithms
    07:57
    for_each Algorithm
    03:17
    Copying Algorithms
    03:03
    Write Algorithms
    05:09
    Removing Algorithms
    04:35
    Removing Algorithms Continued
    06:32
    Transform Algorithm
    06:19
    Merging Algorithms
    03:53
    Reordering Algorithms
    05:34
    Partitioning Algorithms
    04:21
    Sorting Algorithms
    03:39
    Sorting Algorithms Continued
    06:21
    Permutation Algorithms
    04:01
    Min and Max Algorithms
    03:10
    Further Numeric Algorithms
    04:32
    Further Numeric Algorithms Continued
    06:30
    Introduction to Random Numbers
    03:48
    Random Numbers in Older C++
    04:37
    Random Numbers in Modern C++
    06:57
    Random Number Algorithms
    03:26
    Palindrome Checker Practical
    07:16
    Random Walk Practical
    07:41
    Algorithms and Iterators Workshop

    1 question

    Container Introduction

02:49
Standard Library Array
04:47
Forward List
05:23
List
05:16
List Operations
05:49
Deque
05:13
Sequential Containers
4 questions
Sequential Containers Part Two
2 questions
Tree Data Structure
04:34
Sets
07:56
Map
09:40
Maps
4 questions
Maps and Insertion
04:02
Maps in C++17
07:49
Multiset and Multimap
04:29
Searching Multimaps
08:07
Unordered Associative Containers
07:22
Unordered Associative Containers Continued
04:04
Associative Containers and Custom Types
09:35
Nested Maps
05:15
Queues
06:06
Priority Queues
05:31
Stack
05:02
Emplacement
07:10
Mastermind Game Practical

    11:06
    Containers Workshop
    00:06

    Class Hierarchies and Inheritance

02:40
Base and Derived Classes
05:23
Member Functions and Inheritance
04:41
Overloading Member Functions
03:10
Pointers, References and Inheritance
05:55
Static and Dynamic Type
04:08
Virtual Functions
04:54
Virtual Functions in C++11
05:23
Virtual Functions

    5 questions
    Virtual Destructor
    06:15
    Interfaces and Virtual Functions
    08:03
    Virtual Function Implementation
    03:20
    Polymorphism
    06:00

    Error Handling
    03:39
    Error codes and Exceptions
    06:47
    Exceptions Introduction
    06:10
    Try and Catch Blocks

    06:53
    Catch-all Handlers
    05:53
    Exception Mechanism
    05:30
    std::exception Hierarchy
    05:59
    Standard Exception Subclasses
    05:16
    Exceptions and Special Member Functions
    04:09
    Custom Exception Class
    05:53
    Exception Safety
    03:27
    The throw() Exception Specifier
    03:32
    The noexcept keyword
    04:33
    Swap Function
    04:36
    Exception-safe Class
    04:26
    Copy and Swap
    05:15
    Comparison with Java and C# Exceptions
    05:26

    Move Semantics
    05:13
    Lvalues and Rvalues

    06:05
    Lvalue and Rvalue References
    08:15
    Value Categories
    03:11
    Move Operators
    09:08
    RAII Class with Move Operators
    07:21
    Move-only Types and RAII
    07:38
    Special Member Functions in C++11
    04:02
    Using Special Member Functions in C++11
    05:47
    Function Arguments and Move Semantics
    07:00
    Forwarding References
    08:00
    Perfect Forwarding
    08:28
    Perfect Forwarding Practical
    03:17

    Smart Pointers Introduction
    05:58
    Unique Pointer
    08:17
    Unique Pointers and Polymorphism
    07:02
    Unique Pointers and Custom Deleters
    05:49
    The Handle-Body Pattern
    06:08
    The pImpl Idiom
    06:33
    Reference Counting
    10:59
    Shared pointer
    06:43
    Weak Pointer
    08:24
    Weak Pointer and Cycle Prevention
    03:22

    Chrono Library Introduction
    02:51
    Chrono Duration Types
    05:26
    Chrono Clocks and Time Points
    06:25
    Bitsets
    06:14
    Tuples
    06:21
    Tuples in C++17
    04:22
    Unions

05:37
Unions Continued
06:57
Mathematical Types
06:40
Bind
07:33
Callable Objects
05:45
Member Function Pointers
06:38
Interfacing to C
10:50
Interfacing to C

    2 questions
    Run-time Type Information
    07:23
    Multiple Inheritance
    06:14
    Virtual Inheritance
    04:53
    Inline Namespaces
    05:58
    Attributes
    07:13

    Compile-time Programming Overview
    08:01
    Constant Expressions
    04:39
    Constexpr Functions
    06:49
    Classes and Templates
    05:32
    Template Specialization
    08:13
    Extern Templates
    09:18
    Variadic Templates
    09:33
    Miscellaneous Template Features
    05:07
    Library-defined Operators
    05:26
    Constexpr If Statement
    09:52
    Constexpr If Examples
    04:36
    The decltype Keyword
    09:03

    Project Breakout

01:28
SFML Introduction
03:07
Compiler Configuration for SFML
05:17
Basic Window
04:16
Random Walk Revisited
05:46
Sprite
06:30
Ball
04:09
Bouncing Ball
04:18
Paddle
03:03
Moving Paddle
04:26
Ball-Paddle Interaction
04:31
Bricks
04:48
Ball Interaction with Bricks
08:08
Game Manager
07:13
Entity Manager Overview

    05:31
    Entity Manager and Object Creation
    08:22
    Entity Manager and Object Operations
    08:08
    Brick Strength
    06:34
    More Features
    08:11
    Conclusion
    04:23

    Recommended Books
    00:14
    C++ "Cheat Sheet" Infographics
    00:07
    The "Awesome C++ Frameworks and Libraries" Github
    00:05
    The "Awesome Modern C++ Resources" Github
    00:05
    "Classy Header-only Classes"
    00:10
    Bonus Material
