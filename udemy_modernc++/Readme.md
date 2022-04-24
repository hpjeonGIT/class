## Complete Modern C++ (C++11/14/17)
- Instructor: Umar Lone

33. Inline functions
- Replacement of MACRO in C++11 (?)
- Injects inline function to the code, instead of calling the function
- Some functions will not be inlined
    - Large ones
    - Functions with too many conditional statements
    - Recursive functions
    - Functions invoked through pointers
- Modern compiler may inline non-inlined functions
- Macro vs inline
    - Macro
        - works through text substitution
        - error prone due to substitution nature
        - does not have an address
        - diffult to use with many lines
        - cannot be used for member functions of a class
    - Inline function
        - the call is replaced with the body
        - safe to use as it has function semantics
        - has an address
        - multiple lines can be used
        - class member functions can be used

34. Function pointers
- Pointer that holds the address of the function
- Can be used to indirectly invoke the function even if the function name is not known
- The type is same as the signature of the function (return type & arguments)
```cpp
// <ret> (*fnptr)(args) = &Function
int (*PtrAdd)(int,int) = &Add; // int Add(int,int)
int (*PtrAdd)(int,int) = Add; // & is optional
(*PtrAdd)(3,4); // valid expression
PtrAdd(3,4); // valid as well
```
- Sample code:
```cpp
#include <iostream>
void Prnt(int n, char c) {
    for (int i=0;i<n;i++) std::cout << c;
}
int main() {
    void (*fPnt)(int,char) = Prnt;
    (*fPnt)(3,'#');
    fPnt(5,'!');
    return 0;
}
```

35. Namespace
```cpp
#include<iostream>
namespace Avg {
    float calc(float x, float y) { return (x+y)/2.f;}
}
namespace Add {
    float calc(float x, float y) { return (x+y);}
}
int main(){
    float x=1.0f, y=2.3f;
    std::cout << Avg::calc(x,y) << std::endl;
    std::cout << Add::calc(x,y) << std::endl;
}
```




## Section 4: classes and objects

49. Copy constructor - part I
- Creates copy of the object's state in another object
- Compiler may provide one automatically if there no copy constructor
    - This could be problematic if a class has pointer member data
- Default copy constructor will do shallow copy
    - Use `delete` to disable copy-constructor
    - If the code uses copy constructor, the compiler will produce error message, preventing the use of copy constructor
    - `default` means that copy constructor will use what the compiler provides
- Regarding initialization of objects using ()
```
Myclass x;
...
Myclass z(x); <-- This will try to shallow copy
```
- Disabling constructor:
```
class Myclass {
    ...
    Myclass(const &Myclass) = delete;
    ...
}
```
- If pointer or deep-copy is necessary, provide copy-constructor manually
- If any function returns a class object, it is returned as a value (not reference). Then this will activate copy constructor of that class

50. Copy constructor - part II
- Operator `=` will do the shallow copy as well
    - This is an **assignment operator**
```
Myclass x;
...
Myclass z;
z = x; <--------  shallow copy will be done
```
- **Rule of 3**
    - All of followings must be defined if a user implements any of them
    - Destructor: `~Integer()`
    - Copy constructor: `Integer(const Integer & obj)`
    - Copy assignment operator: `Integer& operator= (A obj)`
- When a copy constructor is implemented, the argument must use address, not value, as it will recursively call another constructor
    - `Integer(const Integer & obj) {...}`

53. L-value, R-value, & Rvalue references 
- L-value
    - Has a name
    - All variables are l-values
    - Can be assigned valueds
    - L-value persists beyond the expression
    - `x` in `x = 1` is l-value
    - `++x` returns l-value
    - Return by reference functions return l-value
- R-value
    - Does not have a name
    - R-value is a temporary value
    - Cannot be assigned values
    - R-value does not persist beyond the expression
    - `(x+y)` returns r-value
    - Return by value functions return r-value
- R-value references
    - Instroduced in C++11
    - A reference created to a temporary
    - Represents a temporary
    - Use `&&` operator
    - cannot point to l-values
    - R-value references always bind to temporaries
    - L-value references always bind to l-values
```
int &&r1 = 10; // R-value reference
int &&r2 = Add(5,8); // Add returns by value (temporary)
int &&r3 = 7+2; // expression returns a temporary value
```
- R-value reference may be used to detect temporary values
- `void Prnt(int x)` can handle Prnt(x) and Prnt(7)
- `void Prnt(int &x)` can handle Prnt(x) but NOT Prnt(7)
- `void Prnt(const int &x)` can handle Prnt(x) and Prnt(7)
- `void Prnt(int &&x)` or `void Prnt(const int &&x)` can handle Prnt(7) but NOT Prnt(x)
```cpp 
#include<iostream>
/*
void Prnt(int x) { // this is ambiguous and will not compile
    std::cout << "Prnt " << x << std::endl;
}
*/
void Prnt(const int& x ){ // handles l-value
    std::cout << "Prnt const int &x " << x << std::endl;
}
void Prnt(const int &&x) { // handles r-value
    std::cout << "Prnt const int&&x" << x << std::endl;
}
int main(){
    int x = 1;
    Prnt(x);
    Prnt(1);
}
```

54. Move semantics
- For an object obj1, which points memory m1
    - Deep copy will make obj2, which points memory m2. m2 is copied from m1
        - Very slow
    - Shallow copy will make obj2, which points memory m1. When obj1 is deleted, obj2 points a wrong location
    - So move will make obj2, which points memory m1 while makes obj1 point null memory
        - This has to be done through class/member function definition

56. Rule of 5 & 0
- Some class owns a resource
- Those resources may be acquired in the constructor
- When copy/move/destroy, users must decide what to do
    - When destroyed, the resource must be released
    - When copy/move, the resource must be handled as well
- Rule of 5
    - If a class has ownership semantics, you must provide a user-defined:
    - destructor
    - copy constructor: may need to allocate new memory
    - copy assignment operator: may need to allocate new memory
    - move constructor: may need to move the resource
    - move assignment operator: may need to move the resource
```Cpp
class Integer {
...    
Integer(); //default constructor
Integer(int value); // parameterized constructor
Integer(const Integer &obj); // copy constructor
Integer(Integer &&obj); // move constructor
Integer & operator=(const Integer &obj); // copy assignment
Integer & operator=(Integer && obj); // move assignment
```
- Sample code:
```cpp
#include <iostream>
class Integer{
private:
    int x;
public:
    Integer() {	x = 0;std::cout << "default constructor\n";}
    ~Integer() {std::cout << "default destructor\n";}
    Integer(int value) { x = value; std::cout << "param constructor\n";}
    Integer(const Integer &obj) {x = obj.x; std::cout << "copy constructor \n";}
    //Integer(const Integer &obj) = default;
    Integer(Integer &&obj) {x = obj.x; std::cout << "move constructor \n";}
    Integer &operator=(const Integer &obj) {x = obj.x;
	std::cout << "copy assign operator\n";}
    Integer &operator=(Integer && obj) {x = obj.x;
	std::cout << "move assign operator\n";}
    int GetValue() {return x;}
    void SetValue(int value) { x = value;}
};

Integer Add(const int a, const int b) {
    return Integer(a+b); // this calls param constructor
}
int main() {
    Integer i1(123), i2(i1), i3;
    // i1(123) calls param constructor
    // i2(i1) calls copy constructor
    // i3 calls default constructor    
    i3 = i1; // this calls copy assign operator
    Integer i4 {i1}; // this calls copy constructor
    i3 = Add(1,2); //When Add() returns, it calls move assign operator
    Integer i5(std::move(i3)); // default/move constructors are called
    return 0;
}
```
- If copy constructors are commented out, the compile will fail
    - Rule of 5
    - The above constructors are same as `Integer(const Integer &obj) = default;` as there is no pointer member data
- Rule of 0
    - If a class doesn't have ownership semantics, do not provide any of Rule of 5
    - Compiler will decide
        - User defined function may conflict
- Custom copy constructor will delete the default move constructor and operator
    - ? 
    - To use move constructor/operator, the custom move constructor/operator must be made
- parameterized constructor CANNOT use default/delete

57. Copy elision
- `intClass a = 3` can be recognized as `intClass a = intClass(3)`
    - This will call move constructor
    - Intermediate constructor will not be used
    - **Return value optimization**

58. std::move
- std::move(x): a syntactic sugar of `static_cast<Myclass&&>(x)`

## Section 6: Operator Overloading

60. Operator overloading - part I
- Custom implementation for primitive operators
- Allows usage of user-defined objects in mathematical expressions
- Overloaded as functions but provides a convenient notation
- Implemented as member or global functions
- Usage: `<ret> operator <#> (<arguments>)`
    - `Integer &operator=(const Integer &obj) {}`
- Number of arguments
    - For global functions, same no. of arguments as the operands is required
    - For member functions,
        - Binary operator will require only one explicit argument
        - Unary operator will not require any argument
- Operator overloading is a **syntatic sugar** of functions
- postfix operator needs `int` argument type
    - Ref: https://docs.microsoft.com/en-us/cpp/cpp/increment-and-decrement-operator-overloading-cpp?view=msvc-170
    - `Integer &operator++();` : prefix increment like ++obj
    - `Integer operator++(int);` : postfix increment like obj++
        - postfix increment requires a temporary object and this is why prefix increment is more efficient
        - Ref: https://stackoverflow.com/questions/24901/is-there-a-performance-difference-between-i-and-i-in-c

61. Operator overloading - part II
- For `a=a`, address checking mechanism is necessary for `operator=`

62. Operator overloading - part III
- standard-out: `std::cout << a;`
```cpp
std::ostream & operator << (std::ostream &out, const Integer &a) {
    out << a.GetValue();
    return out;
}
```
- standard-in: `std::cin >> a;`
```cpp
std::istream & operator >> (std::istream &inp, Integer &a) {
    int x;
    inp >> x;
    a.SetValue(x);
    return inp
}
- Function call operator: `a();`
```cpp
void operator()() { ... }
```

63. Operator overloading - part IV
- Use `friend` to access member data (?)

64. Operator overloading - part V
- Overloading `->` and `*`
- May add a class with destructor
    - Allows automatic memory deallocation when out-of scope
    - This is how **smart pointer** works

65. Operator overloading - part VI

66. Operator overloading - part VII

67. Type conversion - part I
- C-style casting is not recommended : `(float) a`
    - Type check is not done
- `static_cast<float> (a)`
- `reinterpret_cast` might be done b/w different types (int->char)
- `const_cast` ?

68. Type conversion - part II
- Compiler may convert initialization parameter implicitly
    - `Integer i1 = 3;`
    - To avoid such implicit conversion, use `explicit` keyword for the parameterized constructor
        - `explicit Integer(int value) { ... }`

69. Type conversion - part III
- Type conversion operator
    - `operator <type> ()`
        - `operator int() { return x; }`
    - No argument/No return type
    - Now `int x = i6;` is allowed
- To avoid implicit conversion, may use `explicit` keyword
    - `explicit operator int() { return x; }`
    - Now `int x = static_cast<int>(i6);` is allowed

70. Initialization vs. assignment & member initialization list
- Initialization list (calls parameterized constructor) is preferred than assignment as assignment has more function calls (default constructor + assignment operator)

71. Raw pointers
- Q: factory function?
- Some good practice
```cpp
Integer *p = new Integer{value};
...
delete p; // deallocate memory
p = nulltpr; // clean the addres it is pointing
```
- Raw pointer work OK but smart pointer is recommended as it will prevent memory leak

72. std::unique_ptr
- Cannot be copied
- To send the unique pointer as an argument, use std::move
```cpp
void fnc1(std::unique_ptr<Integer> p) {...}
...
    fnc1(std::move(p));// after func1(p), p is deallocated
```
    - After fnc1(), p will be deallocated. To use again, reset()
- If the function argument is used as reference, the pointer still holds the memory after function
```cpp
void fnc2(std::unique_ptr<Integer> &p) {...}
...
    fnc2(p);// after func2(p), p is still allocated
    std::cout << p->GetValue() << std::endl;
```
- Unique pointer member function: release(), reset(), swap(), get(), ...
- Adding GetPointer() function as a member function
```cpp
class Integer{
...
    Integer *GetPointer() {return this;}
};
int main() {
    Integer *i1 = new Integer{123};
    std::shared_ptr<Integer> p{i1->GetPointer()};
    std::cout << p->GetValue() << std::endl;
    return 0;
}
```

73. Sharing Pointers
- When multipe pointers share a pointer from a different class
- Need to deallocate one by one in the end

74. Sharing std::unique_ptr

75. std::shared_ptr
- Increments as it is shared again
```cpp
class Employee{
    std::shared_ptr<Integer> m_pInteger{};
public:
    void SetInteger(std::shared_ptr<Integer> &i1) {
        m_pInteger = i1;
    }
    const std::shared_ptr<Integer>& GetInteger() const {
        return m_pInteger;
    }
};
int main() {
    std::shared_ptr<Integer> p{GetPointer(3)};
    p->SetValue(123); std::cout << p.use_count() << std::endl; //-> will be 1
    std::shared_ptr<Employee> e1 {new Employee{}};
    e1->SetInteger(p); std::cout << p.use_count() << std::endl; //-> will be 2
    std::shared_ptr<Employee> e2 {new Employee{}};
    e2->SetInteger(p); std::cout << p.use_count() << std::endl; //-> will be 3
    return 0;
}
```
- To remove shared ptr, use `p=nullptr;` or `p.reset()`

76. Weak ownership
- When using shared_ptr:
```cpp
class Employee{
    ...
    void SetInteger(std::shared_ptr<Integer> &i1) {
        m_pInteger = i1;
    }
    void PrintStatus() {
        std::cout << "At employee " <<m_pInteger.use_count() << std::endl;
    }
};
int main() {
    ...
    e1->SetInteger(p);
    ...
    e2->SetInteger(p);
    ...
    p = nullptr; std::cout << p.use_count() << std::endl; //-> prints 0
    e1->PrintStatus(); e2->PrintStatus(); // -> Still prints 2
    return 0;
}
```
    - Even though the memory of `p` at main is deallocated, the memory within Employee objects are not deallocated until the end

77. std::weak_ptr internals
- A weak pointer must be defined from a shared pointer
    - It referes the control object (count number) of the shared pointer
    - We check if the weak pointer is expired or not
    - When valid, use `.lock()` to retun the shared pointer
```cpp
class Employee{
    std::weak_ptr<Integer> m_pInteger{};
public:
    void SetInteger(std::weak_ptr<Integer> i1) {
        m_pInteger = i1;
    }
    const std::weak_ptr<Integer>& GetInteger() const {
        return m_pInteger;
    }
    void PrintStatus() {
        if (m_pInteger.expired()) {
            std::cout << " no longer available \n"; return;
        }
        std::cout << "At employee " <<m_pInteger.lock().use_count() << std::endl;
    }
};
int main() {
    std::shared_ptr<Integer> p{GetPointer(3)};
    p->SetValue(123); std::cout << p.use_count() << std::endl; // prints 1
    std::shared_ptr<Employee> e1 {new Employee{}};
    e1->SetInteger(p); std::cout << p.use_count() << std::endl; // prints 1
    e1->PrintStatus(); // prints 2
    std::shared_ptr<Employee> e2 {new Employee{}};
    e2->SetInteger(p); std::cout << p.use_count() << std::endl; // prints 1
    p->SetValue(456);
    e2->PrintStatus(); // prints 2
    p = nullptr; std::cout << p.use_count() << std::endl;
    e1->PrintStatus(); e2->PrintStatus(); // no longer available
    return 0;
}
```

78. Circular References
- If shared_ptr uses circular reference of classes, destructor may not work
    - Using weak poiners could be a solution
    - Not necessary to replace all shared_ptr into weak_ptr. Only one side may need weak_ptr

79. Deleter

80. Dynamic arrays
- std::unique_ptr can handle dynamic arrays but STL is recommended

81. Make functions
- `auto p = std::make_unique<Integer> (3);` is equivalent to `std::unique_ptr<Integer> p{GetPointer(3)};`
- `auto e1 = std::make_shared<Employee>();` is equivalent to `std::shared_ptr<Employee> e1 {new Employee{}};`

## Section 8: More C++ goodies

82. Enums - Part I
- Enumerator is converted into unsized integer but integer is not convereted into Enumerator
- Start from zero

83. Enums - Part II (Scoped Enums since C++11)
- `enum Color {RED, GREEN, BLUE};`: 0,1,2
- `enum Color {RED=4, GREEN, BLUE};`: 4,5,6
- `enum Color : char {RED='c', GREEN, BLUE};`: ASCII number of c,d,e
- Scoped Enums:
```cpp
enum class Color(RED, GREEN, BLUE);
...
Color c = Color::RED;
```
- But scoped Enums needs static_cast<int>() to std::cout

86. Assignment 1
```cpp
#include <iostream>
#include <string>
#include <cstdio>
std::string ToUpper(const std::string &str) {
  std::string upper;
  for (const auto &el: str) {
    upper += toupper(el);
  }
  return upper;
}
std::string ToLower(const std::string &str) {
  std::string upper;
  for (const auto &el: str) {
    upper += tolower(el);
  }
  return upper;
}
int main(){
  std::string s0 = "Hello World";
  std::cout << ToUpper(s0) << ToLower(s0) << std::endl;
}
```

87. String streams
- std::stringstream: read/write
- std::istringstream: read only
- std::ostringstream: write only

88. Assignment 2
```cpp
#include <iostream>
#include <string>
enum class Case{SENSITIVE, INSENSITIVE};
size_t Find( const std::string &src,
             const std::string &search,
             Case scase = Case::INSENSITIVE,
             size_t offset=0) {
  std::string l_src, l_search;
  std::cout << static_cast<int>(scase) << std::endl;
  if (scase == Case::INSENSITIVE) {
    for(auto &el: src) l_src += toupper(el);
    for(auto &el: search) l_search += toupper(el);
    return l_src.find(l_search, offset);
  } else {
    return src.find(search, offset);
  }
}
int main(){
  std::string s0 = "abc Hello World";
  std::cout << Find(s0, "hello") <<std::endl;
  std::cout << Find(s0, "bye") <<std::endl;
  std::cout << Find(s0, "hell0", Case::SENSITIVE) << std::endl;  
}
```
- Compare the return value with `std::string::npos` to see if the location is found or not

89. User-Defined literals
- A literal is a fixed value that appears directly in the code
    - 14u: unsized integer
    - 621l: long integer
    - 3.14f: float
    - 1'000'000: 1million since C++14
    - ref: https://en.cppreference.com/w/cpp/language/integer_literal
- User defined literals
    - `31.2_F` or `31.2_C` as Fahrenheit or Celsius
    - Needs underscore in the beginning
    - Literal operator functions cannot be member functions. Global functions only
```cpp
#include<iostream>
class Distance {
  long double m_km;
public:
  //Distance() = default;
  Distance(long double km) : m_km{km} {}
  //~Distance() = default;
  void print() {std::cout << m_km << "km" << std::endl;}
};
Distance operator"" _mi(long double v) {
  return Distance{v*1.6};
}
Distance operator"" _meter(long double v) {
  return Distance{v/1000.};
}
int main() {
  Distance d0{32.0_mi}; d0.print();
  Distance d1{1600.0_meter}; d1.print();
  return 0;
}
```
- `double` is not supported: https://stackoverflow.com/questions/16596864/c11-operator-with-double-parameter
    - `long double` works OK
    - `1600.0_meter` works but not `1600_meter`
        - Must be double
- Note `d0{32.0_mi}`, not `d0()`. This uses the initialization list
- Q: Is this really useful? What if a class requires multiple argument like `d0{10.0_mi, 3.1_km}`, can the  user-defined literal work?

90. Constant Expressions - constexpr
- Since C++11
- Represents an expression that is constant
- Can be evaluated at compile time
    - In other words, must be obvious to the compiler
- Constant exprssion function rules
    - Should accept and return literal type only: void, scalar of int, float, char, references, ...
    - Only one return is allowed at C++11
        - C++14 allows conditional multiple return
    - Implicitly inlined -> Use constexpr function in the header files only
```cpp
#include <iostream>
constexpr int GetNumber() { return 123; }
constexpr int Max(int x,int y) {
  if (x>y) {return x;}
  else {return y;}
  // return x>y ? x :y; // for C++11
}
int main() {
  constexpr int i = GetNumber(); std::cout << i << std::endl;
  constexpr int j = Max(i, 3);std::cout << j << std::endl;
  constexpr int k = Max(4, 3);std::cout << k << std::endl;
  return 0;
}
```

91. std::initializer_list(C++11)
- Enables a list input of {...} as function arguments
- Needs to use iterator to loop over the elements of the list

93. Assignment III
- Find an integer vector containing the positions of search word in the string
```cpp
#include <iostream>
#include <string>
#include <vector>
enum class Case{SENSITIVE, INSENSITIVE};
std::vector<int> FindAll(
  const std::string &src,
  const std::string &search,
  Case scase = Case::INSENSITIVE, size_t offset=0) {
  std::string l_src, l_search;
  std::vector<int> voffset;
  size_t loc = 0;
  if (scase == Case::INSENSITIVE) {
    for(auto &el: src) l_src += toupper(el);
    for(auto &el: search) l_search += toupper(el);    
  } else {
    l_src = src;
    l_search = search;    
  }
  while(true) {
    auto loc = l_src.find(l_search, offset);
    if (loc == std::string::npos) break;
    voffset.push_back(loc);
    offset = loc+l_search.size();
  }    
  return voffset;
}
int main() {
  auto x = FindAll("HelloHelloheLLo","hello");
  for (auto &el : x) std::cout << el << std::endl;
}
```

94. Union - I
- Gives the ability to represent all the members in the same memory
    - Not saving all of member data simultaneously. Only the newest is valid
- C++11 allows user-defined constructor for Unions
- Size of the storage is determined by the size of the largest member data
    - When a member data is updated, other member data will be over-ridden or contaminated

95. Union - II

## Section 9: Object oriented programming

97. Inheritance & Composition
- Composition
    - Object composed in another object
    - Represents "has-a" relation
    - Reuse behavior
- Inheritance
    - Class inherits the feature of another class
    - Reuse & inherit behavior
    - Represents "is-a" relationship

98. Inheritance & Access modifiers
- Access modifier
    - private: only accessible from the class
    - public: accessible anywhere
    - protected: accessible only to child classes
- Inheritance modifiler
    - `class Child : <modifier> Base {};`
    - private: makes all access modifiers of Base as private
    - protected: public/protected of Base -> protected. Private remains as private
- Object construction
    - Constructors execute from base to child
    - Destructors execute from child to base

99. Project - I (Introduction)
- Banking application
    - Manage accounts
    - Customer operations : withdrawal, deposit
    - Admin tasks by the bank
    - Account class
        - Member data: name, accno, balance, ...
        - Member function: deposit, withdraw, transfer, ...
        - Savings/Checking child class

100. Project - II
- Child class CANNOT access private data of the base class
    - When child class implements a constructor, it MUST use the constructor of the base class
```cpp
class Account {
private:
  std::string m_Name;  float m_Balance;
public:
  Account(const std::string &name, float balance): m_Name(name), m_Balance(balance) {    }
...
};
class Savings: public Account {
  float m_Rate;
public:
  /* Savings(const std::string &name, float balance, float rate)
      : m_Name(name), m_Balance(balance), m_Rate(rate) {
      This will not work as it cannot acces m_Name, m_Balance, m_Rate which are private in the base class
  */
  Savings(const std::string &name, float balance, float rate)
  : Account(name, balance), m_Rate(rate) { }
```

101. Assignment
- Add Checking class
```cpp
#include <iostream>
class Account {
private:
  std::string m_Name;
  int m_AccNo;
  static int s_ANGenerator;
protected:
  float m_Balance;  
public:
  //Account();
  Account(const std::string &name, float balance): m_Name{name}, m_Balance{balance} {
    m_AccNo = ++s_ANGenerator;
    }
  ~Account()=default;
  const std::string GetName() const {return m_Name;}
  float GetBalance() const {return m_Balance; }
  int GetAccountNo() const {return m_AccNo; }
  void Deposit(float balance) { m_Balance += balance; }
  void Withdraw(float balance) { m_Balance -= balance; }
};
class Savings: public Account {
  float m_Rate;
public:
  /*
  Savings(const std::string &name, float balance, float rate)
  : m_Name(name), m_Balance(balance), m_Rate(rate) { m_AccNo = ++s_ANGenerator;  }; => This doesn't work */
  Savings(const std::string &name, float balance, float rate)
  : Account{name, balance}, m_Rate{rate} {} // MUST use the constructor of the base class
  ~Savings() = default;
};
class Checking: public Account {
  static float s_CheckingLimit;
public:
  Checking(const std::string &name, float balance): Account{name, balance} {}
  ~Checking() = default;
  void Withdraw(float balance) {
    if (m_Balance-balance < s_CheckingLimit) {
      std::cout << "Reached checking Limit. Withdrawal is disabled\n";
    } else {
      m_Balance -= balance; 
    }
  }
};
int Account::s_ANGenerator = 1000; // needs int in the beginning
float Checking::s_CheckingLimit = 50.00f; 
int main() {
  Account acc01("Bob",1000.00f);
  std::cout << "Initial balance " << acc01.GetBalance() << std::endl;
  acc01.Deposit(200.00f);
  acc01.Withdraw(380.00f);
  std::cout << "Current balance " << acc01.GetBalance() << std::endl;
  Savings acc02("Bob",1000.00f, 0.01f);
  std::cout << "Initial balance " << acc02.GetBalance() << std::endl;
  acc02.Deposit(200.00f);
  acc02.Withdraw(380.00f);
  std::cout << "Current balance " << acc02.GetBalance() << std::endl;
  Checking acc03("Bob",1000.00f);
  std::cout << "Initial balance " << acc03.GetBalance() << std::endl;
  acc03.Withdraw(700.00f);
  acc03.Withdraw(300.00f);
  std::cout << "Current balance " << acc03.GetBalance() << std::endl;
  return 0;
}
```

102. Project - III
- For Withdraw() function in Checking class, we can re-use the member function of the base class
```cpp
void Withdraw(float balance) {
    if (m_Balance-balance < s_CheckingLimit) {
      std::cout << "Reached checking Limit. Withdrawal is disabled\n";
    } else {
      Account::Withdraw(balance);
    }
  }
```
- Inheriting constructor since C++11
    - In the constructor of Checking class, `Checking(const std::string &name, float balance): Account{name, balance} {}` can be replaced with `using Account::Account;`, invoking the constructor of the base class
    - Savings account cannot use this method as it has one more argument
    - `objdump -d a.out -C > de-mangle.log` can confirm that `Checking::Account` actually calls `Account::Account`
```assembly
    00000000000012ea <Checking::Account(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, float)>:
...
    1318:	e8 35 fe ff ff       	callq  1152 <Account::Account(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, float)>
```

103. Project - IV
- Virtual keyword: prevents the member function of the base class from working instead of the functions of the derived class
- Polymorphism: Let compiler know the base class only. Appropriate mapping to the derived class is done later.

104. Project - V
- Polymorphism
    - Different forms of the function are provided
    - The call is resolved at compile time or runtime
        - Ref: https://www.geeksforgeeks.org/polymorphism-in-c/
        - Runtime polymorphism or dynamic binding
            - Implemented through virtual mechanism
        - Compile time polymorphism or static binding
            - Achieved through function/operator overloading
            - Only through pointers or references
- Vtable & Vptr
    - Virtual table keeps the address of virtual functions
        - An array of pointers
    - Virtual pointer is generated by the compiler and points the virtual function in the virtual table

105. Project - VI
- When destructed, the destructor of Checking -> Account is executed accordingly
- But when `Account *acc04 = new Checking {"John", 10.}; delete acc04;`, the destructor of checking will not be invoked. This is why **the destructor of the base class MUST be `virtual`**

106. Project - VII
- `final` keyword: prevents a class from becoming a base class
    - `class Abc final {...};`
    - When tried to load, the compiler will yield an error message
- `override`: with `virtual` keyword in the base class, let derived class override the member function
    - If the member function of the base class doesn't have `virtual` keyword, compile error will appear
    - Example:
        - In the base class: `virtual float GetInterestRate() const {return m_Rate;}`
        - In the derived class: `float GetInterestRate() const override {return m_Rate;}`
            - `float GetInterstRate() override const {}` doesn't work. It breaks the signature (shape) of the fuction
- `override final`
    - Will disable the override by next derived classes
    - Ref: https://www.cppstories.com/2021/override-final/
    - In the base class `virtual void doSomething();`
    - In the first derived class `void doSomething() override final;`
    - In the next derived class `void doSomething() override;`=>This will yield compiler error

107. Project - VIII
- Upcasting: cast the derived class into the base class
    - `Checking ch; Account *p = &ch;`
    - Automatically done
- Downcast: cast the base class into a derived class
    - `Checking pch = static_cast<Checking*> (p)`
    - Needs manual casting
- Object slicing
    - A situation in which the compiler deliberately removes some part of an object
    - Occurs when child class object is assigned to a concrete base class object
    - ` Account a; Savings s; a = s`

108. Project - IX
- How to use the method of a derived class in the base class?
- Even though downcast may work, there is no gaurantee that it will point the correct function
- `typeid()`
    - For non-polymorphic object, it works at compile-time
    - For polymorphic, works at run-time
```cpp
int main() {
  Checking acc03("Bob",1000.00f);
  Account *p = &acc03;
  const std::type_info &ti = typeid(*p);
  if (ti == typeid(Savings)) {
    std::cout << " points to Savings object\n";
  } else {
    std::cout << " not Savings object\n";
  }
  std::cout << ti.name() << std::endl;
  return 0;
}
```

109. Project - X
- Using `dynamic_cast` instead of `typeid()`
```cpp
int main() {
  Checking acc03("Bob",1000.00f);
  Account *p = &acc03;
  Savings *s = dynamic_cast<Savings*> (p);
  if (s) {
    std::cout << " points to Savings object\n";
  } else {
    std::cout << " not Savings object\n";
  }
  return 0;
}
```
- `dynamic_cast` works for reference too
    - `dynamic_cast<Savings&>(p)`
    - But nullptr is not returned when failed. Instead, it will throw an error. May use try/catch

110. Abstract class
- Pure virtual
    - Disables the member function to be activated
    - Makes the base class as abstract - cannot be instantiated
        - But a pointer or a reference still works
    - Use `pure specifier` as `= 0` for the base member function
- Abstract class
    - At least one pure virtual function
    - Can contain member data, non-virtual function, ...
    - Cannot be instantiated but used as a pointer or a reference
    - Establishes a contract with clients
        - Contract: checkable interfaces for SW components 
        - Ref: https://www.modernescpp.com/index.php/c-core-guidelines-a-detour-to-contracts
    - Used for creating interface
```cpp
#include <iostream>
class Document {
  public:
    virtual void Serialize() = 0;
};
class Text : public Document {
  public:
    void Serialize() override {std::cout << " from Text\n"; };
};
class XML : public Document {
  public:
    void Serialize() override {std::cout << " from XML\n"; };
};
int main() {
  XML xml;
  xml.Serialize();
  return 0;
}
```

111. Multiple inheritance
- Diamond inheritance
```
      A
    /   \
   B     C
    \   /
      D
```
- Constructor of D may invoke of the constructor A twice
    - In B and C, must use `virtual public A` instead of `public A`    

## Section 10: Exception handling

112. Exception Handling - part I
- Mechanism to handle errors in programs that occur at runtime
- These errors are called exceptions
- Exist outside of the normal functioning of the program
- Requires immediate handling by the program
- If not handled, the program crashes
- Cannot be ignored, unlike C error handling (retunr 0 or -1)
- Mechanism
    - try
        - Testing code
    - throw
        - Exception from the try block
        - Exception is an object that is constructed in throw statement
    - catch
        - Handler that cathes the exception block
        - Multiple cat blocks can exist
```cpp
#include <iostream>
#include <vector>
int Process(int16_t count) {
  int16_t* x = (int16_t*)malloc(count*sizeof(int16_t));
  if (x==nullptr) {
    throw std::runtime_error("cannot allocate");
  }
  free(x);
  return 0;    
}
int main() {
  try {
    int rv = Process(std::numeric_limits<int16_t>::max()+1);
  } catch (std::runtime_error &ex) {
    std::cout << "we have: " << ex.what() << std::endl;
  }
  return 0;
}
```

113. Exception Handling - part II
```cpp
#include <iostream>
#include <vector>
int Process(int16_t count) {
  if (count < 10 & count > 0) 
    throw std::out_of_range("Count should be >10");
  if (count == 100)
    throw std::runtime_error("Designed to throw at 100");
  int16_t* x = new int16_t[count];
  delete[] x;
  return 0;    
}
int main() {
  try {
    int rv = Process(33000);
  } catch (std::runtime_error &ex) {
    std::cout << "error_message is : " << ex.what() << std::endl;
  }
  catch (std::out_of_range &ex) {
    std::cout << "error_message is : " << ex.what() << std::endl;
  }
  catch (std::bad_alloc &ex) {
    std::cout << "error_message is : " << ex.what() << std::endl;
  }
  catch (std::exception &ex) {
    std::cout << "error_message IS : " << ex.what() << std::endl;
  }
  catch (...) {
    std::cout << "no identity of exception " << std::endl;
  }
  return 0;
}
```
- All of exception is inherited from std::exception and  `catch (std::exception &ex) {    std::cout << "error_message IS : " << ex.what() << std::endl;  }` might be located in the end of catch block
- `catch (...) {    std::cout << "no identity of exception " << std::endl;  }` can catch anything but it doesn't tell any details of the exception caught. Not recommeded to use

114. Exception Handling - Part III
- Stack unwinding
    - When exception is caught, local object which was created on stack will be destructed using the destructor
    - If an object is created on heap, memory deallocation is not done and memory leak will happen
        - **Use smart pointer instead of a raw pointer**
        - Or use std::vector
```cpp
#include <iostream>
#include <memory>
class myObj {
public:
  myObj()  {std::cout << "created\n";}
  ~myObj() {std::cout << "removed\n";}
};
int Process(int16_t count) {
  myObj t1;
  myObj *t2 = new myObj;
  std::unique_ptr<myObj> t3 {new myObj};
  if (count <10 )
    throw std::out_of_range("too small");
  delete t2;
  return 0;    
}
int main() {
  try {
    int rv = Process(3);
  }
  catch (std::exception &ex) {
    std::cout << "error_message IS : " << ex.what() << std::endl;
  }
  return 0;
}
```

115. Exception Handling - Part IV
- Nested exception
    - std::vector<bool> does not obey the standard container rules : https://stackoverflow.com/questions/34079390/range-for-loops-and-stdvectorbool
```cpp
#include <iostream>
#include <vector>
int Process(int16_t count) {
  std::vector<bool> b1 {true, false, true, false, true, false};
  int ierr {};
  for (auto el : b1) {
    try {
      if (!el) {
        ++ierr;
        throw std::runtime_error("false found");
      }
    } catch (std::runtime_error &ex) {
      std::cout << "Error found: " << ex.what() << std::endl;
      if (ierr>2) {
        throw std::out_of_range("throwing to the first layer");
      }
    }
  }
  return 0;    
}
int main() {
  try {
    int rv = Process(3);
  }
  catch (std::exception &ex) {
    std::cout << "first layer error_message IS : " << ex.what() << std::endl;
  }
  return 0;
}
```

116. Exception handling - Part V
- If an error is to be thrown from a constructor, raw pointers must be avoided as member data
    - The destructur will not be invoked
- Use STL or smart pointers

117. Exception handling - Part VI
- noexcept
    - Applied to functions
    - Indicates the function does not throw exceptions
    - Compiler can do optimization on such functions
    - An exception from such functions will terminate the program
```cpp
int Sum(int x, int y) noexcept(true) {...}
```

## Section 11: File Input & Output

118. Raw string literals (C++11)
- `\` is a tap and `\n` is a new line
    - `"C:\TEMP\newfile.txt"` => `"C:\\TEMP\\newfile.txt"`
    - Or as a Raw String `R"(C:\TEMP\newfile.txt)"`
    - Use Raw string for path/directory or file names
- Using customized delimiter for complex string (C++17)
```cpp
#include<iostream>
#include<string>
int main() {
  std::cout << std::string("C:\temp\newfile.txt") << std::endl;
  std::cout << std::string("C:\\temp\\newfile.txt") << std::endl;
  std::cout << std::string(R"(C:\temp\newfile.txt)") << std::endl;
  std::cout << std::string(R"MSG(C:\temp\newfile.txt)MSG") << std::endl;  
  return 0;
}
```
```bash
$ g++ -std=c++17 118_rawstring/main.cc 
$ ./a.out 
C:	emp
ewfile.txt
C:\temp\newfile.txt
C:\temp\newfile.txt
C:\temp\newfile.txt
```

119. Introduction to Filesystem library
- `experimental/filesystem` in c++17 (?)
    - Not sure this works at c++11/14 only or at c++17
```cpp
#include <iostream>
#include <experimental/filesystem>
int main() {
  using namespace std::experimental::filesystem;
  path p{ R"(/etc)"};
  if (p.has_filename()) std::cout << p.filename() << std::endl;
  //
  directory_iterator beg {p};
  directory_iterator end{};
  while (beg !=end) {
    std::cout << *beg << std::endl;
    ++beg;
  }
  return 0;
}
```
- Needs `-lstdc++fs` 
```bash
$ g++ -std=c++17 119_fs/main.cc  -lstdc++fs
$ ./a.out 
"etc"
"/etc/anacrontab"
"/etc/pcmcia"
...
```

120. File IO - part I
- File open modes
    - app: appending
    - binary: open as binary
    - in: for reading
    - out: for writing
    - trunc: Overwriting
    - ate: seek to end after open
    - can be combiled such as `std::ios:in | std::ios::out`

121. File IO - part II
- Stream state flags
    - good(): no error
    - bad(): irrecoverable stream error
    - fail(): IO Operation failed
    - eof(): end of file reached during input
- When no more characters can be read from a file, eofbit and  failbit are set while goodbit is set as false
```cpp
#include<iostream>
#include<fstream>
#include<string>
void Write() {
  std::ofstream out{"data.txt"};
  out << "hello world\n";
  out << 123 << std::endl;
  out.close();
}
void Read() {
  std::ifstream inp { "data.txt"};
  if(!inp.is_open()) {
    std::cout << "Cannot open the file\n";
    return;
  }
  std::string msg;
  std::getline(inp,msg);
  int value {};
  inp >> value;inp >> value; // 2nd value is not valid
  if (inp.fail()) {std::cout << "cannot read\n";}
  if (inp.eof()) {std::cout << "EOF reached\n";}
  if (inp.good()) {
    std::cout << "IO successful\n";
  } else {
    std::cout << "IO failed\n";
  }
  inp.close();
  std::cout << msg << ":" << value << std::endl;
}
int main() {
  Write();
  Read();
  return 0;
}
```
- `inp.clear()` will clear the fail status
- `inp.setstate(std::ios::failbit)` may set new fail status

122. File IO - part III
- Reading entire lines of a file
```cpp
std::ifstream inp {"input.data"};
std::string line;
while (!std::getline(input.line).eof()) {
    std::cout << line << std::endl;
}
```

123. File IO - part IV
- Writing character
    - `out.tellp()` shows the current position
```cpp
std::ofstream out {"output.txt"};
std::string msg {"hello world"};
for (char ch : msg) out.put(ch);
```
- Reading character
    - `inp.tellg()` shows the current position
    - `inp.seekg(offset,mode)` sets the current position
        - mode would be one of std::ios::beg, std::ios::curr, std::ios::end
```cpp
std::ifstream inp {"input.data"};
char ch {};
while (inp.get(ch)) {
    std::cout << ch;
}
```

124. File IO - part V
- No EOF character in binary files

125. Assignment I 
- Copying binary files
```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <experimental/filesystem>
int main() {
  using namespace std::experimental::filesystem;
  path source(current_path());
  source /= "abc.jpg";
  path dest(current_path());
  dest /= "Copy.jpg";
  std::ifstream input { source, std::ios::binary|std::ios::in };
  if (!input) {
    std::cout << " source file missing\n";
    return -1;
  }
  if (exists(dest)) {
    std::cout << "file : " << dest << " will be over-written\n";
  }
  std::ofstream output { dest, std::ios::binary|std::ios::out };
  output << input.rdbuf();
  return 0;
}
```
- Ref: https://stackoverflow.com/questions/5420317/reading-and-writing-binary-file

126. Assignment II
- Copy the entire contents of the folder

## Section 12: Templates
- For generic programming

129. Introduction to Templates
- Templates
  - Generalizes SW components
  - High performance algorithms & classes
  - Compile time; no runtime costs are involved  
- If the templated function is not called, compiler will not instantiate the functions at all
  - if int/float types are called, only functions of int/float are produced/compiled
```cpp
#include <iostream>
template<typename T>
T find_max(T a, T b) {
  return a>b ? a: b;
}
int main() {
  std::cout << find_max(1,3) << std::endl;
  std::cout << find_max(1.f,3.f) << std::endl;
  return 0;
}
```
- Used find_max() with integer or float only
```assembly
00000000000009ab <int find_max<int>(int, int)>:
...
00000000000009c7 <float find_max<float>(float, float)>:
```
- Binary finds only int and float version only
  - If double data type is called, then compiler will instantiate double version in the assembly

130. Assignment I
- Convert following functions into templates
  - int Add(int x, int y)
  - int ArraySum(const in *pArr, size_t arrSize);
  - int Max(const int *pArr, size_t arrSize);
  - std::pair<int,int>MinMax(const int *pArr, size_t arrSize);
```cpp
#include <iostream>
template<typename T>
T Add(T a, T b) {
  return a+b;
}
template<typename T>
T ArraySum(const T *pArr, size_t arrSize) {
  T local_sum {};
  for(size_t i=0; i< arrSize; ++i) {
    local_sum += pArr[i];
  }
  return local_sum;
}
template<typename T>
T Max(const T *pArr, size_t arrSize) {
  T local_max {pArr[0]};
  for (size_t i=1; i< arrSize; ++i) {
    local_max = local_max > pArr[i] ? local_max : pArr[i];
  }
  return local_max;
}
template<typename T>
std::pair<T,T> MinMax(const T *pArr, size_t arrSize) {
  T local_max {pArr[0]};
  T local_min {pArr[0]};
  for (size_t i=1; i< arrSize; ++i) {
    local_max = local_max > pArr[i] ? local_max : pArr[i];
    local_min = local_min < pArr[i] ? local_min : pArr[i];
  }
  return std::make_pair(local_min,local_max);
}
int main() {
  std::cout << Add(1,3) << std::endl;
  std::cout << Add(1.f,3.f) << std::endl;
  size_t sArr = 3;
  int* iArr = new int[sArr];
  iArr[0] = 1; iArr[1] = 5; iArr[2] = 2;
  float* fArr = new float[sArr];
  fArr[0] = 1.f; fArr[1] = 5.f; fArr[2] = 2.f;
  std::cout << ArraySum(iArr, sArr) << std::endl;
  std::cout << ArraySum(fArr, sArr) << std::endl;
  std::cout << Max(iArr, sArr) << std::endl;
  std::cout << Max(fArr, sArr) << std::endl;
  auto p1 = MinMax(iArr, sArr);
  auto p2 = MinMax(fArr, sArr);
  std::cout << p1.first << " " << p1.second << std::endl;
  std::cout << p2.first << " " << p2.second << std::endl;
  delete[] iArr;
  delete[] fArr;
  return 0;
}
```

131. Template argument deduction & instantiation
- Template instantiation
  - A template function or class only acts as a blueprint
  - The compiler generates code from the blueprint at compile time
  - Known as template instantiation
  - Occurs implicitly when
    - a function template is invoked
    - taking address of a function template
      - `int (*pfn)(int, int) = find_max;`
    - using explicit instantiation
    - creating explicit specialization
- Define in header file
  - When defined in a cpp source, the object file will be empty if there is no instantiation
- Over-riding argument type deduction
```cpp
template<typename T>
T find_max(T a, T b) {
  return a>b ? a: b;
}
...
find_max<double>(3, 1.1f);
```
- `find_max<double>()` will override the argument type as double

133. Explicit specialization
- `char b {'B'}`: works OK
- `char *b {'B'}`: NOT allowed
- `char *b {"B"}`: NOT allowed
- `const char *b {"B"}`: works OK
```cpp
  std::cout << find_max('a','b') << std::endl; // prints b
  char b {'B'}; char a {'A'};  
  std::cout << find_max(a,b) << std::endl; // prints B
  const char* pb {"B"}; const char* pa {"A"};  
  std::cout << find_max(pa,pb) << std::endl; // prints A due to comparison of the address, not value
```
- Explicit specialization
  - Specialized template for a particular type
  - Provides correct semantics for some datatype
  - Or provides an algorithm optimally for a specific type
  - Explicitly specialized functions must be defined in a .cpp file
```cpp
template<> const char* find_max(const char* a, const char* b) {
  return strcmp(a,b) > 0 ? a : b;
}
```
- Note that we need `template<>` in the beginning then redefine the function

134. Non-type template arguments
- In template definition, actual data type instead of `typename` then the argument becomes the function parameter
  - It is `constexpr` and a regular variable is NOT allowed
    - Addresses, references, integrals, nullptr, enums
      - `std::string msg` is not allowed but `std::string &msg` will work
      - https://stackoverflow.com/questions/5687540/non-type-template-parameters
    - Must be defined at compile time
    - Used by std::begin & std::end functions
  - This can be used to deduce the size of a static array
    - `template<typename T, size_t arrSize> T ArraySum(T (&pArr)[arrSize]) {` then calling `ArraySum(iArr)` will map the data type of `iArr` into T (typename), and then the size of `iArr`will be mapped into `arrSize`
    - Ref: https://stackoverflow.com/questions/3368883/how-does-this-size-of-array-template-function-work
```cpp
#include<iostream>
template<int myInt>
void Print() {
  std::cout << myInt << std::endl;
}
int main() {  
  Print<3>(); // works OK
  //int i=3; Print<i>();  // compile fails
  const int i=3; Print<i>(); // works OK
  return 0;
}
```

135. Assignment III
- Using a reference instead of a pointer
```cpp
#include <iostream>
template<typename T, size_t arrSize>
T ArraySum(T (&pArr)[arrSize]) {
  T local_sum {};
  for(size_t i=0; i< arrSize; ++i) {
    local_sum += pArr[i];
  }
  return local_sum;
}
int main() {
  size_t sArr = 3;
  int iArr[] {1,5,2};
  float fArr[] {1.f,5.f,2.f};  
  std::cout << ArraySum(iArr) << std::endl;
  std::cout << ArraySum(fArr) << std::endl;
  return 0;
}
```
  - Note that iArr[] is an array, not a pointer
  - **We don't need the size of array as the second argument**
  - The size of Array is taken as non-type template argument (?)
- Specialize for an array of strings (const char *)
  - Q: How to define return type? If the return type is const char for the Sum(), then only one char is returned.
- Specialize for an array of std::strings

136. Perfect forwarding - Part I

137. Perfect forwarding - Part II
- using `std::forward<T>(x)`
- Q: Why R value reference is used?

138. Variadic templates - part I
- For an array input, if the data type is changed in the middle, it will not be compiled
  - Variadic template can avoid such cases
  - Variadic templates are written like a recursive code
    - https://eli.thegreenplace.net/2014/variadic-templates-in-c/
- `...` : ellipsis
- Parameter pack: a template parameter that accepts zero or more arguments
- Using recursion, we may reach arguments one by one
```cpp
#include <iostream>
// base case function
void Print() {
  std::cout <<"reached the base function\n";
}
template<typename T, typename...Params>
void Print(T a, Params... args){
  std::cout << "calling " << a << std::endl;
  Print(args...);
}
int main() {
  Print(1, 2.5f, 3.0, "4");
  return 0;
}
```
- First argument is taken by `T a` and the left-over will be sent to next recursion, until reaching Print(), which has no argument and is the base case function
- This will produce 5 different print functions:
```assembly
00000000000009aa <Print()>:
0000000000000a52 <void Print<int, float, double, char const*>
0000000000000ac6 <void Print<float, double, char const*>(float, double, char const*)>:
0000000000000b36 <void Print<double, char const*>(double, char const*)>:
0000000000000b96 <void Print<char const*>(char const*)>:
```

139. Variadic templates - part II
- Finding the size of parameter pack
```cpp
void Print(T a, Params... args){
  std::cout << sizeof...(args) << " " 
    << sizeof...(Params) << std::endl;
  Print(args...);
}
```
- To call another function using the parameter pack, use `func(args...);`

140. Assignment IV
- Create a factory function
```cpp
#include <iostream>
#include <string>
class Employee {
  std::string Name;
  int Id;
  int Salary;
public:
  Employee() = default;
  Employee(std::string name, int id, int salary) 
    : Name(name), Id(id), Salary(salary) { 
      std::cout << Name << Id << Salary << std::endl;
    }
  ~Employee() = default;
};
class Contact {
  std::string Name;
  long int PhoneNumber;
  std::string Address;
  std::string Email;
public:
  Contact() = default;
  Contact(std::string name, long int pn, std::string address, std::string email) 
  : Name(name), PhoneNumber(pn), Address(address), Email(email) {
    std::cout << Name << PhoneNumber << Address << Email << std::endl;
  }
  ~Contact() = default;
};
template<typename T, typename...Params> 
T* CreateObject(Params... args) {
  T* obj = new T(args...);
  return obj;
}
int main() {
  int *p1 = CreateObject<int>(5);
  std::string *s = CreateObject<std::string>();
  Employee *emp = CreateObject<Employee>("Bob", 101, 1000);
  Contact *p = CreateObject<Contact>("Joey", 123456789, "One street MA", "myEmail@altavista.com");
  return 0;
}
```

141. Class Templates







182. Concurrency basics
- Provides better user experience at GUI

183. Thread creation
- callable: pointer, function, lambda function ...
- std::thread
    - Accepts a callable as constructor argument
    - The callable is executed in a separated thread
    - The constructor does not wait for the thread to start

185. Thread synchronization using std::mutex
- If mutex is not unlocked after locked, it will hang -> deadlock

186. std::lock_guard
- Destructor is called when leaving the scope

187. std::thread functions and std::this_thread namespace

188. Task based concurrency - part 1
```cpp
#include <future>
#include <iostream>
#include <thread>
#include <chrono>
void func1() {
    for (auto &x : {1,2,3,4,5,6,7,8,9}) {
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}
int main() {
    std::future<void> result = std::async(func1);
    std::cout << "downloading\n" ;
    result.get();
}
```
- Template of std::future is the return type of the function callable
- .get() waits for the thread to close
- If std::flush is not used, the print of `...` is buffered

189. Task based concurrency - part 2
- std::async
    - High level concurrency    
    - Executes a callable object or a function in a separate thread
    - Returns std::future object that provides access to the result of the callable
    - future<return_type> async(Callable, args)
    - future<return_type> async(launch policy, callable, args)
    - Arguments are passed by value as default
    - To pass by reference, use std::ref or std::cref for constant reference
- Launch policy
    - std::launch::deferred - the task is executed synchronously
    - std::launch::async - the task is executed asynchronously
    - Compiler dependency
    - To force std::async to execute the task asynchronously, use async launch policy explicitly
        - `std::future<void> result = std::async(std::launch::async, func1);`
- std::future
    - Used for communication b/w threads
    - Has a shared state that can be accessed from a different thread
    - Created through std::promise object
        - Created by std::async, that directly returns a future object
        - std::promise is an input channel
        - std::future is the output channel
    - The thread that reads the shared state will wait until the future is ready
    - The pair of promise/future allows safe data sharing b/w threads

190. Launch policies
```cpp
#include <future>
#include <iostream>
#include <thread>
#include <chrono>
int func1(const int&& count) {
    int sum{};
    for (int i=0;i<count;i++) {
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        sum +=i;
    }
    return sum;
}
int main() {
    std::future<int> result = std::async(std::launch::async, func1, 10);
    std::cout << "downloading\n" ;
    if (result.valid()) {
        auto sum = result.get();
        std::cout << sum << std::endl;
    }
}
```

191. std::future wait function
- result.wait_for(4s): returns std::future_status::deferred or std::future_status::ready or std::future_status::timeout
- result.wait_until(timepoint): 

192. std::promise
- Provides a way to store a value or an exception
- this is called the shared state
- This state can be accessed later from another thread through a future object
- promise/future are two endpoints of a communication channel
- One operation stores a value in a promise then the other operation will retrieve it through a future asynchronously
- promise object can be used only once
```cpp
#include <future>
#include <iostream>
#include <thread>
#include <chrono>
int func1(std::promise<int> &inp) {
    auto f = inp.get_future();
    std::cout << "waiting for inp\n";
    auto count = f.get(); // up to here, it is blocked
    std::cout << "inp received\n";
    int sum{};
    for (int i=0;i<count;i++) {
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        sum +=i;
    }
    return sum;
}
int main() {
    std::promise<int> inp;
    std::future<int> result = std::async(std::launch::async, func1, std::ref(inp));
    std::cout << "downloading\n" ;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    inp.set_value(10);
    if (result.valid()) {
        auto sum = result.get();
        std::cout << sum << std::endl;
    }
}
```
- Note that the promise (inp) was sent even before it was assigned
    - .set_value() was applied to setup the value later

199. Propagating exception
- Instead of inp.set_value(), use inp.set_exception
    - `data.set_exception(std::make_exception_ptr(ex))` for `std::exception &ex`

