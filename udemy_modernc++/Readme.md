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
```
class Myclass {
    ...
    Myclass(const &Myclass) = delete;
    ...
}
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

76. Weak ownership
- 





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