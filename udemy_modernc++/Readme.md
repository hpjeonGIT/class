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
- Rule of 0
    - If a class doesn't have ownership semantics, do not provide any of Rule of 5
    - Compiler will decide
        - User defined function may conflict


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
