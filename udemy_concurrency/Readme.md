## Modern C++ Concurrency in Depth ( C++17/20)
- Instructor: Kasun Liyanage

2. Introduction to parallel computing
- Process: an instance of a computer program that is being executed
- Context: collection of data about process which allows processor to suspend or hold the execution of a process and restart the execution later
- Thread
    - Thread of execution is the smallest sequence of programmed instructions that can be managed independently by a scheduler
    - Thread is component of a process. Every process has at least on thread called main thread which is the entry point for the program
- Process and threads
    - The typical difference b/w the threads and process is that threads of the same process run in a **shared memory space** while processes run in **separate memory space**
- Parallelism vs concurrency
    - Concurrency: many threads by a single processor using context switching. Actually they run sequentially
    - Parallelism: many threads on many processors

4. How to launch a thread

5. Programming exercise 1: Launching the threads
```cpp
#include <iostream>
#include <thread>
void test() {
    std::cout << "hello from test\n";
}
void functionA() {
    std::thread threadC(test);
    threadC.join();
    std::cout << "hello from funciton A\n";
}
void functionB() {
    std::cout << "hello from function B\n";
}
int main() {
    std::thread threadA(functionA);
    std::thread threadB(functionB);
    threadA.join();
    threadB.join();
    return 0;
}
```
- Execution order is random. How to make sequential?
```cpp
int main() {
    std::thread threadA(functionA);
    threadA.join();
    std::thread threadB(functionB);
    threadB.join();
    return 0;
}
```

6. Joinability of threads
- Joinable: **Properly constructed** thread object which represents an active thread of execution in HW level
    - Must be used with join() or detach()
    - After join() or detach(), it becomes non-joinable
    - Without join() or detach(), std::terminate() will be called when destructed
        - When std::terminate() is called, it means the program is NOT safe
    - `thread.joinable()` shows if it is joinable or not
- `Properly constructed`: a valid object must be sent as an argument
    - `std::thread thread2;` is not joinable

7. join() and detach()
- join(): introduces a synchronize point b/w launched thread and the parent thread
    - Blocks the execution of the parent thread until the launched thread finishes
- detach(): separates the launched thread from the parent thread, allowing to continue independently
    - Allocated resources are freed when execution finishes
```cpp
#include <thread>
#include <iostream>
#include <chrono>
void foo() {
    std::cout << " from foo;\n";
}
void bar() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    std::cout << " from bar;\n";
}
int main() {
    std::thread t_foo(foo);
    std::thread t_bar(bar);
    t_bar.detach();
    t_foo.join();
    return 0;
}
```
- foo is executed but we don't see the result from bar

8. How to handle join  in exception scenario
- We can call detach() as soon as we launch a thread, as detach() will not block the parent thread
- In some cases, we cannot call join() as soon as we launch a thread, as join() will bock the parent thread
- Ex) Spawning thread -> ... -> thread.join()
    - What if it crashes before join()?
    - May use try/catch
```cpp
#include <thread>
#include <iostream>
void foo() {
    std::cout << " from foo;\n";
}
void somefunction() {
    throw std::runtime_error("crashed!");
}
int main() {
    std::thread t1(foo);
    try {
        somefunction();
        t1.join();
    } catch(...) {
        t1.join();
    }
    return 0;
}
```
- RAII: Resource Acquisition Is Initialization
    - Constructor requires resources/Destructor releases resources
```cpp
#include <thread>
#include <iostream>
class thread_guard {
    std::thread& t;
    public:
    explicit thread_guard(std::thread& _t)   : t(_t)
    {}
    ~thread_guard() {
        if (t.joinable()) {
            t.join();
        }
    }
    // disable copy
    thread_guard(const thread_guard & ) = delete; 
    thread_guard & operator= (const thread_guard &) = delete;
};
void foo() {
    std::cout << " from foo;\n";
}
void somefunction() {
    throw std::runtime_error("crashed!");
}
int main() {
    std::thread t1(foo);
    thread_guard tg(t1);
    try {
        somefunction();
    } catch(...) {        
    }
    return 0;
}
```
- When thread_guard object is destroyed, it will join the thread if joinable

9. Programming exercise 2: 
- highlight the responsibility of a given rol. Each role should be performed by different functions. Launch a thread with particular function to perform a given order
- Captain by main thread. Captain issues 3 commands
    - Order cleaning crew to clean. does not wait on the command until done
    - Two more commands to engine. Captain waits until the engine crewe finishes (full speed or stop)
- Write a program takes input from console
- Input will be 1 for cleaning, 2 for full speed, 3 for stop and 100 for exit
- For other input, print "invalid order"
```cpp
#include <iostream>
#include <thread>
void cleaning() {
    std::cout << "now cleaning ... \n";
}
void fullspeed() {
    std::cout << "now going full speed ... \n";
}
void stop_ship() {
    std::cout << "now stops the ship\n";
}
int main() {
    int n;
    bool NotDone = true;
    while (NotDone) {
        std::cin >> n;
        if (n==100) NotDone = false;
        else if (n==1) {
            std::thread t_ccrew(cleaning);
            t_ccrew.detach();
        } else if (n==2) {
            std::thread t_engine(fullspeed);
            t_engine.join();
        } else if (n==3) {
            std::thread t_engine(stop_ship);
            t_engine.join();
        } else 
            std::cout << "invalid order \n";
        
    }
    return 0;
}
```
- ? Do we need thread guard? What would be the idea behind of engine vs cleaning crew?

10. How to pass parameters to a thread
- May need to send arguments as by value
- Sending by reference will be affected when the value is changed in the main thread
    - Need to wrap wtih std::ref()
```cpp
#include <iostream>
#include <thread>
void func1(const int x, const int y) {
    std::cout << "X + Y = " << x + y << std::endl;
}
void func2(const int& x, const int& y) {
    for (int i=0;i<5;i++) {
        std::cout << "X + Y = " << x + y << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
int main() {
    int p=9;
    int q=8;
    std::thread t1(func1, p, q);
    t1.join();
    std::thread t2(func2, std::ref(p), std::ref(q));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    p += 10;
    q += 10;
   t2.join();
    return 0;
}
```
- Command
```bash
$ g++ -std=c++20 10_parameters/main.cc -pthread
$ ./a.out 
X + Y = 17  <-- call by value
X + Y = 17  <-- call by reference
X + Y = 17
X + Y = 37  <-- Now affected by the change in the parent thread
X + Y = 37
X + Y = 37
```

11. Problematic situation when passing parameters
```cpp
#include <iostream>
#include <thread>
void func2(const int& x) {
    while(true) {
        try {
            std::cout << x << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        } catch (...) {
            throw std::runtime_error("runtime error");
        }
    }
}
void func1() {
    int x=5;
    std::thread t2(func2, std::ref(x));
    t2.detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    std::cout << "thread1 finished \n";
}
int main() {
    std::thread t1(func1);
    t1.join();
    return 0;
}
```
- As func1() closes, memory of x is released but func2() will still try to reach x, which is already expired.

12. Transfering ownership
- Thread object is not copiable
- `std::thread t2 = std::move(t1)` works but t1 will be empty
```cpp
#include <thread>
#include <iostream>
#include <chrono>
void foo() {
    std::cout << " from foo;\n";
}
void bar() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    std::cout << " from bar;\n";
}
int main() {
    std::thread t1(foo);
    std::thread t2 = std::move(t1);
    //t1 = std::thread(bar);
    std::thread t3(foo);
    t1 = std::move(t3);
    t1.join();
    t2.join();
    //t3.join();
    return 0;
}
```
- Don't use `t1 = std::thread(bar)`. Needs explicit std::move() or constructor.

13. Some useful operations on thread
- get_id():
    - Prints ID of the thread
    - When the thread is not active, 0 is printed for Windows(?) while Linux generates a message saying that it is not active
```cpp
std::thread t1(foo);
std::cout << "t1=" << t1.get_id() << std::endl; //=> t1=139827655472896
std::thread t2 = std::move(t1);
//t1 = std::thread(bar); -> if we use this, get_id() crashes
std::cout << "t1=" << t1.get_id() << std::endl; //=> t1=thread::id of a non-executing thread
std::thread t3(foo);
t1 = std::move(t3);
std::cout << "t1=" << t1.get_id() << std::endl; //=> t1=139827647080192
```
- sleep_for():
- yield(): gives up the current time slide and re-inserts the thread into the scheduling queue
- hardware_concurrency(): returns the number of concurrent threads supported by the implementation. Would be the number of logical core but could be different

14. Programming exercise 3
- Use 2 std::queue type variables: Engine work queue and clean work queue


15. Serial accumulate
- vector sum: `std::accumulate(v.begin(),v.end(),0)`
- vector prodcut: `std::accumulate(v.begin(),v.end(),1,std::multiplies<int>())`

16. Parallel acccumulate

17. Thread local storage
- `thread_local std::atomic<int>`

19. Locking mechanism
- Introduction of mutex

20. Concept of invariants
- Broken invariants
    - Ex) deleting intermediate elements of a linked list -> race condition

21. Mutex
- A mutex class is a synchronization primitive
- Provides MUTual EXclusive access of shared data
- STL or list is not thread safe at all
    - Pushing into list is not thread safe
- mutex functions
    - lock
    - try_lock
    - unlock
```cpp
#include <mutex>
...
std::mutex m;
void add_to_list(int const& x) {
    m.lock();
    my_list.push_front(x);
    m.unlock();
}
```
- lock_guard
    - a class wraps a mutex, providing RAII style mechanism
    - The above implementation will be:
```cpp
void add_to_list(int const& x) {
    std::lock_guard<std::mutex> lg(m);
    my_list.push_front(x);
}
```

22. Things to remember when we use mutex
- Scenarios
    - Returning pointer or reference to the proteced data
    - Passing code to the protected data structure which we don't have control with

23. Thread safe stack implementation
- Stack: LIFO (Last In First Out) data structure

24. Thread safe stack implementation
```cpp
template<typename T>
class trivial_thread_safe_stack() {
    std::stack<T> stk;
    std::mutex m;
public:
    void push(T element) {
        std::lock_guard<std::mutex> lg(m);
        stk.push(element);
    }
    void pop() {
        std::lock_guard<std::mutex> lg(m);
        stk.pop();
    }
    T& pop() {
        std::lock_guard<std::mutex> lg(m);
        return stk.pop();
    }
}
```
- Add lock_guard into every method
- Still possibility of race conditions

25. Thread safe stack implementation: race condition inherit from the interface
- Check if it is empty before calling push()/pop()
- Using shared ptr

26. Deadlocks
- 2 mutexes wait for each other to be available
- 2 threads may join each other's
- To avoid:
    - Avoid nested locks
    - Avoid calling user supplied code while holding locks
    - Acquire locks in the same order

27. Unique locks
- unique_lock: general purpose mutex ownership wrapper
- lock_guard object locks associated mutex at the construction where unique_locks constructs without locking the associated mutex using locking strategy