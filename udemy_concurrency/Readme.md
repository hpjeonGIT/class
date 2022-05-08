## Modern C++ Concurrency in Depth ( C++17/20)
- Instructor: Kasun Liyanage

## Section 1: Thread management guide 

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

## Section 2: Thread safe access to shared data and locking mechanism

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

## Section 3: Communication b/w thread using condition variables and futures

28. Introduction to condition variables

29. Condition variable
- a mechanism waiting for an event to be triggered by another thread
- over night bus analogy
    - event: arriving to the destination
    - waiting threads - sleeping passenger
- Condition variable wake up can be due to:
    - notification from another thread
    - it can be spurious wake

30. Thread safe queue implementation: queue data structure
- Queue: FIFO
- Stack: LIFO
- Avoiding race conditions in pop() in queue
    - When two threads execute pop() simultaneously
- empty vs front
- empty vs back
- pop vs back

31. Thread safe queue
```cpp
#include <thread>
#include <mutex>
#include <stack>
template<typename T>
class thread_safe_queue() {
    std::mutex m;
    std::condition_variable cv;
    std::queue<std::shared_ptr<T>> queue;
public:
    thread_safe_queue();
    thread_safe_queue(thread_safe_queue const& other_queue);
    void push(T value) {
        std::lock_guard<std::mutex> lg(m);
        queue.push(std::make_shared<T>(value));
        cv.notify_one();
    }
    bool pop(T& ref);
    std::shared_ptr<T> pop(){
        std::lock_guard<std::mutex> lg(m);
        if (queue.empty()){
            return std::shared_ptr<T>();
        } else {
            std::shared_ptr<T> ref(queue.front());
            queue.pop();
            return ref;
        }
    }
    std::shared_ptr<T> wait_pop(){
        std::unique_lock<std::mutex> lg(m);
        cv.wait(lg, [this] {
            return !queue.empty();
        });
        std::shared_ptr<T> ref = queue.front();
        queue.pop();
        return ref;
    }
    bool empty() {
        std::lock_guard<std::mutex> lg(m);
        return queue.empty();
    }
}
```
- cv.wait(lock_guard, predicate): waits until notify_one() or notify_all(). Or may wait until predicate becomes true. At true, it proceeds to the next line

32. Introduction to futures and async
- Summary of steps
    - Creator of the async task has to get the future associate with async task
    - When creator of async task needs the result of that async task, call the get method on future
    - Get method may block if async operation has not yet completed
    - When async operation is ready to send a result to the creator, it can do so by modifying shared state which is linked to the creator's std::future    

33. async tasks detailed discussion
- std::async(std::launch policy, Function&&, Args&& ...)
- Launch policy
    - std::launch::async
    - std::launch::deferred
```cpp
#include <iostream>
#include <future>
#include <string>
void printing() {
    std::cout <<"printing runs on thread-" << std::this_thread::get_id() << std::endl;
}
int addition(int x, int y) {
    std::cout <<"addition runs on thread-" << std::this_thread::get_id() << std::endl;
    return x+y;
}
int subtract(int x, int y) {
    std::cout <<"subtract runs on thread-" << std::this_thread::get_id() << std::endl;
    return x - y;
}
int main() {
    std::cout <<"main thread-" << std::this_thread::get_id() << std::endl;
    int x = 100;
    int y = 50;
    std::future<void> f1 = std::async(std::launch::async, printing);
    std::future<int>  f2 = std::async(std::launch::deferred, addition,x,y);
    std::future<int>  f3 = std::async(std::launch::deferred | std::launch::async, subtract,x,y);
    f1.get();
    std::cout << "value from f2 future: " << f2.get() << std::endl;
    std::cout << "value from f3 future: " << f3.get() << std::endl;
    return 0;
}
```
- why addition thread is same to main thread?
    - std::launch::deferred

34. Parallel accumulate algorithm implementation with async task

35. Introduction to package_task
- The class template std::packaged_task wraps any callable target so that it can be invoked asynchronously
- Return value or exception is stored in a shared state which can be accessed through std::future objects
- `std::packaged_task<int(int,int)>task(callable_object)
- Running deferred
    - task_normal() below
    - Must be executed explicitly: `task1(3,4);`
- Running async
    - The task is sent to the thread usign std::move() then detach it
```cpp
#include <iostream>
#include <future>
#include <string>
#include <chrono>
int add(int x, int y) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout <<"addition runs on thread-" << std::this_thread::get_id() << std::endl;
    return x+y;
}
void task_thread() {
    std::packaged_task<int(int,int)> task1(add);
    std::future<int> f1 = task1.get_future();
    std::thread t1(std::move(task1),13,14);
    t1.detach();
    std::cout << "task1 thread = " << f1.get() << std::endl;
}
void task_normal() {
    std::packaged_task<int(int,int)> task1(add);
    std::future<int> f1 = task1.get_future();
    task1(3,4);
    std::cout << "task1 normal = " << f1.get() << std::endl;
}
int main() {
    std::cout <<"main thread-" << std::this_thread::get_id() << std::endl;
    task_thread();
    task_normal();
    return 0;
}
```
- Command:
```bash
$ g++ -std=c++20 35_package_task/main.cc -pthread
hpjeon@hakune:~/hw/class/udemy_concurrency$ ./a.out 
main thread-140464902215488
task1 thread = addition runs on thread-140464884107008
27
addition runs on thread-140464902215488
task1 normal = 7
```
- Note that task_normal() runs on the main thread

36. Communication b/w threads using std::promises
- std::promise object is paired with std::future object
- A thread with access to the std::future object can wait for the result to be set, while another thread that has access to the corresponding std::promise object can call set_value() to store the value and make the future ready
```cpp
#include <iostream>
#include <functional>
#include <future>
#include <string>
#include <stdexcept>
#include <chrono>
void print_int(std::future<int>& fut) {
    std::cout << "waiting for value from print thread\n"; // then this will wait until prom.set_value() runs
    std::cout << "value; " << fut.get() << std::endl;
}
int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    std::thread print_thread(print_int, std::ref(fut));
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "setting the value in the main thread\n";
    prom.set_value(10);
    print_thread.join();
    return 0;
}
```
- When print_int() runs, it will wait until prom.set_value()
- After prom.set_value(), finally fut.get() will run
    - This is done prior to print_thread.join()

37. Retrieving exception using std::futures
- Promise + Exception
```cpp
#include <iostream>
#include <functional>
#include <future>
#include <string>
#include <stdexcept>
#include <chrono>
#include <cmath>
void throw_exception() {
    throw  std::invalid_argument("input cannot be negative");
}
void calculate_sqroot(std::promise<int>& prom) {
    int x = 1;
    std::cout << "enter an integer number:";
    try {
        std::cin >> x;
        if (x <0) {
            throw_exception();
        }
        prom.set_value(std::sqrt(x));
    } catch (std::exception&) {
        prom.set_exception(std::current_exception());
    }
}
void print_result(std::future<int>& fut) {
    try {
        int x = fut.get();
        std::cout << "value: "  << x << std::endl;
    } catch(std::exception& e) {
        std::cout << "[exception caught: " << e.what() << "]\n";
    }
}
int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    std::thread print_thread(print_result, std::ref(fut));
    std::thread calc_thread(calculate_sqroot,std::ref(prom));
    print_thread.join();
    calc_thread.join();
    return 0;
}
```
- Can propagate exception through promise and future
    - fut.get() in print_result() will wait until prom.set_value() of calculate_sqroot()
    - When exception is thrown, prom.set_exception() is executed and will throw exception at print_result()

38. std::shared_futures
- What if multiple threads wait for the same future?
- Once a future object becomes invalid after called
    - May use `fut.valid()` prior to `fut.get()`
    - But this will cause race-condition
- Solution: used std::shared_future
```cpp
#include <iostream>
#include <functional>
#include <future>
#include <string>
#include <stdexcept>
#include <chrono>
#include <cstdlib>
void print_result(std::shared_future<int>& fut) {
    auto x = rand();
    if (x> RAND_MAX/2) 
        std::this_thread::sleep_for(std::chrono::milliseconds(100));    
    std::cout << fut.get() << " : valid future\n";
}
int main() {
    std::promise<int> prom;
    std::shared_future<int> fut = prom.get_future();
    std::thread t1(print_result, std::ref(fut));
    std::thread t2(print_result, std::ref(fut));
    prom.set_value(5);
    t1.join();
    t2.join();
    return 0;
}
```

## Section 4: Lock based thread safe data structures and algorithm implementation

39. Introduction to lock based thread safe data structure and algorithms
- How we can improve the above-mentioned implementation?

40. Queue data structure using linked list

41. Thread safe queue

42. Parallel STL introduction
- Execution policy
    - `std::sort(std::execution::par, v.begin(),v.end());`
    - Q: how to control number of threads? -> maybe NOT: https://stackoverflow.com/questions/47028495/how-do-the-c-stl-executionpolicy-algorithms-determine-how-many-parallel-thre
- Types of policy
    - sequential_policy
    - parallel_policy
    - parallel_unsequenced_policy
- Sample run
    - `std::sort(sorted.begin(),sorted.end());` or `std::sort(std::execution:seq, sorted.begin(),sorted.end());`
    - `std::sort(std::execution::par, sorted.begin(),sorted.end());`
        - Actually this is slower than sequential (!!!)

44. Parallel for each
- Split vectors into Nthreads-1 chunks
    - The main thread will manage the split/collection

45. Parallel find with package task

46. Parallel find with async

47. Prefix sum
- STP support
    - Inclusive scan
    - Exclusive scan
    - Partial_sum

48. Partial sum implementationa

50. Parallel matrix multiplication

52. Factors affecting the performance of concurrent code
- Number of processors
- Data contention and cache ping pong
- False sharing
- Closeness of data
- Division of array elements

## Section 5: C++20 Concurrency features

53. Jthread: introduction
- Why?
    - Thread has to explicityly either join or detach
    - We cannot interrupt the standard threads execution after we launch it
- Jthread Availability
    - Not in MSVC yet
    - GCC10 supports it
        - Needs `-std=c++20 -pthread`
- Jthread execution interruption
    - `stop_token`
    - We can introduce condition checks with `stop_token` to specify interrupt points
    
54. Jthread: 

55. C++ coroutines: Introduction
- `-fcoroutines -pthread` option is necessary for gcc10
- Subroutines: normal functions
    - Invokes
    - Finalizes
    - Functions are located in run time stack
- Coroutines:
    - Invokes
    - Suspends
    - Resumes
    - Finalizes
    - Coroutine state object is located in heap then copied to stack
    - Contains co_await, co_yield, co_return keywords

56. C++ coroutines: resume
- Coroutine elements:
    - A promise object: User defined promise_type object. Return results via this object
    - Handle: is used to resume or destroy coroutine from outside
    - Coroutine state: heap allocatd. Contains promise objects, arguments to coroutine and local varialbes
```cpp
#include <iostream>
#include <coroutine>
#include <cassert>
class resumable{
public:
  struct promise_type;
  using coro_handle = std::coroutine_handle<promise_type>;
  resumable(coro_handle handle): handle_(handle) {assert(handle);}
  resumable(resumable&) = delete;
  resumable(resumable&&) = delete; // no copy/move constructur needed
  bool resume() {
    if (not handle_.done()) handle_.resume();
    return not handle_.done();
  }
  ~resumable() {handle_.destroy();}
private:
  coro_handle handle_;
};
struct resumable:: promise_type {
  using coro_handle = std::coroutine_handle<promise_type>;
  auto get_return_object() noexcept {
    return coro_handle::from_promise(*this);
  }
  auto initial_suspend() noexcept { return std::suspend_always(); }
  auto final_suspend() noexcept { return std::suspend_always(); }
  void return_void() noexcept {}
  void unhandled_exception() noexcept { std::terminate();}
};
resumable foo() { // note there is no return type
  std::cout << "a\n";
  co_await std::suspend_always();
  std::cout << "b\n";
  co_await std::suspend_always();
  std::cout << "c\n";  
}
int main() {
  resumable res1 = foo();
  res1.resume();
  res1.resume();
  res1.resume();
  return 0;
}
```
    - Q: initial_suspend(), final_suspend(), unhandled_exception() are reserved names?
    - Q: why noexcept is required?
- Commands
```bash
$ g++ -std=c++20 -fcoroutines -pthread 56_coroutine/main.cxx 
$ ./a.out 
a
b
c
```

57. C++ coroutines: Lazy generator
- A function generates a sequence of numbers based on the demand from the caller function

58. C++ Barriers
- A sync mechanism that make threads to wait until the reuqired number of threads has reached a certain point in code.
- Supported in Boost, not in gcc yet

## Section 6: C++ memory model and atomic operations

59. Introduction to atomic operations
- How to provide lock free function

60. Functionality of std::atomic_flag
- Very basic atomic operation
- Boolean
- clear() and test_and_set()
- Is only exception for lock_free mechanism
- Cannot be initialized using regular boolean
    - Use ATOMIC_FLAG_INIT
```cpp
#include<thread>
#include<iostream>
#include<atomic>
int main() {
  std::atomic_flag flag = ATOMIC_FLAG_INIT; // set as false
  std::cout << flag.test_and_set() << std::endl; // prints 0 as false. Then set as 1
  std::cout << flag.test_and_set() << std::endl; // now prints 1
  std::cout << flag.test_and_set() << std::endl; // still prints 1
  flag.clear();                                  // set as false (=0)
  std::cout << flag.test_and_set() << std::endl; // prints 0 as false then set as 1
  std::cout << flag.test_and_set() << std::endl; // prints 
  return 0;
}
```

61. Functionality of std::atomic_bool
- `atomic<*>`
    - Neither copy assignable nor copy constructible
    - Can assign atomic Booleans and can be construct using non-atomic booleans
- Functions
    - is_lock_free
    - store
    - load
    - exchange
    - Compare_exchange_weak
    - Compare_exchange_strong
```cpp
#include <iostream>
#include <thread>
#include <atomic>
int main() {
  std::atomic<bool> x(false);
  std::cout << "atomic boolean is lock free - " 
            << (x.is_lock_free() ? "yes" : "no") << std::endl;
  //std::atomic<bool> y(x); // not working. No copy constructor
  //std::atomic<bool> z=x;  // not working. No assign operator
  std::atomic<bool> y(true);
  x.store(false);
  x.store(y);
  std::cout << "y=" << y.load() << std::endl; // prints 1=true
  bool z = x.exchange(false);
  std::cout << "now x =" << x.load() << std::endl; // prints 0=false
  std::cout << "previous x =" << z << std::endl; // prints 1=true
  return 0;
}
```

62. Explanation of compare_exchange functions
- bool r = x.compare_exchange_weak(T& expected, T desired);
    - Comparing x with expected, desired is stored to x whey they are equal
    - When they are not equal, expected variable is updated with the value of x
    - r becomes true when they are equal. False otherwise
    - This might not be gauranteed (false even when they are equal)
- compare_exchange_strong() is gauranteed as it uses more instruction but more expensive
- They can be used as conditions for looping

63. Atomic pointers
- `atomic<T*>`
    - Does not mean that the object is pointed to is atomic
    - But pointer itself is atomic
- Extra functions
    - fetch_add: replaces the current value with the result of addition of the value and arg
    - fetch_sub
    - ++
    - --
```cpp
#include <iostream>
#include <thread>
#include <atomic>
int main() {
  int val[10] {11,22,33,44,55,66,77,88,99,0};
  std::atomic<int*> x_ptr = val;
  std::cout << "atomic boolean is lock free - " 
            << (x_ptr.is_lock_free() ? "yes" : "no") << std::endl; // yes
  int* y_ptr = val + 3;
  x_ptr.store(y_ptr);
  std::cout << "val by the pointer = " << *(x_ptr.load()) << std::endl; // 44
  bool rv = x_ptr.compare_exchange_weak(y_ptr, val+5); 
  std::cout << "store operation was : " << (rv?"yes":"no") << std::endl; // yes
  std::cout << "new val of x_ptr: " << *x_ptr << std::endl; // 66
  int* pre_ptr1 = x_ptr.fetch_add(2); // adding 2 as offset (index)
  std::cout << *pre_ptr1 << " " << *x_ptr << std::endl; //66 & 88
  int* pre_ptr2 = x_ptr.fetch_sub(4); // -4 for offset
  std::cout << *pre_ptr2 << " " << *x_ptr << std::endl; // 88 & 44
  x_ptr++;  std::cout << *x_ptr << std::endl; // 55
  x_ptr--;  std::cout << *x_ptr << std::endl; // 44
  return 0;
}
```

64. General discussion on atomic types

65. Important relationships related to atomic operations b/w threads
- Important relationship
    - Happen-before
    - Inter-thread-happen-before
    - Synchronized-with

66. Introduction to memory ordering options
- memory_order_seq_cst
- memory_order_relaxed
- memory_order_acquire 
- memory_order_release
- memory_order_acq_rel
- memory_order_consume

67. Discussion on memory_order_seq_cst
- Simple sequential view

68. Introduction to instruction reordering

69. Discussion on memeory_order_relaxed
- Opposite of memory_order_seq_cst
- View of the threads does not need to be consistent to each other
- Inconsistency may happen due to caching, prefetching, ...

70. Discussion on memory_order_acquire and memory_order_release

71. Important aspects of memory_order_acquire and memory_order_release

72. Concept of transitive synchronization

73. Discussion on memory_order_consume

74. Concept of release sequence
- Aftera release operation A is performed on an atomic object M, the longest continuous subsequence of the modification on M that consists of 1) Writes performed by the same thread that performed A 2) atomic read-modify-write operations made to M by any thread is known as **release sequence headed** by A

75. Implementation of spin lock mutex

## Section 7: Lock free data structures and algorithms

76. Introduction and some terminology
- Blocking vs nonblocking
- Lock free vs wait free

77. Stack recap
- Lock free stack - avoiding race conditions in push()
```cpp
class lock_free_stack {
    ...
    std::atomic<node *> head;
public:
    void push(T const & data) {
        node * const new_node = new node(data);
        new_node->next = head.load();
        while(head.compare_exchange_weak(new_node->next, new_node));
    }
}
```

78. Simple lock free thread safe stack
- How to avoid race conditions in pop()
- Declare node data as shared_ptr
```cpp
std::shared_ptr<T>  pop() {
    node * old_head = head.load();
    while(old_head&& !head.compare_exchange_weak(old_head, 
                        old_head->next));
    return old_head? old_head->data: std::shared_ptr<T>();
}
```
- It has memory leak issue. See next section

79. Stack memory reclaim mechanism using thread counting
- Count the number of threads in the pop function
- If thread count is > 1, we cannot delete the nodes safely
- Book-keep the nodes to be deleted
- When thread count is 1, all nodes are safe to delete

80. Stack memory reclaim mechanism using hazard pointers

81. Stack memory reclaim mechanism using reference counting
- Using external/internal counters
    - External count is increased when the pointer is read
    - Internal count is decreased when a reader is finished with the node

## Section 8: Thread pools

82. Simple thread pool
- When you have work to do, call a function to put it on the queue of pending work. Each worker thread takes work off the queue, running the specified task and then going back to the queue for more work

83. Thread pool which allowed to wait on submitted tasks

84. Thread pool with waiting tasks

85. Minimizing contention on work queue
- To avoid cache pingpong, use a separate work queue per thread
    - Use uniq_ptr for the local work queue

86. Thread pool with work stealing

## Section 9: Bonus section: Parallel programming in massively parallel devices with CUDA

87. Setting up the environment for CUDA

88. Elements of CUDA program
```cpp
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>
__global__ void hello_cuda(){  printf("Hello CUDA world\n");}
int main() {
  dim3 block(4);
  dim3 grid(8);
  hello_cuda<<<grid,block>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```

89. Organization of threads in CUDA program 1

90. Organization of threads in CUDA program 2

91. Unique index calculation for threads in a grid

92. Unique index calculation for threads in a 2D grid

93. Unique index calculation for threads in a 2D grid 2

94. Timing a CUDA program

95. CUDA memory transfer

96. Sum array example

97. Error handling in a CUDA program

98. CUDA device properties

