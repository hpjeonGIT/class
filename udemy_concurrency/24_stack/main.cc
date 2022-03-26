#include <iostream>
#include <thread>
#include <mutex>
#include <stack>
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