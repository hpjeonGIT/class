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