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