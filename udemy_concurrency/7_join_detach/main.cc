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