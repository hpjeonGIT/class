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
    std::cout << "t1=" << t1.get_id() << std::endl;
    std::thread t2 = std::move(t1);
    //t1 = std::thread(bar);
    std::cout << "t1=" << t1.get_id() << std::endl;
    std::thread t3(foo);
    t1 = std::move(t3);
    std::cout << "t1=" << t1.get_id() << " " << t1.hardware_concurrency() << std::endl;
    t1.join();
    t2.join();
   std::cout << "t1=" << t1.get_id() << std::endl;
     //t3.join();
    return 0;
}
