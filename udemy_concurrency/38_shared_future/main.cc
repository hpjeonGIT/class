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