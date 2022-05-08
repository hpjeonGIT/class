#include <iostream>
#include <functional>
#include <future>
#include <string>
#include <stdexcept>
#include <chrono>
void print_int(std::future<int>& fut) {
    std::cout << "waiting for value from print thread\n";
    std::cout << "value= " << fut.get() << std::endl;
}
int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    std::thread print_thread(print_int, std::ref(fut));
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    prom.set_value(10);
    std::cout << "setting the value in the main thread\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    print_thread.join();
    return 0;
}