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