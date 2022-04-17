#include <future>
#include <iostream>
#include <thread>
#include <chrono>
void func1() {
    for (auto &x : {1,2,3,4,5,6,7,8,9}) {
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}

int main() {
    std::future<void> result = std::async(std::launch::async, func1);
    std::cout << "downloading\n" ;
    result.get();
}