#include <future>
#include <iostream>
#include <thread>
#include <chrono>
int func1(const int&& count) {
    int sum{};
    for (int i=0;i<count;i++) {
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        sum +=i;
    }
    return sum;
}
int main() {
    std::future<int> result = std::async(std::launch::async, func1, 10);
    std::cout << "downloading\n" ;
    if (result.valid()) {
        auto sum = result.get();
        std::cout << sum << std::endl;
    }
}