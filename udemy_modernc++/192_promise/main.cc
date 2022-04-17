#include <future>
#include <iostream>
#include <thread>
#include <chrono>
int func1(std::promise<int> &inp) {
    auto f = inp.get_future();
    std::cout << "waiting for inp\n";
    auto count = f.get();
    std::cout << "inp received\n";
    int sum{};
    for (int i=0;i<count;i++) {
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        sum +=i;
    }
    return sum;
}
int main() {
    std::promise<int> inp;
    std::future<int> result = std::async(std::launch::async, func1, std::ref(inp));
    std::cout << "downloading\n" ;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    inp.set_value(10);
    if (result.valid()) {
        auto sum = result.get();
        std::cout << sum << std::endl;
    }
}