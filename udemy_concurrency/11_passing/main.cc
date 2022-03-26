#include <iostream>
#include <thread>
void func2(const int& x) {
    while(true) {
        try {
            std::cout << x << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        } catch (...) {
            throw std::runtime_error("runtime error");
        }
    }
}
void func3(const int x) {
    while(true) {
        try {
            std::cout << x << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        } catch (...) {
            throw std::runtime_error("runtime error");
        }
    }
}

void func1() {
    int x=5;
    std::thread t2(func2, std::ref(x));
    //std::thread t2(func3, x);
    t2.detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    std::cout << "thread1 finished \n";
}
int main() {
    std::thread t1(func1);
    t1.join();
    return 0;
}