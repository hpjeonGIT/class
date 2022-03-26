#include <iostream>
#include <thread>
void func1(const int x, const int y) {
    std::cout << "X + Y = " << x + y << std::endl;
}
void func2(const int& x, const int& y) {
    for (int i=0;i<5;i++) {
        std::cout << "X + Y = " << x + y << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
int main() {
    int p=9;
    int q=8;
    std::thread t1(func1, p, q);
    t1.join();
    std::thread t2(func2, std::ref(p), std::ref(q));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    p += 10;
    q += 10;
    t2.join();
    return 0;
}