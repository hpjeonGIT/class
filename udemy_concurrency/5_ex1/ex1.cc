#include <iostream>
#include <thread>

void test() {
    std::cout << "hello from test\n";
}
void functionA() {
    std::thread threadC(test);
    threadC.join();
    std::cout << "hello from funciton A\n";
}
void functionB() {
    std::cout << "hello from function B\n";
}

int main() {
    std::thread threadA(functionA);
    threadA.join();
    std::thread threadB(functionB);
    threadB.join();
    return 0;
}