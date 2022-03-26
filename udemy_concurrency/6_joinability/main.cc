#include <thread>
#include <iostream>
void test() {
    std::cout << "hello \n";
}
int main() {
    std::thread thread1(test);
    std::cout << "thread1 is joinable or not:" << thread1.joinable() << std::endl;
    thread1.join();
    std::cout << "thread1 is joinable or not:" << thread1.joinable() << std::endl;
    std::thread thread2;
    std::cout << "thread2 is joinable or not:" << thread2.joinable() << std::endl;
    return 0;

}