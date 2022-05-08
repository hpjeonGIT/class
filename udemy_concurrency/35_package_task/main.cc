#include <iostream>
#include <future>
#include <string>
#include <chrono>
int add(int x, int y) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout <<"addition runs on thread-" << std::this_thread::get_id() << std::endl;
    return x+y;
}
void task_thread() {
    std::packaged_task<int(int,int)> task1(add);
    std::future<int> f1 = task1.get_future();
    std::thread t1(std::move(task1),13,14);
    t1.detach();
    std::cout << "task1 thread = " << f1.get() << std::endl;
}
void task_normal() {
    std::packaged_task<int(int,int)> task1(add);
    std::future<int> f1 = task1.get_future();
    task1(3,4);
    std::cout << "task1 normal = " << f1.get() << std::endl;
}
int main() {
    std::cout <<"main thread-" << std::this_thread::get_id() << std::endl;
    task_thread();
    task_normal();
    return 0;
}