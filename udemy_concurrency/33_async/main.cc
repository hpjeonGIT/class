#include <iostream>
#include <future>
#include <string>
void printing() {
    std::cout <<"printing runs on thread-" << std::this_thread::get_id() << std::endl;
}
int addition(int x, int y) {
    std::cout <<"addition runs on thread-" << std::this_thread::get_id() << std::endl;
    return x+y;
}
int subtract(int x, int y) {
    std::cout <<"subtract runs on thread-" << std::this_thread::get_id() << std::endl;
    return x - y;
}
int main() {
    std::cout <<"main thread-" << std::this_thread::get_id() << std::endl;
    int x = 100;
    int y = 50;
    std::future<void> f1 = std::async(std::launch::async, printing);
    std::future<int>  f2 = std::async(std::launch::deferred, addition,x,y);
    std::future<int>  f3 = std::async(std::launch::deferred | std::launch::async, subtract,x,y);
    f1.get();
    std::cout << "value from f2 future: " << f2.get() << std::endl;
    std::cout << "value from f3 future: " << f3.get() << std::endl;
    return 0;
}