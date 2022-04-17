#include <iostream>
#include <thread>

int main() {
    std::cout << "hello world " << std::this_thread::get_id() << std::endl;
}