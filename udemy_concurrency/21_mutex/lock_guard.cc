#include <iostream>
#include <mutex>
#include <list>
#include <thread>
std::list<int> my_list;
std::mutex m;
void add_to_list(int const& x) {
    std::lock_guard<std::mutex> lg(m);
    my_list.push_front(x);
}
void size() {
    m.lock();
    int size = my_list.size();
    m.unlock();
    std::cout << "size of the list is: " << size << std::endl;
}
int main() {
    std::thread t1(add_to_list,4);
    std::thread t2(add_to_list,11);
    t1.join();
    t2.join();
    return 0;
}