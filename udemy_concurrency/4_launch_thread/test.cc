#include <iostream>
#include <thread>

void foo() {
    std::cout << "hello from foo\n" << std::this_thread::get_id() << std::endl;
}
class callable_class {
    public:
    void operator() (){
        std::cout << "Hello from class with function call operator\n"
        << std::this_thread::get_id() << std::endl;
    };
};    
void run() {
    std::thread thread1(foo);
    callable_class obj;
    std::thread thread2(obj);
    std::thread thread3([] { 
        std::cout << "hello from lambda\n"
        << std::this_thread::get_id() << std::endl;});
    thread1.join();
    thread2.join();
    thread3.join();
    std::cout << "Hello from main\n";

}

int main() {
    run();
    return 0;
}