#include <iostream>
#include <thread>
void cleaning() {
    std::cout << "now cleaning ... \n";
}
void fullspeed() {
    std::cout << "now going full speed ... \n";
}
void stop_ship() {
    std::cout << "now stops the ship\n";
}
int main() {
    int n;
    bool NotDone = true;
    std::queue<int> q_work, q_clean;
    while (NotDone) {
        std::cin >> n;
        if (n==100) NotDone = false;
        else if (n==1) {
            std::thread t_ccrew(cleaning);
            t_ccrew.detach();
        } else if (n==2) {
            std::thread t_engine(fullspeed);
            t_engine.join();
        } else if (n==3) {
            std::thread t_engine(stop_ship);
            t_engine.join();
        } else 
            std::cout << "invalid order \n";
        
    }
    return 0;
}