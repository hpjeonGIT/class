#include<iostream>
/*
void Prnt(int x) {
    std::cout << "Prnt " << x << std::endl;
}
*/
void Prnt(const int& x ){
    std::cout << "Prnt const int &x " << x << std::endl;
}
void Prnt(const int &&x) {
    std::cout << "Prnt const int&&x" << x << std::endl;
}
int main(){
    int x = 1;
    Prnt(x);
    Prnt(1);
}