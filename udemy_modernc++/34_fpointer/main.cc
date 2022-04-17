#include <iostream>
void Prnt(int n, char c) {
    for (int i=0;i<n;i++) std::cout << c;
}
int main() {
    void (*fPnt)(int,char) = Prnt;
    (*fPnt)(3,'#');
    fPnt(5,'!');
    return 0;
}