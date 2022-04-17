#include<iostream>
namespace Avg {
    float calc(float x, float y) { return (x+y)/2.f;}
}
namespace Add {
    float calc(float x, float y) { return (x+y);}
}
int main(){
    float x=1.0f, y=2.3f;
    std::cout << Avg::calc(x,y) << std::endl;
    std::cout << Add::calc(x,y) << std::endl;
}