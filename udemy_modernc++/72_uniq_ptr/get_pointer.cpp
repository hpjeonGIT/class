#include <iostream>
#include <memory>
class Integer{
private:
    int x;
public:
    Integer() {	x = 0;std::cout << "default constructor\n";}
    ~Integer() {std::cout << "default destructor\n";}
    Integer(int value) { x = value; std::cout << "param constructor\n";}
    Integer(const Integer &obj) {x = obj.x; std::cout << "copy constructor \n";}
    Integer(Integer &&obj) {x = obj.x; std::cout << "move constructor \n";}
    Integer &operator=(const Integer &obj) {x = obj.x;	std::cout << "copy assign operator\n";}
    Integer &operator=(Integer && obj) {x = obj.x;std::cout << "move assign operator\n";}
    int GetValue() {return x;}
    void SetValue(int value) { x = value;}
    explicit operator int() { return x; }
    Integer *GetPointer() {return this;}
};
int main() {
    Integer *i1 = new Integer{123};
    std::shared_ptr<Integer> p{i1->GetPointer()};
    std::cout << p->GetValue() << std::endl;
    return 0;
}
