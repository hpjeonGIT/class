#include <iostream>
class Integer{
private:
    int x;
public:
    Integer() {	x = 0;std::cout << "default constructor\n";}
    ~Integer() {std::cout << "default destructor\n";}
    Integer(int value) { x = value; std::cout << "param constructor\n";}
    Integer(const Integer &obj) {x = obj.x; std::cout << "copy constructor \n";}
    //Integer(const Integer &obj) = default;
    Integer(Integer &&obj) {x = obj.x; std::cout << "move constructor \n";}
    Integer &operator=(const Integer &obj) {x = obj.x;
	std::cout << "copy assign operator\n";}
    Integer &operator=(Integer && obj) {x = obj.x;std::cout << "move assign operator\n";}
    int GetValue() {return x;}
    void SetValue(int value) { x = value;}
    explicit operator int() { return x; }
};

Integer Add(const int a, const int b) {
    return Integer(a+b);
}
int main() {
    Integer i1(123),  i3;
    i3 = i1;
    Integer i4 {i1};
    i3 = Add(1,2);
    Integer i5(std::move(i3));
    Integer i6 = 3;
    //int x = i6;
    int x = static_cast<int>(i6);
    return 0;
}
