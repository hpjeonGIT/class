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
};
Integer * GetPointer(int v) {
  Integer *p = new Integer{v};
  return p;
}
void fnc1(std::unique_ptr<Integer> p) { }
void fnc2(std::unique_ptr<Integer> &p) { }
int main() {
    std::unique_ptr<Integer> p{GetPointer(3)};
    p->SetValue(123);
    fnc1(std::move(p));// after func1(p), p is deallocated
    p.reset(new Integer{456}); // reallocate
    fnc2(p);// after func2(p), p is still allocated
    std::cout << p->GetValue() << std::endl;
    return 0;
}
