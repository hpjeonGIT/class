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
    Integer *tmp = new Integer {v};
    return tmp;
}
class Employee{
    std::shared_ptr<Integer> m_pInteger{};
public:
    void SetInteger(std::shared_ptr<Integer> &i1) {
        m_pInteger = i1;
    }
    const std::shared_ptr<Integer>& GetInteger() const {
        return m_pInteger;
    }
};
int main() {
    std::shared_ptr<Integer> p{GetPointer(3)};
    p->SetValue(123); std::cout << p.use_count() << std::endl;    
    std::shared_ptr<Employee> e1 {new Employee{}};
    e1->SetInteger(p); std::cout << p.use_count() << std::endl;
    std::cout << "Employee " << e1->GetInteger()->GetValue() << std::endl;
    std::shared_ptr<Employee> e2 {new Employee{}};
    e2->SetInteger(p); std::cout << p.use_count() << std::endl;
    p->SetValue(456);
    std::cout << "Employee " << e2->GetInteger()->GetValue() << std::endl;
    return 0;
}
