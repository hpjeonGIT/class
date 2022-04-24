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
    std::weak_ptr<Integer> m_pInteger{};
public:
    void SetInteger(std::weak_ptr<Integer> i1) {
        m_pInteger = i1;
    }
    const std::weak_ptr<Integer>& GetInteger() const {
        return m_pInteger;
    }
    void PrintStatus() {
        if (m_pInteger.expired()) {
            std::cout << " no longer available \n"; return;
        }
        std::cout << "At employee " <<m_pInteger.lock().use_count() << std::endl;
    }
};
int main() {
    std::shared_ptr<Integer> p{GetPointer(3)};
    p->SetValue(123); std::cout << p.use_count() << std::endl;    
    std::shared_ptr<Employee> e1 {new Employee{}};
    e1->SetInteger(p); std::cout << p.use_count() << std::endl;
    e1->PrintStatus();
    std::shared_ptr<Employee> e2 {new Employee{}};
    e2->SetInteger(p); std::cout << p.use_count() << std::endl;
    p->SetValue(456);
    e2->PrintStatus();
    p = nullptr; std::cout << p.use_count() << std::endl;
    e1->PrintStatus(); e2->PrintStatus();
    return 0;
}
