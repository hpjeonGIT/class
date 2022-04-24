#include <iostream>
#include <string>
class Employee {
  std::string Name;
  int Id;
  int Salary;
public:
  Employee() = default;
  Employee(std::string name, int id, int salary) 
    : Name(name), Id(id), Salary(salary) { 
      std::cout << Name << Id << Salary << std::endl;
    }
  ~Employee() = default;
};
class Contact {
  std::string Name;
  long int PhoneNumber;
  std::string Address;
  std::string Email;
public:
  Contact() = default;
  Contact(std::string name, long int pn, std::string address, std::string email) 
  : Name(name), PhoneNumber(pn), Address(address), Email(email) {
    std::cout << Name << PhoneNumber << Address << Email << std::endl;
  }
  ~Contact() = default;
};
template<typename T, typename...Params> 
T* CreateObject(Params... args) {
  T* obj = new T(args...);
  return obj;
}
int main() {
  int *p1 = CreateObject<int>(5);
  std::string *s = CreateObject<std::string>();
  Employee *emp = CreateObject<Employee>("Bob", 101, 1000);
  Contact *p = CreateObject<Contact>("Joey", 123456789, "One street MA", "myEmail@altavista.com");
  return 0;
}
