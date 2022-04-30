#include <iostream>
#include <string>
#include <any>
struct Employee {
  int Id;
  std::string Name;
  Employee() = default;
  Employee(int i, std::string name) : Id(i), Name(name) {}
  ~Employee() { std::cout << "Destructed\n";}
};
int main() {
  std::any v = 5;
  std::cout << std::any_cast<int>(v) << std::endl;
  //v = "Hello"; // this will produce bad_any_cast in the any_cast<> as this is char *
  v = std::string("Hello");
  std::cout << std::any_cast<std::string>(v) <<std::endl;
  v = Employee(432,"J Johnson");
  v.reset();
  v = 5;
  auto &v2 = std::any_cast<int&> (v);
  v2 = 100;
  std::cout << std::any_cast<int> (v) << std::endl;
  return 0;
}