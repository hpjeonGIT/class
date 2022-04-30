#include <iostream>
#include <string>
#include <map>
#include <vector>
struct Employee {
  int Id;
  std::string Name;
  Employee() = default;
  Employee(int i, std::string name) : Id(i), Name(name) {}
};
int main() {
  // passing by value
  Employee emp{ 123, "John"};
  auto [n, txt] = emp;
  std::cout << n << " " << txt << std::endl;
  // passing by reference  
  auto &[i, name] = emp;
  i++; name += " is the firstname";
  std::cout << i << " " << name << std::endl;
  // mapping of key vs value
  std::map<std::string, std::string> groupdata {
    {"firstK", "Piano"}, {"secondK", "Guitar"}
  };
  for (auto &[key, value] : groupdata) {
    std::cout << key << " " << value << std::endl;
  }
  // 
  //std::vector<int> myV {11, 33, 22}; not working for structured bindings
  int myV[] {11,22,33};
  auto [a,b,c] = myV;
  std::cout << a << " " << b << " " << c << std::endl;
  return 0;
}