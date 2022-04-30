#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
class Employee {
  int m_Id;
  std::string m_Name;
  std::string m_pLang;
public:
  Employee(const int i, const std::string &Name, const std::string &Lang) 
  : m_Id(i), m_Name(Name), m_pLang(Lang) { }
  ~Employee() = default;
  const std::string & GetName() const {
    return m_Name;
  }
  const std::string & GetProgramLang() const {
    return m_pLang;
  }
  const int GetId() const {
    return m_Id;
  }
};
int main() {
  std::vector<Employee> emp { Employee {101, "John", "C++"}, 
                              Employee {201, "Amy", "Java"},
                              Employee {301, "Bob", "C++"} };
  // sorting by Name
  std::sort(emp.begin(), emp.end(), [] (const auto &e1, 
                                        const auto &e2) {
    return e1.GetName() < e2.GetName(); }); 
  for (const auto &el : emp) {
    std::cout << el.GetId() << std::endl;
  }
  // Count how many C++
  auto cppCount = std::count_if(emp.begin(), emp.end(), [](const auto &el) {
    return el.GetProgramLang() == "C++";  }  );
  std::cout << "C++ users = " << cppCount << std::endl;
  // Print user id for C++
  std::for_each(emp.begin(), emp.end(), [](const auto &el) {
    if (el.GetProgramLang() == "C++") 
      std::cout << "C++ user id= " << el.GetId() << std::endl;
  });
  return 0;
}