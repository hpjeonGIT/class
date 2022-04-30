#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
class Contact {
  std::string m_firstName, m_lastName, 
       m_primPhoneNumber, m_secPhoneNumber,
       m_emailId, m_address, m_company;
  std::unordered_map<std::string,std::string> m_group;
public:
  Contact() = default;
  Contact(std::string firstName, std::string lastName,
          std::string primPN,    std::string secPN, 
          std::string emailId,   std::string address,
          std::string company)
          : m_firstName(firstName), m_lastName(lastName), 
       m_primPhoneNumber(primPN), m_secPhoneNumber(secPN),
       m_emailId(emailId), m_address(address), m_company(company) {}
  ~Contact() = default;
  const std::string GetFName() const { return m_firstName; }
  const std::string GetLName() const { return m_lastName; }
  const std::string GetPrimaryPhoneNumber() const { return m_primPhoneNumber; }
  const std::string GetCompany() const { return m_company; }
};
int main() {
  std::vector<Contact> myV {
    Contact {"Clark", "Kent", "0123", "456", "abc@alpha", "1 street", "IBM"},
    Contact {"James", "Khan", "314", "753", "xyz@beta", "2 street", "AWS"},
    Contact {"Shaun", "Dave", "789", "135", "pi@alpha", "3 street", "IBM"} 
    };
  // sorting
  std::sort(myV.begin(), myV.end(), [] (const auto & e1, const auto &e2) { 
    return e1.GetFName() < e2.GetFName();});
  for (auto &el : myV) std::cout << el.GetFName() << std::endl;
  std::sort(myV.begin(), myV.end(), [] (const auto & e1, const auto &e2) { 
    return e1.GetLName() < e2.GetLName();});
  for (auto &el : myV) std::cout << el.GetLName() << std::endl;
  // display
  std::for_each(myV.begin(), myV.end(), [](const auto &el) {
    std::cout << el.GetFName() << " " << el.GetPrimaryPhoneNumber() << std::endl;
  });
  // Find same company only
  std::for_each(myV.begin(), myV.end(), [] (const auto &el) {
    if (el.GetCompany().compare("IBM") == 0) std::cout << el.GetFName() << " " 
      << el.GetLName() << " " << el.GetCompany() << std::endl;
  });
  return 0;
}