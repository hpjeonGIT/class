#include "basic.h"
#include <iostream>
void RealSubject::Request() {
  std::cout << "[RealSubject] Request processed\n";
}
void Proxy::Request() {
  if (m_pSubject == nullptr) {
    std::cout << "[Proxy] Creating RealSubject\n";
    m_pSubject = std::make_shared<RealSubject>();
  }
  std::cout << "[Proxy] Additional behavior\n";
  m_pSubject -> Request();
}
void Operate (std::shared_ptr<Subject> s) {
  s->Request();
}
int main() {
  //auto sub = std::make_shared<RealSubject>();
  auto sub = std::make_shared<Proxy>();
  Operate(sub);
  return 0;
}