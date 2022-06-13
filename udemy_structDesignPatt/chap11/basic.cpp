#include "basic.h"
#include <iostream>
void Client(Target *pTarget) {
  pTarget -> Request();
}
void Adaptee::SpecificRequest(){
  std::cout << "[Adaptee] SpecificRequest\n"  ;
}
void Adapter::Request() {
  std::cout << "[Adapter] Calling SpeficRequest\n" ;
  //m_Adaptee.SpecificRequest();
  SpecificRequest();
}
int main() {
  Adapter a;
  Client(&a);
}