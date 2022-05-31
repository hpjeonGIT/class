#include <iostream>
#include "Singleton.h"
Singleton Singleton::m_Instance;
Singleton& Singleton::Instance() { return m_Instance; }
void Singleton::MethodA() { }
int main() {
  Singleton &s = Singleton::Instance();
  s.MethodA();
  //Singleton s2; // not compiled
}