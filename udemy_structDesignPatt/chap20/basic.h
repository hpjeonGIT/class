#pragma once
#include <iostream>
#include <memory>
class A {
public:
  void CallA() { std::cout<<"Called A\n";}
};
class B {
public:
  void CallB() {std::cout<<"Called B\n";}
};
class C {
public:
  void CallC() {std::cout<<"Called C\n";}
};
class Client {
  std::shared_ptr<A> m_pA;
  std::shared_ptr<B> m_pB;
  std::shared_ptr<C> m_pC;
public:
  Client();
  ~Client() = default;
  void Invoke();
};