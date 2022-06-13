#pragma once
#include <memory>
class Subject {
public:
  virtual void Request() = 0;
  virtual ~Subject() = default;
};
class RealSubject: public Subject { 
public:
  void Request();
};
class Proxy : public Subject{
  std::shared_ptr<RealSubject> m_pSubject {}; //default is nullptr
public:
  void Request() override;
  Proxy() = default;
  ~Proxy() = default;
};