#pragma once
class Target {
public:
  virtual void Request() = 0;
  virtual ~Target() = default;
};
class Adaptee {
public:
  void SpecificRequest();
};
class Adapter: public Target, public Adaptee {
  Adaptee m_Adaptee;
public:
  void Request() override;
};