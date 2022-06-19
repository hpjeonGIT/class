#pragma once
class Component {
public:
  virtual void Operation() = 0;
  virtual ~Component() = default;
};
class ConcreteComponent: public Component {
public:
  void Operation() override;
};
class ConcreteDecoratorA: public Component {
    Component *m_ptr{};
public:
  ConcreteDecoratorA(Component * ptr) : m_ptr{ptr} {}
  void Operation() override;
};
class ConcreteDecoratorB: public Component {
    Component *m_ptr{};
public:
  ConcreteDecoratorB(Component * ptr) : m_ptr{ptr} {}
  void Operation() override;
};