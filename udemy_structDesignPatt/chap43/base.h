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
class Decorator: public Component {
  Component * m_ptr{};
public:
  Decorator(Component * ptr) : m_ptr {ptr} {}
  void Operation() override;
};
class ConcreteDecoratorA: public Decorator {
  using Decorator::Decorator;
public:
};
class ConcreteDecoratorB: public Decorator {
  using Decorator::Decorator;
public:
};
