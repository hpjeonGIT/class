#pragma once
class Implementor {
public:
  virtual void OperationImpl() = 0;
  virtual ~Implementor() = default;
};
class ConcreteImplementorA: public Implementor {
public:
  void OperationImpl() override;
};
class ConcreteImplementorB: public Implementor {
public:
  void OperationImpl() override;
};
class Abstraction {
protected:
  Implementor *m_pImplementor{};
public:
  explicit Abstraction(Implementor* m_p_implementor)
     : m_pImplementor(m_p_implementor) {}
  virtual void Operation() = 0;
  virtual ~Abstraction() = default;
};
class RefinedAbstraction: public Abstraction {
  using Abstraction::Abstraction;
public:
  void Operation() override;
};