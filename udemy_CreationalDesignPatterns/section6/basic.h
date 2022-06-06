#pragma
class Prototype {
public:
  virtual Prototype * Clone() = 0;
  virtual ~Prototype() = default;
};
class ConcretePrototype1: public Prototype {
public:
  Prototype* Clone() override;
};
class ConcretePrototype2: public Prototype {
public:
  Prototype* Clone() override;
};
class Client {
  Prototype *prototype;
public:
  void SetPrototype(Prototype *p) {
    prototype = p;
  }
  void Operation();
};