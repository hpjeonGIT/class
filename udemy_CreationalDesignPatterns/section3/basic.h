#pragma once
class Product {
public:
  virtual void Operation() = 0;
  virtual ~Product() = default;
};
class ConcreteProduct: public Product {
  void Operation();
};
class ConcreteProduct2: public Product {
  void Operation();
};
class Creator {
  Product *m_pProduct;
public:
  void AnOperation();
  virtual Product * Create() {return nullptr; }
};
class ConcreteCreator: public Creator {
public:
  Product * Create() override;  
};
class ConcreteCreator2: public Creator {
public:
  Product * Create() override;  
};