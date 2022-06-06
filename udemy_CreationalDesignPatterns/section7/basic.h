#pragma once
class Product {
};
class Builder {
public:
  virtual void BuildPart() = 0;
  virtual ~Builder() = default;
};
class ConcreteBuilder: public Builder {
  Product *product;
public:
  ConcreteBuilder();
  void BuildPart() override;
  Product * GetResult();
};
class Director {
  Builder *builder;
public:
  Director(Builder* builder);
  void Construct();
};