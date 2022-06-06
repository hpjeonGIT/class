#pragma once
#include <iostream>
class AbstractProductA {
public:
  virtual void ProductA() = 0;
  virtual ~AbstractProductA() = default;
};
class AbstractProductB {
public:
  virtual void ProductB() = 0;
  virtual ~AbstractProductB() = default;
};
class ProductA1 : public AbstractProductA {
public:
  void ProductA() override { std::cout << "[1] Product A\n"; }
};
class ProductB1 : public AbstractProductB {
public:
  void ProductB() override { std::cout << "[1] Product B\n"; }
};
class ProductA2 : public AbstractProductA {
public:
  void ProductA() override { std::cout << "[2] Product A\n"; }
};
class ProductB2 : public AbstractProductB {
public:
  void ProductB() override { std::cout << "[2] Product B\n"; }
};