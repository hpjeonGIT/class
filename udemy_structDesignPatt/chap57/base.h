#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
class Component {
public:
  virtual void Operation() = 0;
  virtual void Add(Component *pComponent) = 0;
  virtual void Remove(Component *pComponent) = 0;
  virtual Component * GetChild(int index) = 0;
  virtual ~Component() = default;
};
class Leaf: public Component {
public:
  void Add(Component* pComponent) override;
  Component* GetChild(int index) override;
  void Operation() override;
  void Remove(Component* pComponent) override;
};
class Composite : public Component {
  std::vector<Component*> m_Children {};
public:
  void Add(Component* pComponent) override;
  Component* GetChild(int index) override;
  void Operation() override;
  void Remove(Component* pComponent) override;
};