#pragma once
#include <unordered_map>
class Flyweight {
public:
  virtual void Operation(int extrinsic) = 0;
  virtual ~Flyweight() = default;
};
class ConcreteFlyweight: public Flyweight {
  int *m_pIntrinsicState{};
public:
  ConcreteFlyweight(int* mPIntrinsicState)
  : m_pIntrinsicState(mPIntrinsicState){}
  void Operation(int extrinsic) override;
};
class UnsharedConcreteFlyweight: public Flyweight {
  int m_InternalState{};
public:
  UnsharedConcreteFlyweight(int m_InternalState)
  : m_InternalState(m_InternalState){}
  void Operation(int extrinsic) override;
};
class FlyweightFactory {
  inline static std::unordered_map<int, Flyweight *> m_Flyweights{};
public:
  Flyweight* GetFlyweight(int key);
  Flyweight* GetUnsharedFlyweight(int value);
};