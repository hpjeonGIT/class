#pragma once
#include <vector>
class SharedObject {
  bool m_IsUsed{true};
  int m_ID;
public:
  bool IsUsed() const {return m_IsUsed;}
  void SetUsedState(bool used) {m_IsUsed = used;}
  void MethodA();
  void MethodB();
  void Reset();
  void SetID(int i_) {m_ID = i_;}
  int GetID() {return m_ID;}
};
class ObjectPool {
  ObjectPool() = default;
  inline static std::vector<SharedObject*> m_PooledObjects{};
public:
  static SharedObject * AcquireObject();
  static void ReleaseObject(SharedObject *pSO);
  static int ncount;
};