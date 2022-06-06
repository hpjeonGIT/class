#pragma once
#include <vector>
#include <memory>
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
using SharedObjectUniqPtr = std::unique_ptr<SharedObject>;
using SharedObjectSharedPtr = std::shared_ptr<SharedObject>;
class ObjectPool {
  ObjectPool() = default;
  //inline static std::vector<SharedObjectUniqPtr> m_PooledObjects{};
  inline static std::vector<SharedObjectSharedPtr> m_PooledObjects{};
public:
  //static SharedObjectUniqPtr AcquireObject();
  //static void ReleaseObject(SharedObjectUniqPtr pSO);
  static SharedObjectSharedPtr AcquireObject();
  static void ReleaseObject(SharedObjectSharedPtr pSO);
  static int ncount;
};