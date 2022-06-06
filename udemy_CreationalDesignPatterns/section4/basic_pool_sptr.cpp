#include "basic_pool_sptr.h"
#include <iostream>
void SharedObject::MethodA() { std::cout << "MethodA ID=" << GetID() << " \n"; }
void SharedObject::MethodB() { std::cout << "MethodB\n"; }
void SharedObject::Reset() { std::cout << "Resetting the state\n"; }
//SharedObjectUniqPtr ObjectPool::AcquireObject() {
SharedObjectSharedPtr ObjectPool::AcquireObject() {
  for(auto & so: m_PooledObjects) {
    if (!so->IsUsed()) {
      std::cout << "[POOL] Returning an existing object\n";
      so->SetUsedState(true);
      so->Reset();
      return so;
    }
  }
  std::cout << "[POOL] Creating a new object\n";
  ncount ++;
  //auto so = std::make_unique<SharedObject> ();
  auto so = std::make_shared<SharedObject> ();
  so->SetID(ncount);
  m_PooledObjects.push_back(so);
  return so;
}
//void ObjectPool::ReleaseObject(SharedObjectUniqPtr pSO) {
void ObjectPool::ReleaseObject(SharedObjectSharedPtr pSO) {
  for (auto &so: m_PooledObjects) {
    if (so == pSO) {
      so->SetUsedState(false);
    }
  }
}
int ObjectPool::ncount = 0;
int main() {
  auto s1 = ObjectPool::AcquireObject();
  s1->MethodA();
  s1->MethodB();
  auto s2 = ObjectPool::AcquireObject();
  s2->MethodA();
  s2->MethodB();
  ObjectPool::ReleaseObject(s1);
  auto s3 = ObjectPool::AcquireObject();
  s3->MethodA();
  s3->MethodB();
  return 0;
}