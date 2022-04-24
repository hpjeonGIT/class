#include <iostream>
template<typename T>
T Add(T a, T b) {
  return a+b;
}
template<typename T>
T ArraySum(const T *pArr, size_t arrSize) {
  T local_sum {};
  for(size_t i=0; i< arrSize; ++i) {
    local_sum += pArr[i];
  }
  return local_sum;
}
template<typename T>
T Max(const T *pArr, size_t arrSize) {
  T local_max {pArr[0]};
  for (size_t i=1; i< arrSize; ++i) {
    local_max = local_max > pArr[i] ? local_max : pArr[i];
  }
  return local_max;
}
template<typename T>
std::pair<T,T> MinMax(const T *pArr, size_t arrSize) {
  T local_max {pArr[0]};
  T local_min {pArr[0]};
  for (size_t i=1; i< arrSize; ++i) {
    local_max = local_max > pArr[i] ? local_max : pArr[i];
    local_min = local_min < pArr[i] ? local_min : pArr[i];
  }
  return std::make_pair(local_min,local_max);
}
int main() {
  std::cout << Add(1,3) << std::endl;
  std::cout << Add(1.f,3.f) << std::endl;
  size_t sArr = 3;
  int* iArr = new int[sArr];
  iArr[0] = 1; iArr[1] = 5; iArr[2] = 2;
  float* fArr = new float[sArr];
  fArr[0] = 1.f; fArr[1] = 5.f; fArr[2] = 2.f;
  std::cout << ArraySum(iArr, sArr) << std::endl;
  std::cout << ArraySum(fArr, sArr) << std::endl;
  std::cout << Max(iArr, sArr) << std::endl;
  std::cout << Max(fArr, sArr) << std::endl;
  auto p1 = MinMax(iArr, sArr);
  auto p2 = MinMax(fArr, sArr);
  std::cout << p1.first << " " << p1.second << std::endl;
  std::cout << p2.first << " " << p2.second << std::endl;
  delete[] iArr;
  delete[] fArr;
  return 0;
}
