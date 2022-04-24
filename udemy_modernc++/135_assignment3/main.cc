#include <iostream>
#include <string>
template<typename T>
T ArraySum(T *pArr, size_t arrSize) {
  T local_sum {};
  for(size_t i=0; i< arrSize; ++i) {
    local_sum += pArr[i];
  }
  return local_sum;
}
template<typename T, size_t arrSize>
T ArraySum(T (&pArr)[arrSize]) {
  T local_sum {};
  for(size_t i=0; i< arrSize; ++i) {
    local_sum += pArr[i];
    std::cout << "pArr = " << pArr[i] << std::endl;
  }
  return local_sum;
}
template<> const char ArraySum(const char *pArr, size_t arrSize) {
  char* local_sum = new char[arrSize];
  for(size_t i=0; i< arrSize; ++i) {
    local_sum[i] = pArr[i];
  }
  return *local_sum;
}
template<> std::string ArraySum(std::string *pArr, size_t arrSize) {
  std::string msg {};
  std::cout << pArr << std::endl;
  return pArr[0];
}

int main() {
  const size_t sArr = 3;
  int iArr[] {1,5,2};
  float fArr[] {1.f,5.f,2.f};  
  char cArr[] {'a','b','c'};
  const char *pcArr {"abc"};
  std::string strArr {"abc"};
  std::cout << ArraySum(iArr) << std::endl;
  std::cout << ArraySum(fArr) << std::endl;
  std::cout << ArraySum(cArr) << std::endl; // not working correctly
  std::cout << ArraySum(pcArr,sArr) << std::endl;
  std::cout << ArraySum(&strArr,sArr) << std::endl;
  return 0;
}
