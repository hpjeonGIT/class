#include <iostream>
#include <vector>
#include <iterator>
template<typename T> struct fnctr1 {
  //T Max() {}; // No constructor necessary
  T operator() (T a, T b) const {
    return a > b ? a : b;
  }
};
template<typename T> struct fnctr2 {
  bool operator() (T a, T b) const {
    return a > b ? true : false;
  }
};
template<typename T> struct fnctr3 {
  bool operator() (T a, T b) const {
    return a > b ? false : true ;
  }
};
template<typename T> struct fnctr4 {
  std::pair<T,T> operator() (typename std::vector<T>::iterator begin, typename std::vector<T> end) {
    auto minX = begin;
    auto maxX = begin;
    for (auto it = begin; it != end; ++it) {
      if (*minX > *it) minX=it;
      if (*maxX < *it) maxX=it;
    }
    return std::make_pair(*minX, *maxX);
  }
};


int main() {
  // Assignment 1
  auto l_Max = [](auto a, auto b) {
    return a > b ? a : b;
  };
  auto l_Greater = [](auto a, auto b) {
    return a > b ? true : false;
  };
  auto l_Less = [](auto a, auto b) {
    return a > b ? false : true;
  };
  auto l_MinMax = [](auto a, auto b) {
    auto minX = a;
    auto maxX = a;
    for (auto it=a; it !=b; ++it) {
      std::cout << *it << " " << *minX << " " << *maxX << std::endl;
      if (*minX > *it) minX = it;
      if (*maxX < *it) maxX = it;      
    }
    return std::make_pair(minX, maxX);
  };
  fnctr1<float> a_Max;
  fnctr2<float> a_Greater;
  fnctr3<float> a_Less;
  std::vector<int> myV = {1, 5, 2, 11, -3, 7};
  
  std::cout << a_Max(1.1f, 2.5f) << std::endl;;
  std::cout << l_Max(1.1f, 2.5f) << std::endl;
  std::cout << a_Greater(1.1f, 2.5f) << std::endl;;
  std::cout << l_Greater(1.1f, 2.5f) << std::endl;
  std::cout << a_Less(1.1f, 2.5f) << std::endl;;
  std::cout << l_Less(1.1f, 2.5f) << std::endl;
  fnctr4<int> a_MinMax;
  auto rv =   a_MinMax(begin(myV), end(myV));
  // std::cout << rv.first << " " << rv.second << std::endl;
  // std::cout << findX(begin(myV), end(myV))[3] << std::endl;
  // //std::cout << begin(myV)[0] << std::endl;
  //auto rv = l_MinMax(myV.begin(), myV.end());
  std::cout << rv.first << " " << rv.second << std::endl;
  return 0;
}
