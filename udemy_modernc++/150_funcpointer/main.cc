#include <iostream>
using Comparator = bool(*)(int,int);
template<typename T, int size>
void Sort(T(&arr)[size], Comparator comp) {
  for (int i=0;i<size-1;++i) {
    for (int j=0;j<size-1;++j) {
      if (comp(arr[j],arr[j+1])) {
        T temp = std::move(arr[j]);
        arr[j] = std::move(arr[j+1]);
        arr[j+1] = std::move(temp);
      }
    }
  }
}
bool Comp_asc(int x, int y) {
  return x > y;
}
bool Comp_dsc(int x, int y) {
  return x < y;
}
int main() {
  int arr[] {3,7,2,9};
  Sort(arr, Comp_dsc); 
  for (auto &x: arr) std::cout << x << std::endl;
  return 0;
}