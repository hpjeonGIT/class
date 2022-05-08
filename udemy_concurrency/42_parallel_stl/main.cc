#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <execution>
double drand() {
  double x = (double) rand()/ (double) RAND_MAX;
  return x;
}
const int nsize=1'000'000;
const int niter=10;
using vd = std::vector<double>;
int main(){
  vd x(nsize);
  for (auto &i : x) i = drand();
  for (int i=0;i<niter;i++) {
    vd sorted(x);
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::sort(sorted.begin(),sorted.end());
    const auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Serial vector " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;
  }
  for (int i=0;i<niter;i++) {
    vd sorted(x);
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::par, sorted.begin(),sorted.end());
    const auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel vector " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;
  }
  return 0;
}