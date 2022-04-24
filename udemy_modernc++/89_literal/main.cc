#include<iostream>
class Distance {
  long double m_km;
public:
  //Distance() = default;
  Distance(long double km) : m_km{km} {}
  //~Distance() = default;
  void print() {std::cout << m_km << "km" << std::endl;}
};
Distance operator"" _mi(long double v) {
  return Distance{v*1.6};
}
Distance operator"" _meter(long double v) {
  return Distance{v/1000.};
}
int main() {
  Distance d0{32.0_mi}; d0.print();
  Distance d1{1600._meter}; d1.print();
  return 0;
}