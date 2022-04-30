#include <iostream>
#include <vector>
template<typename T>
class PrettyPrinter {
	T *m_pData;
public:
	PrettyPrinter(T *data) :m_pData(data) {	}
	void Print() {		std::cout << "{" << *m_pData << "}" << std::endl;	}
	T * GetData() {		return m_pData;	}
};
//Explicit specialization of a member function should appear outside the class
template<>
void PrettyPrinter<std::vector<int>>::Print() {
	std::cout << "{";
	for (const auto &x : *m_pData) {
		std::cout << x;
	}
	std::cout << "}" << std::endl;
}
template<>
void PrettyPrinter<std::vector<std::vector<int>>>::Print() {
	std::cout << "{";
	for (const auto &x : *m_pData) {
    std::cout << "{";
		for (const auto &el : x ) {
      std::cout << el;
    }
    std::cout << "}";
	}
	std::cout << "}" << std::endl;
}
int main() {
	std::vector<int> v{ 1,2,3,4,5 };
	PrettyPrinter<std::vector<int>> pv(&v);
	pv.Print();
  std::vector<std::vector<int>> v2{ {1,2,3},{4,5,6}};
  PrettyPrinter<std::vector<std::vector<int>>> pv2(&v2);
  pv2.Print();	
	return 0;
}