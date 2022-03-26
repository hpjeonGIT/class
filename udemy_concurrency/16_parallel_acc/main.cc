#include <iostream>
#include <thread>
#include <vector>
#include <numeric>
using vint = std::vector<int>;
void serial_accum(const vint& v) {
    std::cout << std::accumulate(v.begin(),v.end(),0) << std::endl;
    std::cout << std::accumulate(v.begin(),v.end(),1,std::multiplies<int>()) << std::endl;
}
void parallel_accum() {
    int num_threads=3; // would be decided by num cores
    int block_size = (v.size() + 1)/num_threads;
    std::vector<int> results(num_threads);
    std::vector<std::thread> threads(num_threads-1);
    iterator last;
    for (int i=0; i< num_threads-1; i++) {
        last = v.begin();
        std
    }
}
int main() {
    vint v = {1,2,3,4};
    serial_accum(v);
    return 0;
}