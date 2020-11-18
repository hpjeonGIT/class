#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>

using std::cout;
using std::endl;
using std::vector;
using std::chrono::system_clock;
using std::chrono::duration;
using std::default_random_engine;

int main(int argc, char** argv){
  int N = 1<<25;
  int j, local_sum;
  cout <<"Vector size = " << N << endl;
  vector<int> avect(N), ivect(N), svect(N);
  for(uint i=0; i<N; i++) {
    avect[i] = rand()%N;
    ivect[i] = i;
    svect[i] = i;
  }

  unsigned seed = system_clock::now().time_since_epoch().count();
  shuffle (svect.begin(), svect.end(), default_random_engine(seed));

  // array in the order
  auto start = system_clock::now();
  local_sum = 0;
  for(uint i=0;i<N;i++) {
    j = ivect[i];
    local_sum += avect[j];
  }
  auto end = system_clock::now();
  duration<double> elapsed_seconds = end-start;
  cout << "Ordered array - Elapsed time: " << elapsed_seconds.count() << "sec with sum="
       << local_sum <<endl;

   // shuffled array
   start = system_clock::now();
   local_sum = 0;
   for(uint i=0;i<N;i++) {
     j = svect[i];
     local_sum += avect[j];
   }
   end = system_clock::now();
   elapsed_seconds = end-start;
   cout << "Shuffled array - Elapsed time: " << elapsed_seconds.count() << "sec with sum="
        << local_sum <<endl;


  return 0;
}
