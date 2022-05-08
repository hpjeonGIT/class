#include <iostream>
#include <thread>
#include <future>
#include <chrono>
#include <execution>
#include <memory>
const int nsize=100'000'000;
const int niter=10;
using vi = std::vector<int>;
using vt = std::vector<std::thread>;
class join_threads {
  vt& threads;
public:
  explicit join_threads(vt& _threads): threads(_threads) {}
  ~join_threads() {
    for (auto &x : threads) {
      if (x.joinable()) x.join();
    }
  }
};
//
template<typename Iterator, typename MatchType>
Iterator parallel_find_pt(Iterator first, Iterator last, MatchType match)
{
	struct find_element 
	{
		void operator()(Iterator begin, Iterator end,
			MatchType match,
			std::promise<Iterator>* result,
			std::atomic<bool>* done_flag)
		{
			try
			{
				for( ; (begin != end) && !std::atomic_load(done_flag); ++begin)
				{
					if (*begin == match)
					{
						result->set_value(begin);
						std::atomic_store(done_flag, true);
						return;
					}
				}
			}
			catch (...)
			{
				result->set_exception(std::current_exception());
				done_flag->store(true);
			}
		}
	};
	size_t const length = std::distance(first, last);
  size_t const num_threads = 3;
  size_t const block_size = length/num_threads;
  std::promise<Iterator> result;
	std::atomic<bool> done_flag(false);
	std::vector<std::thread> threads(num_threads - 1);
	{
		join_threads joiner(threads);
		Iterator block_start = first;
		for (unsigned long i = 0; i < (num_threads - 1); i++)
		{
			Iterator block_end = block_start;
			std::advance(block_end, block_size);
			threads[i] = std::thread(find_element(), block_start, block_end, match, &result, &done_flag);
			block_start = block_end;
		}
    find_element()(block_start, last, match, &result, &done_flag);
	}
	if (!done_flag.load()) {
		return last;
	}
	return result.get_future().get();
}
int main(){
  vi x(nsize);
  for (int i = 0; i < nsize; i++) x[i] = i;
  auto target = nsize/2;
  {
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto ans = std::find(std::execution::seq, x.begin(), x.end(), target);
    const auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Serial find " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() 
      << " " << std::endl;
    if (ans != x.end()) std::cout << "index = " << ans - x.begin() << std::endl;
  }
  {
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto ans = std::find(std::execution::par, x.begin(), x.end(), target);
    const auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Par find " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() 
      << " " << std::endl;
  }
  {
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto ans = parallel_find_pt(x.begin(), x.end(), target);
    const auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "find_pt " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() 
      << " " << std::endl;
    if (ans != x.end()) std::cout << "index = " << ans - x.begin() << std::endl;
  }  
  return 0;
}
