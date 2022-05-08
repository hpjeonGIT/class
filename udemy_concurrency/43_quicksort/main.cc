template<typename T> <T>
std::list<T> seq_qsort(std::list<T> input) {
  if (input.size() <2) {
    return input;
  }
  std::list<T> result;
  result.splice(result.begin(), input, input.begin());
  T pivot = *result.begin();
  auto div_point = std::partition(input.begin(), input.end(),
  [&] (T const& t)) {
    return t < pivot;
  }
  std::list<T> lower_list;
  lower_list.splice(lower_list.end(), input, input.begin(), div_point);
  auto new_lower(seq_qsort(std::move(lower_list)))
}