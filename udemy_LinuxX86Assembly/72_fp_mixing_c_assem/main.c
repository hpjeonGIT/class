#include <stdio.h>

double sum(double arr[], unsigned len);

int main() {
  double arr[] = {1.6, 1.2, 56.67, 9.0, 87.4};
  printf("%20.7f\n", sum(arr,5));
  return 0;
}
