#include "common.h"
int ret_2x(int &x) {
  return x*2;
}
int comb_A(int &x) {
  return x*2 + ret_3x(x) + ret_4x(x);
}
