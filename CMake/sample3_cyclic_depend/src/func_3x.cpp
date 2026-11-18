#include "common.h"
int ret_3x(int &x) {
  return x*3;
} 
int comb_B(int &x) {
  return x*3 + ret_2x(x) + ret_4x(x);
}
