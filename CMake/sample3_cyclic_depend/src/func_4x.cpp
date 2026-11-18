#include "common.h"
int ret_4x(int &x) {
  return 4*3;
} 
int comb_C(int &x) {
  return 4*3 + ret_2x(x) + ret_3x(x);
}
