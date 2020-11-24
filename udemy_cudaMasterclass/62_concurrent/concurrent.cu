#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
using std::cout;
using std::endl;
__global__ void simple_kernel() {
  printf("Hello from the kernel\n");
}

int main(int argc, char** argv) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if (deviceProp.concurrentKernels == 0)
    cout <<"> GPU does not support concurrent kernel execution\n";

  cudaStream_t str1, str2, str3;
  cudaStreamCreate(&str1);
  cudaStreamCreate(&str2);
  cudaStreamCreate(&str3);

  simple_kernel <<< 1,1,0,str1 >>>();
  simple_kernel <<< 1,1,0,str2 >>>();
  simple_kernel <<< 1,1,0,str3 >>>();

  cudaStreamDestroy(str1);
  cudaStreamDestroy(str2);
  cudaStreamDestroy(str3);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
