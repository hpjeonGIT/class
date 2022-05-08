#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>
__global__ void hello_cuda(){  printf("Hello CUDA world\n");}
int main() {
  dim3 block(4);
  dim3 grid(8);
  hello_cuda<<<grid,block>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}