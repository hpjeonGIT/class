#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void print_threadIds(){
  printf("threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d\n",
          threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char ** argv) {
  int nx, ny;
  nx = 16;
  ny = 16;
  dim3 block(8,8);
  dim3 grid(nx/block.x, ny/block.y);
  print_threadIds << <grid,block>> > ();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
