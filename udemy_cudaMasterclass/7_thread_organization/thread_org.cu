#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void print_details(){
  printf("blockIdx.xyz: %d %d %d, blockDim.xyz: %d %d %d "
         "gridDim.xyz: %d %d %d\n",
         blockIdx.x, blockIdx.y, blockIdx.z,
         blockDim.x, blockDim.y, blockDim.z,
         gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char ** argv) {
  int nx, ny;
  nx = 16;
  ny = 16;
  dim3 block(8,8);
  dim3 grid(nx/block.x, ny/block.y);
  print_details << <grid,block>> > ();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
