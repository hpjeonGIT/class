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
  int nx, ny, nz;
  nx = 4;
  ny = 4;
  nz = 4;
  dim3 block(2,2,2);
  dim3 grid(nx/block.x, ny/block.y, nz/block.z);
  print_details << <grid,block>> > ();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
