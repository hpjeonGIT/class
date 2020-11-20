#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void hello_cuda(){
  printf("Hello CUDA world\n");
}

int main(int argc, char ** argv) {
  dim3 grid(2,2);
  dim3 block(1,3);
  hello_cuda << <grid,block>> > ();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
