#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ref: https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
__device__ int getGlobalIdx_3D_3D() {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x +
                gridDim.x * gridDim.y * blockIdx.z;
  int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z) +
                (threadIdx.z * (blockDim.x * blockDim.y)) +
                (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

__global__ void mem_transfer(int * input, int size){
  int gid = getGlobalIdx_3D_3D();
  if (gid < size) {
    printf("tid.xyz = %d %d %d, gid = %d, value = %d\n",
          threadIdx.x, threadIdx.y, threadIdx.z, gid, input[gid]);
  }
}

int main(int argc, char ** argv) {
  int size = 64;
  int byte_size = size * sizeof(int);
  int *h_input;
  h_input = (int *)malloc(byte_size);
  time_t t;
  srand((unsigned)time(&t));
  for (int i=0; i<size; i++) {
    h_input[i] = (int) (rand() & 0xff);
  }

  int * d_input;
  cudaMalloc((void**)&d_input, byte_size);
  cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
  dim3 block(2,2,2);
  dim3 grid(2,2,2);
  mem_transfer <<<grid, block>>>(d_input, size);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
