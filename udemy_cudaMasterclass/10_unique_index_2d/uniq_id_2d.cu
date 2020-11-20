#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void unique_gid_calc_2d(int * input){
  int tid = threadIdx.x;
  int block_offset = blockIdx.x * blockDim.x;
  int row_offset = blockDim.x * gridDim.x * blockIdx.y;
  int gid = tid + row_offset + block_offset;
  printf("blockIdx.x = %d, threadIdx=%d, gid=%d, value=%d\n",
          blockIdx.x, threadIdx.x, gid, input[gid]);

}

int main(int argc, char ** argv) {
  int array_size = 16;
  int array_byte_size = sizeof(int)*array_size;
  int h_data[] = {23,9,4,53,65,12,1,33, 22, 43,56,4,76,81,94,32};
  for (int i = 0 ; i < array_size; i++){
    printf("%d ", h_data[i]);
  }
  printf("\n \n");

  int *d_data;
  cudaMalloc((void**)&d_data, array_byte_size);
  cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);
  dim3 block(4);
  dim3 grid(2,2);
  //unique_idx_calc_threadIdx <<< grid, block >>> (d_data);
  unique_gid_calc_2d <<< grid, block >>> (d_data);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
