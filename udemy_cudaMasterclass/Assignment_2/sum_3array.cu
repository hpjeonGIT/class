#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

__global__ void sum_array_gpu(int *a, int *b, int *c, int *s, int size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    s[gid] = a[gid] + b[gid] + c[gid];
  }
}

void sum_array_cpu(int *a, int *b, int *c, int *s, int size) {
  for(int i=0;i<size;i++) {
    s[i] = a[i] + b[i] + c[i];
  }
}

void compare_arrays(int *a, int *b, int size) {
  for(int i=0;i<size;i++) {
    assert(a[i] == b[i]);
  }
}

int main(int argc, char ** argv) {
  int size = 1<<22; // 4194304
  int block_size = 64;
  int NO_BYTES = size * sizeof(int);
  int *h_a, *h_b, *gpu_results, *h_c, *h_sum;
  h_a = (int*)malloc(NO_BYTES);
  h_b = (int*)malloc(NO_BYTES);
  h_c = (int*)malloc(NO_BYTES);
  h_sum = (int*)malloc(NO_BYTES);
  gpu_results = (int*)malloc(NO_BYTES);
  time_t t;
  srand((unsigned)time(&t));
  for (int i=0; i<size; i++) {
    h_a[i] = (int) (rand() & 0xff);
    h_b[i] = (int) (rand() & 0xff);
    h_c[i] = (int) (rand() & 0xff);
  }

  //memset(gpu_results,0,NO_BYTES);
  clock_t cpu_start, cpu_end;
  cpu_start = clock();
  sum_array_cpu(h_a, h_b, h_c, h_sum, size);
  cpu_end = clock();
  printf("Sum array CPU wall time = %4.6f\n",
        (double)((double)(cpu_end-cpu_start)/CLOCKS_PER_SEC));

  int *d_a, *d_b, *d_c, *d_sum;
  gpuErrchk(cudaMalloc((int**)&d_a, NO_BYTES));
  gpuErrchk(cudaMalloc((int**)&d_b, NO_BYTES));
  gpuErrchk(cudaMalloc((int**)&d_c, NO_BYTES));
  gpuErrchk(cudaMalloc((int**)&d_sum, NO_BYTES));
  clock_t h2d_start, h2d_end;
  h2d_start = clock();
  gpuErrchk(cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_c, h_c, NO_BYTES, cudaMemcpyHostToDevice));
  h2d_end = clock();
  dim3 block(block_size);
  dim3 grid((size/block.x) +1);
  clock_t gpu_start, gpu_end;
  gpu_start = clock();
  sum_array_gpu <<< grid, block >>> (d_a, d_b, d_c, d_sum, size);
  gpuErrchk(cudaDeviceSynchronize());
  gpu_end = clock();
  clock_t d2h_start, d2h_end;
  d2h_start = clock();
  gpuErrchk(cudaMemcpy(gpu_results, d_sum, NO_BYTES, cudaMemcpyDeviceToHost));
  d2h_end = clock();
  printf("Sum array GPU wall time = %4.6f\n",
        (double)((double)(gpu_end-gpu_start)/CLOCKS_PER_SEC));
  printf("Mem copy from host to device wall time = %4.6f\n",
        (double)((double)(h2d_end-h2d_start)/CLOCKS_PER_SEC));
  printf("Mem copy from device to host wall time = %4.6f\n",
        (double)((double)(d2h_end-d2h_start)/CLOCKS_PER_SEC));
  //
  compare_arrays(gpu_results, h_sum, size);
  gpuErrchk(cudaFree(d_a));
  gpuErrchk(cudaFree(d_b));
  gpuErrchk(cudaFree(d_c));
  gpuErrchk(cudaFree(d_sum));
  free(gpu_results);
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_sum);
  return 0;
}
