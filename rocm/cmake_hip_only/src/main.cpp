#include "hip/hip_runtime.h"
#include <cstdio>
#include <cmath>

#include <hip/hip_runtime.h>

#include <helper_hip.h>
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) ;
void callGPU(int blocksPerGrid, int threadsPerBlock, float *d_A, float *d_B, float *d_C, int numElements );
/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for CUDA calls
  hipError_t err = hipSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector A
  float *d_A = NULL;
  err = hipMalloc((void **)&d_A, size);

  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float *d_B = NULL;
  err = hipMalloc((void **)&d_B, size);

  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float *d_C = NULL;
  err = hipMalloc((void **)&d_C, size);

  if (err != hipSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);

  if (err != hipSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

  if (err != hipSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  callGPU( blocksPerGrid, threadsPerBlock, d_A, d_B, d_C, numElements );
  //vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  err = hipGetLastError();

  if (err != hipSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

  if (err != hipSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = hipFree(d_A);

  if (err != hipSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = hipFree(d_B);

  if (err != hipSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = hipFree(d_C);

  if (err != hipSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;
}
