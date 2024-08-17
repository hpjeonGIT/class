#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

#include <helper_hip.h>
#include <test.hiph>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

void callGPU(int blocksPerGrid, int threadsPerBlock, float *d_A, float *d_B, float *d_C, int numElements )
{
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
}

