// This program calculates matrix multiplication (SGEMM) using cuBLAS
// By: Nick from CoffeeBeforeArch

#include <cublas_v2.h>
#include <curand.h>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#define BILLION 1000000000L ;

using std::cout;
using std::endl;

const int M = 1 << 9;
const int N = 1 << 9;
const int K = 1 << 9;
const int SHMEM_SIZE = N;


__global__ void naive_matmul(const double *a, const double *b, double *c, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}

__global__ void cacheTile_matmul(const double *a, const double *b, double *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double s_a[SHMEM_SIZE];
  __shared__ double s_b[SHMEM_SIZE];
  int tmp = 0;
  for (int i = 0; i < N; i += blockDim.x) {
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];
    __syncthreads();
    for (int j = 0; j < blockDim.x; j++) {
      tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
   }
    __syncthreads();
  }
  c[row * N + col] = tmp;
}

void verify_solution(double *a, double *b, double *c, int M, int N, int K) {
  double epsilon = 0.001;
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      double temp = 0;
      for (int i = 0; i < K; i++) {
        temp += a[row + M * i] * b[col * K + i];
      }
      assert(fabs(c[col * M + row] - temp) <= epsilon);
    }
  }
}

int main() {
  struct timespec start, stop;
  double accum;
  // Dimensions for our matrices
  // MxK * KxN = MxN

  // Pre-calculate the size (in bytes) of our matrices
  const size_t bytes_a = M * K * sizeof(double);
  const size_t bytes_b = K * N * sizeof(double);
  const size_t bytes_c = M * N * sizeof(double);
  const size_t bytes   = N * N * sizeof(double);

  // Vectors for the host data
  std::vector<double> h_a(M * K);
  std::vector<double> h_b(K * N);
  std::vector<double> h_c(M * N);

  // Allocate device memory
  double *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  // Pseudo random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  // Set the seed
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  // Fill the matrix with random numbers on the device
  //curandGenerateUniform(prng, d_a, M * K);
  //curandGenerateUniform(prng, d_b, K * M);
  curandGenerateUniformDouble(prng, d_a, M * K);
  curandGenerateUniformDouble(prng, d_b, K * M);

  // cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Scalaing factors
  double alpha = 1.0;
  double beta = 0.0;
  // 1. Naive multplication
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
  int THREADS = 32; //32
  int BLOCKS = N / THREADS;
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);
  clock_gettime(CLOCK_REALTIME, &start);
  naive_matmul<<<blocks, threads>>>(d_a, d_b, d_c, N);
  //cudaMemcpy(h_a.data(), d_a, bytes_a, cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_b.data(), d_b, bytes_b, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);
  clock_gettime(CLOCK_REALTIME, &stop);
  accum = ( stop.tv_sec - start.tv_sec )+
          ( stop.tv_nsec - start.tv_nsec )/( double ) BILLION ;
  cout << "Using naive method, CUDA wall time=" << accum << endl;
  // 2. Cache Tile multiplication
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
  clock_gettime(CLOCK_REALTIME, &start);
  cacheTile_matmul<<<blocks, threads>>>(d_a, d_b, d_c);
  //cudaMemcpy(h_a.data(), d_a, bytes_a, cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_b.data(), d_b, bytes_b, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);
  clock_gettime(CLOCK_REALTIME, &stop);
  accum = ( stop.tv_sec - start.tv_sec )+
          ( stop.tv_nsec - start.tv_nsec )/( double ) BILLION ;
  cout << "Using Cache Tiling, CUDA wall time=" << accum << endl;
  // 3. CuBlas
  clock_gettime(CLOCK_REALTIME, &start);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, M, d_b, K,
              &beta, d_c, M);
  //cudaMemcpy(h_a.data(), d_a, bytes_a, cudaMemcpyDeviceToHost);
  //cudaMemcpy(h_b.data(), d_b, bytes_b, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);
  clock_gettime(CLOCK_REALTIME, &stop);
  accum = ( stop.tv_sec - start.tv_sec )+
          ( stop.tv_nsec - start.tv_nsec )/( double ) BILLION ;
  cout << "Using cuBlas, CUDA wall time=" << accum << endl;
  // Verify solution
  verify_solution(h_a.data(), h_b.data(), h_c.data(), M, N, K);
  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free our memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
