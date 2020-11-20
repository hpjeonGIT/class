# CUDA Crash Course
- By CoffeeBeforeArch
- For code samples: http://github.com/coffeebeforearch

## 1. Vector Addition
- `using std::cout;` instead of namespace std;
- In cpp, 1<<1=2, 1<<2=4, 1<<3=8

## 2. Unified memory for vector addition
- Using unified memory
  - cudaMalloc() => cudaMallocManaged()
  - Now the pointer can be reached from CPU
  - Now sync is necessary
```
vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);
cudaDeviceSynchronize();
```
- Prefetching
```
int id = cudaGetDevice(&id);
...
cudaMemPrefetchAsync(a, bytes, id);
cudaMemPrefetchAsync(b, bytes, id);
cudaMemPrefetchAsync(c, bytes, id);
...
vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);
cudaDeviceSynchronize();
...
cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);
```

## 3. Matrix Multiplication
- Plain method
```
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
c[row * N + col] = 0;
for (int k = 0; k < N; k++)
  c[row * N + col] += a[row * N + k] * b[k * N + col];
```
- Coalescing writes
  - How to align memory - reducing offset or strides
- CUDA wall time measurement
```
#define BILLION 1000000000L ;
...
int main()
  struct timespec start, stop;
  double accum;
  clock_gettime(CLOCK_REALTIME, &start);
...
  clock_gettime(CLOCK_REALTIME, &stop);
  accum = ( stop.tv_sec - start.tv_sec )+
	        ( stop.tv_nsec - start.tv_nsec )/( double ) BILLION ;
  cout << "CUDA wall time=" << accum << endl;  
```

## 4. Cache Tiled Matrix Multiplication
- A[y][k] * B[k][x]
  - Row is loop invariant for A
  - Column is loop invariant for B
- Copying tiled matrix section into shared memory
  - sync is necessary for threads to complete copying
- 0.07 sec => 0.03 sec

## 5. Coalescing
-

## 6. cuBlas
- nvcc simple_cublas.cu -L/usr/local/cuda/lib64 -lcudart -lcublas
- Vector operation sample

## 7. cuBlas for matrix multiplication
- cublas is column major (like Fortran)
- nvcc cublas_matrix_mul.cu -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand
- Cuda random number generator for host function
```
curandGenerator_t prng;
curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
curandGenerateUniform(prng, d_a, M * K);
```
  - Needs -lcurand
- comprehensive_comparison.cu
  - Naive      : 0.07  sec
  - Cache Tile : 0.029 sec
  - cuBlas     : 0.0055sec
  - When double precision is used, 0.11, 0.24, 0.07
    - Reduce matrix size by a half then 0.0156, 0.0022, 3.45e-5
    - Shared memory size might not fit with double precision + 1<<10

## 8. Sum reduction part 1
- An example of naive sum reduction (sumReduction.cu). As more and more threads become not used through loops, it wouldn't be an efficient method
  - Diverged sum reduction

## 9. Sum reduction part 2
- Module operation (%) is quite expensive. Avoid when possible
- Bank conflicts sum reduction
- From:
```
for (int s = 1; s < blockDim.x; s *= 2) {
  if (threadIdx.x % (2 * s) == 0) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
  __syncthreads();
}
```
- To:
```
for (int s = 1; s < blockDim.x; s *= 2) {
  int index = 2 * s * threadIdx.x;
  if (index < blockDim.x) partial_sum[index] += partial_sum[index + s];
  __syncthreads();
}
```
  - Avoid modulo operation
  - Still keeps the data within tile

## 10. Sum reduction part 3
- How to avoid shared memory bank conflict
- From:
```
for (int s = 1; s < blockDim.x; s *= 2) {
  int index = 2 * s * threadIdx.x;
  if (index < blockDim.x) partial_sum[index] += partial_sum[index + s];
  __syncthreads();
}
```
- To:
```
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		__syncthreads();
	}
```

## 11. Sum reduction part 4
- How to reduce the idling
- From:
```
int tid = blockIdx.x * blockDim.x + threadIdx.x;
partial_sum[threadIdx.x] = v[tid];
__syncthreads();
```
- To:
```
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
__syncthreads();
```

## 12. Sum reduction part 5
- __global__ function() is called from host, running on GPU
- __device__ function is called from GPU, running on GPU


## 14. Programming in Linux
- nvprof ./a.out
```
==15105== NVPROF is profiling process 15105, command: ./a.out
COMPLETED SUCCESSFULLY
==15105== Profiling application: ./a.out
==15105== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.13%  169.63us         1  169.63us  169.63us  169.63us  [CUDA memcpy HtoD]
                   45.01%  155.39us         1  155.39us  155.39us  155.39us  [CUDA memcpy DtoH]
```
- nvprof --print-gpu-trace ./a.out
```
==15561== NVPROF is profiling process 15561, command: ./a.out
COMPLETED SUCCESSFULLY
==15561== Profiling application: ./a.out
==15561== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
347.74ms  169.24us                    -               -         -         -         -  256.00KB  1.4426GB/s    Pageable      Device  GeForce GT 1030         1         7  [CUDA memcpy HtoD]
347.92ms  17.506us            (128 1 1)       (256 1 1)        10  4.0000KB        0B         -           -           -           -  GeForce GT 1030         1         7  sum_reduction(int*, int*) [114]
347.94ms  2.7840us              (1 1 1)       (256 1 1)        10  4.0000KB        0B         -           -           -           -  GeForce GT 1030         1         7  sum_reduction(int*, int*) [115]
```
- nvprof --print-gpu-trace --log-file prof.csv ./a.out
  - Dumps the results to a csv file
- cuobjdump ./a.out
  - cuobjdump -ptx ./a.out  # Dump PTX for all listed device functions
  - cuobjdump -sass ./a.out # Dump CUDA assembly for a single cubin file or all cubin files embedded in the binary
  - cuobjdump -sass -ptx ./a.out
  - Ref: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
