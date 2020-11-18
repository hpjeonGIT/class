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


## 7. cuBlas for matrix multiplication
- cublas is column major (like Fortran)
- nvcc cublas_matrix_mul.cu -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand
- 0.01 sec for floating number
  - Integer matmul function? May not exist
- Cuda random number generator for host function
```
curandGenerator_t prng;
curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
curandGenerateUniform(prng, d_a, M * K);
```
  - Needs -lcurand

## 8. Sum reduction
