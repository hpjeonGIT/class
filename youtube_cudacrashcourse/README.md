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
