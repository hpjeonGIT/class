## CUDA Programming Masterclass with C++
- By Kasun Liyanage
- PPT slides are enclosed in each class

## 5. Basic of CUDA program
- Steps of CUDA apps
  - Initialization of data from CPU
  - Transfer data from CPU to GPU
  - Kernel launch with needed grid/block size
  - Transfer results back to CPU
  - Reclaim the memory from both of CPU/GPU
- Grid: a collection of all threads launch for a kernel
- Block: Threads in a grid is organized in to groups called thread blocks
- `kernel_name <<< n_of_blocks, threads_per_block >>> (arguments)`
- Q: How to optimize 1) N. of blocks (grids) and 2) threads per block (blocks) ?
  - Max threads per block size: x&y<=1024, z <=64, and x*y*z <=1024
  - Max N. blocks : x <=2**32-1, y&z <=65536
  - In many cases, grids >> threads per block
```
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
```

## 6. Organizations of threads in CUDA
- threadIdx
- Same threadIdx will exist for multiple blocks (or grids)
- blockDim = number of threads in each block along x/y/z
- GridDim = number of blocks in x/y/z

## 9. Unique index calculations
- Using threadIdx, blockIdx, blockDim, gridDim, a unique index of each thread is found
  - threadIdx out of blockDim
  - blockIdx out of gridDim
- threadIdx is not unique as the same id exits in other blocks
  - Offset of blockIdx*blockDim

## 10. Unique index calculations for 2D grid
- A unique index = row offset + block offset + tid
- gid = gridDim.x*blockDim.x*blockIdx.y + blockIdx.x*blockDim.x + threadIdx.x

## 12. Memory transfer b/w host and device
- cudaMemCpy(destination ptr, source ptr, size_in_byte, direction)
  - Direction: cudamemcpyhtod, cudamemcpydtoh, cudamemcpydtod
- Let's make thread block size as multiples of 32

## 13. Exercise 2
- Produce a random 64 element array and pass to device. 3D grid of 2x2x2 and each block as 2x2x2 threads
- Use __device__ int getGlobalIdx_3D_3D() to find gid of each thread
  - Ref: https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf

## 14. Sum array example
- memset() is used but seems not necessary

## 15. Error handling
- cudaError cuda_function()
- cudaGetErrorString(error)
- Macro for cuda error check
```
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}
```

## 16. Timing
- CPU
```
clock_t cpu_start, cpu_end;
cpu_start = clock();
...
cpu_end = clock();
printf("Sum array CPU wall time = %4.6f\n",
      (double)((double)(cpu_end-cpu_start)/CLOCKS_PER_SEC));
```


## Assignment 2
- In the assignment you have to implement array summation in GPU which can sum 3 arrays. You have to use error handling mechanisms, timing measuring mechanisms as well.Then you have to measure the execution time of you GPU implementations.
| block_size | cpu  | gpu  | host->device | device->host |
|------------|------|------|--------------|--------------|
|64          |0.0168|0.0027|0.031         | 0.010        |
|128         |0.0186|0.0027|0.031         | 0.010        |
|256         |0.0166|0.0028|0.031         | 0.010        |
|512         |0.0167|0.0027|0.031         | 0.010        |

## 17. Device properties
- warp size is 32 for all HW
- cudaGetDeviceCount() produces the number of devices
- Run 17_device_query.cu:
```
Device 0: GeForce GT 1030
  Number of multiprocessors:                     3
  clock rate :                     1468000
  Compute capability       :                     6.1
  Total amount of global memory:                 2049344.00 KB
  Total amount of constant memory:               64.00 KB
  Total amount of shared memory per block:       48.00 KB
  Total amount of shared memory per MP:          96.00 KB
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum number of threads per multiprocessor:  2048
  Maximum number of warps per multiprocessor:    64
  Maximum Grid size                         :    (2147483647,65535,65535)
  Maximum block dimension                   :    (1024,1024,64)
```

## 19. Understanding the device better
