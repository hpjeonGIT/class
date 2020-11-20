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
- In each block, warp scheduler and shared memory exist
- A single thread block must match one SM (stream multiprocessor)

## 20. Warps
- Thread blocks are divided into smaller units called warps, which have 32 consecutive threads
- CUDA execution model ref: https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores
- GT1030 has 1024 threads per block as max
  - This means 32 warps per block as max
  - 3 SM with 2048 threads each
    - Each SM has 128 CUDA cores
    - There could be 6 blocks with 1024 threads each, implying 192 warps
- Only 4 warps can run simultaneously
- A single thread block -> partitioned over warps -> now those threads run with a single instruction (SIMT)
  - Unit of warp allocation is multiples of 32. Could be waste if less than 32 threads are used

## 21. Warp divergence
- Within a single warp, some threads have different instruction than others, this is a warp divergence
  - Significant penalty
  - Any if statement may yield divergence
  - Make conditional flow in units of warp size (=32)
- How to calculate branch efficiency
  - 100*((N. branches - N. divergent branches)/N. branches)
```
if (tid%2 !=0){
  // do something
} else {
  // do another
}
```
  - 100*(2-1)/2 = 50% branch efficiency
- Comparison of conditional flow by warp id or not
  - Use 3_warp_divergence.cu
  - `sudo /usr/local/cuda-11.1/bin/nvprof --metrics branch_efficiency ./a.out`
  - Default compilation will optimize the code and efficiency will be 100%
  - Use -G as a debug mode, then the difference is found
```
$ nvcc -G 3_warp_divergence.cu
$ sudo /usr/local/cuda-11.1/bin/nvprof --metrics branch_efficiency ./a.out

-----------------------WARP DIVERGENCE EXAMPLE------------------------

==20446== NVPROF is profiling process 20446, command: ./a.out
==20446== Profiling application: ./a.out
==20446== Profiling result:
==20446== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: code_without_divergence(void)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
    Kernel: divergence_code(void)
          1                         branch_efficiency                         Branch Efficiency      83.33%      83.33%      83.33%
```

## 23. Resources partitioning and latency hiding 2
- nvidia-smi -a -q -d CLOCK
```
==============NVSMI LOG==============

Timestamp                                 : Fri Nov 20 10:33:13 2020
Driver Version                            : 455.45.01
CUDA Version                              : 11.1

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Clocks
        Graphics                          : 139 MHz
        SM                                : 139 MHz
        Memory                            : 405 MHz
        Video                             : 544 MHz
    Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Default Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1911 MHz
        SM                                : 1911 MHz
        Memory                            : 3004 MHz
        Video                             : 1708 MHz
    Max Customer Boost Clocks
        Graphics                          : N/A
    SM Clock Samples
        Duration                          : 42158.29 sec
        Number of Samples                 : 100
        Max                               : 1227 MHz
        Min                               : 139 MHz
        Avg                               : 657 MHz
    Memory Clock Samples
        Duration                          : 42158.29 sec
        Number of Samples                 : 100
        Max                               : 3004 MHz
        Min                               : 405 MHz
        Avg                               : 2468 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
```

## 24. Occupancy
- The ratio of active warps to maximum number of warps per SM
- Occupancy calculator - an excel sheet from CUDA development kit
```
$ nvcc -arch=sm_61 --ptxas-options=-v 6_occupancy_test.cu
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z14occupancy_testPi' for 'sm_61'
ptxas info    : Function properties for _Z14occupancy_testPi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 4 registers, 328 bytes cmem[0]
```
  - Use the number of registers into the excel sheet to calculate occupancy
- Best practice
  - Keep the number of threads per block as multiples of 32 (=warp size)
  - Avoid small block sizes. Use 128 or 256 or larger threads per block
  - Keep the number of blocks much larger than the number of SMs

## 25. Profile driven optimization
- nvcc -c common.cpp
- nvcc -link common.o 7_sum_array.cu
- Metrics to check
  - sm_efficiency
  - Achieved_occupancy
  - Branch_efficiency
  - Gld_efficiency
  - Gld_throuput
  - Dram_read_throughput
  - Inst_per_warp
  - Stall_sync
```
sudo  /usr/local/cuda-11.1/bin/nvprof --metrics gld_efficiency,sm_efficiency,achieved_occupancy ./a.out

----------------------- SUM ARRAY EXAMPLE FOR NVPROF ------------------------

Runing 1D grid
Input size : 4194304
Kernel is lauch with grid(32768,1,1) and block(128,1,1)
==23940== NVPROF is profiling process 23940, command: ./a.out
==23940== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==23940== Replaying kernel "sum_arrays_1Dgrid_1Dblock(float*, float*, float*, int)" (1 of 3)..Replaying kernel "sum_arrays_1Dgrid_1Dblock(float*, float*, float*, int)" (done)
Arrays are same al events
==23940== Profiling application: ./a.out
==23940== Profiling result:
==23940== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: sum_arrays_1Dgrid_1Dblock(float*, float*, float*, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                             sm_efficiency                   Multiprocessor Activity      99.69%      99.69%      99.69%
          1                        achieved_occupancy                        Achieved Occupancy    0.910413    0.910413    0.910413
```
