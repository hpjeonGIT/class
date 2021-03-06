## CUDA Programming Masterclass with C++
- By Kasun Liyanage
- PPT slides are enclosed in each class
- So far, the best Udemy class

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

## 26. Parallel reduction
- cudaDeviceSynchronize(): Introduces a global sync in host code
- __syncthreads(): sync within a block
- Parallel reduction
  - Let each thread sum its own data set
  - When threads complete the local reduction, sum each thread result over iterations of entire threads
```
for (int offset=1; offset < blockDim.x; offset *=2) {
  if (tid %(2*offset) == 0)
    input[tid] += input[tid + offset];
  __syncthreads();
}
```

## 27. Parallel reduction reducing warp divergence
- The code of section 26 idles a half of threads when the gpu sum starts
- From:
```	for (int offset = 1; offset <= blockDim.x/2; offset *= 2)	{
		if (tid % (2 * offset) == 0)		{
			input[gid] += input[gid + offset];		}
		__syncthreads();	}
```
- To:
```
for (int offset = 1; offset <= blockDim.x /2 ; offset *= 2)	{
		int index = 2 * offset * tid;
		if (index < blockDim.x)		{
			i_data[index] += i_data[index + offset];		}
		__syncthreads();	}
```
  - This still wastes threads but index of the sum narrows to a warp
- Using interleaved pairs
```
	for (int offset = blockDim.x/ 2; offset > 0; offset = offset/2)	{
		if (tid < offset)		{
			int_array[gid] += int_array[gid + offset];		}
		__syncthreads();	}
```

## 28. Loop unrolling
- Thread block unrolling
```
if ((index + 3 * blockDim.x) < size) {
  int a1 = input[index];
  int a2 = input[index + blockDim.x];
  int a3 = input[index+ 2* blockDim.x];
  int a4 = input[index+ 3 *blockDim.x];
  input[index] = a1 + a2 + a3 + a4;
}
__syncthreads();
```
## 29. Warp unrolling
```
	if (tid < 32)
	{
		volatile int * vsmem = i_data;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}
```  
- volatile qualifier will disable optimization, preventing a register optimization
  - Ref: https://stackoverflow.com/questions/49163482/cuda-reduction-warp-unrolling-school

## 32. Dynamic parallelism
- New GPU kernels from the existing GPU kernel
- Can be recursive
- nvcc -arch=sm_61 -rdc=true 14_dynamic_parallelism.cu
  - An option -rdc=true is necessary to launch kernel from __device__ or __global__
  - Parent kernel waits until the child kernel completes

## 33. Reduction with dynamic parallelism
- nvcc -rdc=true -link common.o reduc.cu
- GPU version is 60x slower than CPU version. How to accelerate?

## 35. CUDA memory model
- gld_efficiency : global memory load efficiency
- gld_throughput : global memory load throughput
- gld_transactions : global memory load transactions
- gld_transactions_per_request : how many memory transactions needed for one memory request
```
$ sudo /usr/local/cuda-11.1/bin/nvprof --metrics gld_efficiency,gld_throughput,gld_transactions,gld_transactions_per_request ./a.out
Runing 1D grid
Entered block size : 128
Input size : 4194304
Kernel is lauch with grid(32768,1,1) and block(128,1,1)
==7649== NVPROF is profiling process 7649, command: ./a.out
==7649== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "test_sum_array_for_memory(float*, float*, float*, int)" (done)
==7649== Profiling application: ./a.out
==7649== Profiling result:
==7649== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: test_sum_array_for_memory(float*, float*, float*, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gld_throughput                    Global Load Throughput  26.542GB/s  26.542GB/s  26.542GB/s
          1                          gld_transactions                  Global Load Transactions     4194306     4194306     4194306
          1              gld_transactions_per_request      Global Load Transactions Per Request   16.000008   16.000008   16.000008
```

## 36. Different memory types in CUDA
- Registers: fastest. Thread-private. Max 255 registers per thread
  - In nvcc, using --ptxas-options=-v shows the number of registers
```
  __global__ void  register_usage_test(int * results, int size)  {
  	int gid = blockDim.x * blockIdx.x + threadIdx.x;  
  	int x1 = 3465;
  	int x2 = 1768;
  	int x3 = 453;
  	int x4 = x1 + x2 + x3;
  	if (gid < size)   	{
  		results[gid] =  x4;  	}
  }
```
- nvcc --ptxas-options=-v 2_register_usage.cu
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z19register_usage_testPii' for 'sm_52'
ptxas info    : Function properties for _Z19register_usage_testPii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 4 registers, 332 bytes cmem[0]
```
  - gid, x1,x2,x3,x4 are allocated and 5 registers may be expected but 4 registers are reported due to optimization
  - If more registers are used than HW limit, it will spill over to local memory, resulting in performance penalty
- Local Memory: local arrays with indices. High latency memory access
- Shared memory: __shared__ L1 cache and shared memory uses the same on-chip memory

## 37. Memory management and pinned memory
- Pinned memory
  - Host allocated host memory is pageable and GPU cannot access safely
  - Pageable memory must be pinneed before copy
  - cudaMallocHost() will pinn memory
    - cudaFreeHost() to deallocate memory
  - Instead of `float *h_a = (float *)malloc(nbytes);...;free(h_a);`,
  ```
	float *h_a;
	cudaMallocHost((float **)&h_a, nbytes);
  ...
  cudaFreeHost(h_a);
  ```
- nvprof --print-gpu-trace ./a.out
- Pageable:
```
  Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
438.61ms  85.106ms                    -               -         -         -         -  128.00MB  1.4688GB/s    Pageable      Device  GeForce GT 1030         1         7  [CUDA memcpy HtoD]
523.72ms  78.946ms                    -               -         -         -         -  128.00MB  1.5834GB/s      Device    Pageable  GeForce GT 1030         1         7  [CUDA memcpy DtoH]
```
- Pinned:
```
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
373.13ms  84.588ms                    -               -         -         -         -  128.00MB  1.4777GB/s      Pinned      Device  GeForce GT 1030         1         7  [CUDA memcpy HtoD]
457.76ms  78.664ms                    -               -         -         -         -  128.00MB  1.5890GB/s      Device      Pinned  GeForce GT 1030         1         7  [CUDA memcpy DtoH]
```
  - Pinned memory reduces 438ms to 373ms

## 38. Zero copy memory
- Pinned memory mapped into the device address space. Host and device have direct access
- No explicit data transfer
- cudaHostAlloc()/cudaFreeHost()
  - cudaHostAllocDefault: same as pinned memory
  - cudaHostAllocPortable: pinned memory and can be used by CUDA
  - cudaHostAllocWriteCombined: written by host and read by device
  - cudaHostAllocMapped: host memory mapped into device address space. The most common option
  - cudaHostGetDevicePointer(): get the mapped device pointer
    - Memory is allocated from host and the device pointer is necessary to launch the kernel
- May result in low performance
  - But may be useful for large memory requirement

## 39. Unified memory
- Let CPU/GPU access the same memory address or pointer
- __device__ __managed__ int y;
- cudaMallocManaged()
- No malloc/free/copy function is required

## 40. Global memory access patterns
- Aligned memory access : First address is an even multiple of the cache granularity
- Coalesced memory access : 32 threads in a warp access a continuous chunk of memory
- Uncached memory (skipping L1 cache) may be fine-grained, and may be useful for mis-aligned or non-coalesced memory access
- nvcc -Xptxas -dlcm=ca 6_misaligned_read.cu
  - -dlcm=ca : default, enabling L1 available
  - -dlcm=cg : L2 only
- Mis-aligned test using given offset
```
int gid = blockIdx.x * blockDim.x + threadIdx.x;
int k = gid + offset;
if (k < size)
  c[gid] = a[k]+ b[k];
```
  - sudo /usr/local/cuda-11.1/bin/nvprof --metrics gld_efficiency,gld_transactions ./a.out
  - As offset size increases, gld_efficiency decreases from 100% to 80%
  - Difference of using -dlcm=ca & -dlcm=cg is not clear. Same results.

## 41. Global memory writes
- Offset is applied into the c[] array, as writing index

## 42. AOS vs SOA
- AOS
```
struct abc
{
  float x;
  float y;
}
struct abc myA[N];
```
- SOA
```
struct abc
{
  float x[N];
  float y[N];
}
struct abc myA;
```
- sudo /usr/local/cuda-11.1/bin/nvprof --metrics gld_efficiency,gld_transactions ./a.out  
```
 Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: test_aos(testStruct*, testStruct*, int)
          1                            gld_efficiency             Global Memory Load Efficiency      50.00%      50.00%      50.00%
          1                          gld_transactions                  Global Load Transactions     4194306     4194306     4194306
...
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: test_soa(structArray*, structArray*, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                          gld_transactions                  Global Load Transactions     4194306     4194306     4194306
```
  - SOA shows 100% gld efficiency while AOS shows 50%

## 43. Matrix transpose
- sudo /usr/local/cuda-11.1/bin/nvprof --metrics  gld_efficiency,gst_efficiency  ./a.out
  - copy_row() shows 100% while copy_column() yields 12.5%, which is non-coalesced
- copy_row(): well-coalesced (not matrix transpose. Just copy)
```
$ sudo /usr/local/cuda-11.1/bin/nvprof --metrics  gld_efficiency,gst_efficiency  ./a.out
    Kernel: copy_row(int*, int*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
```
- copy_column(): non-coalesced (not matrix transpose. Just copy). gld/gst_efficiency are 12.5%
```      
$ sudo /usr/local/cuda-11.1/bin/nvprof --metrics  gld_efficiency,gst_efficiency  ./a.out 1
    Kernel: copy_column(int*, int*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      12.50%      12.50%      12.50%
          1                            gst_efficiency            Global Memory Store Efficiency      12.50%      12.50%      12.50%
```
- read_row_write_column: reading is coalesced while write is not. gst_efficency is 12.5%
```
    Kernel: transpose_read_row_write_column(int*, int*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency      12.50%      12.50%      12.50%
```
- read_column_write_row: reading is non-coalesced while write is. gld_efficiency is 12.5%
```
    Kernel: transpose_read_column_write_row(int*, int*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      12.50%      12.50%      12.50%
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
```

## 45. Matrix transpose with diagonal coordinate
- Partition camping: memory requests are queued at some partitions while other partitions remain unused
- Avoid partition camping using diagonal coordinate system

## 47. Shared memory
- Mis-aligned/non-coalesced memory access can have benefit from using shared memory
- nvcc -link common.o --ptxas-options=-v 1_intro_smem.cu
```
nvcc -link common.o -arch=sm_61  --ptxas-options=-v 1_intro_smem.cu
...
ptxas info    : Used 6 registers, 340 bytes cmem[0]
```
- Shared memory bank conflicts : when multiple threads request the same memory address, the access is serialized, yielding bank conflicts
  - May use cudaDeviceSetSharedMemConfig() as eight bytes for double-precision data
  - Data might be distributed along banks to reduce serialization

## 53. Synchronization in CUDA
- __threadfence_block()
- __threadfence()


## 55. CUDA constant memory
- Can be adjusted from the host
- Must be initialized from the host
- cudaMemcpyToSymbol()

## 56. Matrix transpose with shared memory padding
## 57. Warp shuffle instructions
- Shuffling threads within the same warp

## 60. CUDA streams
- Launch multiple kernels, transferring memory b/w kernels by overlapping execution
- CUDA stream: a sequence of commands that execute in order
- Overlapping is the key to transfer memory within device
- NULL stream: default stream that kernel launches and data transfers use
  - Implicitly declared stream

## 61. Asynchronous functions
- cudaMemCpyAsync()
  - Host pointers should be pinned memory
  - Stream is assigned
```
cudaStream_t str;
cudaStreamCreate(&str);
cudaMemcpyAsync(d_in, h_in, byte_size, cudaMemcpyHostToDevice, str);
cuda_func <<< grid, block ,0, str >>>(...);
cudaMemcpyAsync(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost, str);
cudaStreamSynchronize(str);
cudaStreamDestroy(str);
```
- 3_simple_cuda_stream_modified.cu produced different results than the lecture. Two kernel executions actually don't overlap

## 62. How to use CUDA streams
- Objective: overlap kernel executions so we can reduce the overhead of memory transfer
  - Launch multiple kernels
    - Concurrent kernel executions
  - Perform asynchronous memory transfer
- Default stream will execute kernel functions one by one
  - No parallel execution
- concurrent.cu
  - std::cout not working in the device
```
cudaStream_t str1, str2, str3;
cudaStreamCreate(&str1);
cudaStreamCreate(&str2);
cudaStreamCreate(&str3);
simple_kernel <<< 1,1,0,str1 >>>();
simple_kernel <<< 1,1,0,str2 >>>();
simple_kernel <<< 1,1,0,str3 >>>();
cudaStreamDestroy(str1);
cudaStreamDestroy(str2);
cudaStreamDestroy(str3);
cudaDeviceSynchronize();
cudaDeviceReset();
```
  - Kernels are overlapping

## 63. Overlapping memory transfer and kernel execution
- ? Kernels didn't overlap?

## 64
- NULL stream can block non-null stream
- Non-NULL streams
  - blocking : can be blocked by NULL stream
  - non-blocking : cannot be blocked by NULL stream
- Streams created by cudaStreamCreate() are blocking streams
1 For blocking streams only
```
gpuErrchk(cudaStreamCreate(&stm1));
gpuErrchk(cudaStreamCreate(&stm2));
gpuErrchk(cudaStreamCreate(&stm3));
...
blocking_nonblocking_test1 << <grid, block, 0, stm1 >> > ();
blocking_nonblocking_test1 << <grid, block, 0, stm2 >> > ();
blocking_nonblocking_test1 << <grid, block, 0, stm3 >> > ();
```
- Yields:
```
---test1
---test1
---test1
```
2 For blocking streams + default stream
```
gpuErrchk(cudaStreamCreate(&stm1));
gpuErrchk(cudaStreamCreate(&stm2));
gpuErrchk(cudaStreamCreate(&stm3));
...
blocking_nonblocking_test1 << <grid, block, 0, stm1 >> > ();
blocking_nonblocking_test1 << <grid, block >> > ();
blocking_nonblocking_test1 << <grid, block, 0, stm3 >> > ();
```
- Yields:
```
--------test1
---test1
-------------test1
```
  - Without stream option, the launched kernel becomes the default stream (2nd blocking_nonblocking_test1 <<< >>>)

3 For one blocking stream + one nonblocking stream + default stream
```
gpuErrchk(cudaStreamCreate(&stm1));
gpuErrchk(cudaStreamCreate(&stm2));
gpuErrchk(cudaStreamCreateWithFlags(&stm3,cudaStreamNonBlocking));
...
blocking_nonblocking_test1 << <grid, block, 0, stm1 >> > ();
blocking_nonblocking_test1 << <grid, block >> > ();
blocking_nonblocking_test1 << <grid, block, 0, stm3 >> > ();
```
- Yields:
```
--------test1 (This is the default stream)
---test1
---test1
```
4 For nonblocking streams + default stream
```
gpuErrchk(cudaStreamCreateWithFlags(&stm1,cudaStreamNonBlocking));
gpuErrchk(cudaStreamCreate(&stm2));
gpuErrchk(cudaStreamCreateWithFlags(&stm3,cudaStreamNonBlocking));
...
blocking_nonblocking_test1 << <grid, block, 0, stm1 >> > ();
blocking_nonblocking_test1 << <grid, block >> > ();
blocking_nonblocking_test1 << <grid, block, 0, stm3 >> > ();
```
- Yields:
```
---test1 (This is the default stream)
---test1
---test1
```

## 65. Explicit/implicit synchronization
- Explicit sync
  - cudaDeviceSynchronize()
  - cudaStreamSynchronize()
  - cudaEventSynchronize()
  - cudaStreamWaitEvent()
- Implicit sync
  - Page-locked host memory allocation
  - Device memory allocation
  - Device memory set
  - Memory copy b/w two addresses to the same device memory
  - Any CUDA command to the NULL stream
  - Switch b/w L1 and shared memory configurations

## 66. CUDA events
- CUDA event : a marker in CUDA stream
  - Can sync stream execution
  - Can Monitor device progress
  - Can measure the execution time of the kernel
- cudaEventCreate()/cudaEventDestroy()
- cudaEventRecord() will queue an event
- cudaEventElapsedTime() will return the elapsed time in msec
```
cudaEvent_t start, end;
cudaEventCreate(&start);
cudaEventCreate(&end);
cudaEventRecord(start);
event_test << < grid,block >>> ();
cudaEventRecord(end);
cudaEventSynchronize(end);
float time;
cudaEventElapsedTime(&time, start, end);
printf("Kernel execution time using events : %f \n",time);
cudaEventDestroy(start);
cudaEventDestroy(end);
```

## 67. Inter stream dependency
- Events can be used to add inter-stream dependencies
```
cudaEvent_t event1;
cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
k1 << <grid, block, 0, stm1 >> > ();
cudaEventRecord(event1, stm1);
cudaStreamWaitEvent(stm3, event1, 0);
k2 << <grid, block, 0, stm2 >> > ();
k3 << <grid, block, 0, stm3 >> > ();
```
- Yields:
```
---k1()
----k2()
-------k3()
```
  - k3() is executed after k1() is completed

## 68. Not all instructions are created equally
- SAXPY
  - MAD() in CUDA has less numerical accuracy

## 69. Single vs Double FPE
- Round_to_zero
- Round_up
- Round_down
- Round_to_nearest
- Coping double precision from device/host may be larger than 2x more than single precision

## 70. Standard vs intrinsic functions
- Standard functions can be used in host and device
  - sqrt, pow, ... from math.h
- Intrinsic functions can be used in device only
  - Uses fewer instructions than the equivalent in standard functions
  - Faster but less accurate than standard functions
  - __powf() is less accurate than powf()
- nvcc --ptx 3_standard_intrinsic.cu
- Open 3_standard_intrinsic.ptx
- Intrinsic function kernel
```
// .globl	_Z9intrinsicPf
.visible .entry _Z9intrinsicPf(
.param .u64 _Z9intrinsicPf_param_0
)
{
.reg .f32 	%f<5>;
.reg .b64 	%rd<3>;
ld.param.u64 	%rd1, [_Z9intrinsicPf_param_0];
cvta.to.global.u64 	%rd2, %rd1;
ld.global.f32 	%f1, [%rd2];
lg2.approx.f32 	%f2, %f1;
add.f32 	%f3, %f2, %f2;
ex2.approx.f32 	%f4, %f3;
st.global.f32 	[%rd2], %f4;
ret;
}
```
  - Much shorter than standard function kernel and approx. instructions are found
- Standard function kernel
```
// .globl	_Z8standardPf
.visible .entry _Z8standardPf(
.param .u64 _Z8standardPf_param_0
)
{
.reg .pred 	%p<17>;
.reg .f32 	%f<103>;
.reg .b32 	%r<15>;
.reg .b64 	%rd<3>;
ld.param.u64 	%rd2, [_Z8standardPf_param_0];
cvta.to.global.u64 	%rd1, %rd2;
mov.f32 	%f14, 0f3F800000;
cvt.rzi.f32.f32	%f15, %f14;
add.f32 	%f16, %f15, %f15;
...
...
...
```
  - Much longer and standard instructions
- Singe precision of powf() comparison
```
Host calculated	        		66932852.000000  # powf() in the host
Standard Device calculated	66932848.000000  # powf() in the device
Intrinsic Device calculated	66932804.000000  # __powf() in the device
```
- Double precision of pow() in host and device. Intrinsic result used __powf() as there is no double-precision equivalent of __powf()
```
Host calculated		        	66932851.562500  # pow() in the host
Standard Device calculated	66932851.562500  # pow() in the device
Intrinsic Device calculated	66932804.000000  # __powf() in the device
```

## 72. Scan algorithm
- Prefix sum (or called `scan`)
  - Cumulative sum in computer science
```
y0 = x0
y1 = x0 + x1
y2 = x0 + x1 + x2
...
```
- MPI : MPI_Scan(), MPI_Exscan()
- C++ : std::inclusive_scan(), std::exclusive_scan()

## 73. Simple parallel scan
- CPU scan : N-1
- Naive scan: NlogN - (N-1)

## 74. Balanced tree model
- Reduction + down sweep (exclusive scan) phases
- Workload: 2*(N-1)

## 75. Efficient inclusive scan
- Reduction + down sweep (inclusive scan) phases
- Workload: 2*(N-1)

## 76. Parallel scan for large data
