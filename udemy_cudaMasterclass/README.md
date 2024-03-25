## CUDA Programming Masterclass with C++
- Instructor: Kasun Liyanage
- Ref: https://www.olcf.ornl.gov/cuda-training-series/

## Section 1: Introduction to CUDA programming and CUDA programming model

### 1. Very very important

### 2. Introduction to Parallel Programming
- Thread
  - Thread of execution is the smallest sequence of programmed instructions that can be managed independent by a scheduler
  - Thread is component of a process. Every process has at least on thread called main thread which is th entry point for the program

### 3. Parallel computing and super computing

### 4. How to install CUDA toolkit and first look at CUDA program
- Installing Nvidia driver at ubuntu20
- https://forums.developer.nvidia.com/t/nvidia-smi-has-failed-because-it-couldnt-communicate-with-the-nvidia-driver-make-sure-that-the-latest-nvidia-driver-is-installed-and-running/197141/2

### 5. Basic elements of CUDA program
- Steps of CUDA apps
  - Initialization of data from CPU
  - Transfer data from CPU to GPU
  - Kernel launch with needed grid/block size
  - Transfer results back to CPU
  - Reclaim the memory from both of CPU/GPU
- Elements of a CUDA program
  - Host code (main function): runs in CPU
  - Device code: runs in GPU
- ch5.cu: 
```c
#include <cuda_runtime.h>  
#include "device_launch_parameters.h"
#include <stdio.h>
__global__ void hello_cuda()
{
  printf("Hello CUDA world\n");
}
int main () 
{
  hello_cuda << <1,1 >> > ();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```
- Demo:
```bash
$ nvcc ch5.cu # the file extension must be *.cu, not *.c
$ ./a.out 
Hello CUDA world
```
- Grid: a collection of all threads launch for a kernel
- Block: Threads in a grid is organized in to groups called thread blocks
- `kernel_name <<< n_of_blocks, threads_per_block >>> (arguments)`
- Ex:
  - When there are 32 threads while they are grouped every four, making each block
  - dim3 block(4,1,1)
  - dim3 grid(8,1,1)
  - kernel_name <<< grid, block >>> ()
  - ch5_1.cu
```c
#include <cuda_runtime.h>  
#include "device_launch_parameters.h"
#include <stdio.h>
__global__ void hello_cuda()
{
  printf("Hello CUDA world\n");
}
int main () 
{
  dim3 block(4);
  dim3 grid(8);
  hello_cuda <<<grid,block >>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```
  - This will print Hello ... 32 times
```c
  #include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
__global__ void hello_cuda()
{
  printf("Hello CUDA world\n");
}
int main ()
{
  int nx, ny;
  nx = 16; // total number of threads = 16*4 = 64
  ny = 4;
  dim3 block(8, 2);
  dim3 grid(nx/block.x, ny/block.y); // may couple with block data
  hello_cuda <<<grid,block >>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```
- Q: How to optimize 1) N. of blocks (grids) and 2) threads per block (blocks) ?
  - Max threads per block size: x&y<=1024, z <=64, and x\*y\*z <=1024
  - Max N. blocks : x <=2**32-1, y&z <=65536
  - In many cases, grids >> threads per block
```bash
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
```

### 6. Organizations of threads in CUDA
- threadIdx
  - Determined by blocks (not grid)
  - Same threadIdx will exist for multiple blocks (or grids)
  - Ex:
    - |ABCDEFGH| threads
      - threadIdx.x = 0 1 2 3 4 5 6 7
      - threadIdx.y = 0 0 0 0 0 0 0 0
      - threadIdx.z = 0 0 0 0 0 0 0 0
    - |ABCD| |EFGH| threads
      - threadIdx.x = 0 1 2 3 0 1 2 3
      - threadIdx.y = 0 0 0 0 0 0 0 0
      - threadIdx.z = 0 0 0 0 0 0 0 0
- Assume 16x16 threads, a grid of 4 blocks  
  - The entire threads are 16x16
  - Each block is of 8x8 threads
  - The grid has 2x2 blocks
```c
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
__global__ void print_threadIds()
{
  printf("threadIdx.x/y/z: {%d,%d,%d}\n", threadIdx.x,threadIdx.y,threadIdx.z);
}
int main ()
{
  int nx, ny;
  nx = 16; // total number of threads = 16*16 = 256
  ny = 16;
  dim3 block(8, 8);
  dim3 grid(nx/block.x, ny/block.y); // 2x2 block data
  print_threadIds <<<grid,block >>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```  
- Demo:
```bash
$ nvcc ch06.cu
$ ./a.out |& grep "{7,6,0}"
threadIdx.x/y/z: {7,6,0}
threadIdx.x/y/z: {7,6,0}
threadIdx.x/y/z: {7,6,0}
threadIdx.x/y/z: {7,6,0}
```
  - Note that same indices are found 4x, as there are 2x2 blocks

## 7. Organization of thread in a CUDA program - blockidx,blockDim,gridDim
- blockIdx: CUDA runtime uniquely initializes blockIdx variable for each thread depending on the coordinate of the belonging thread block in the grid (dim3 type data)
  - Ex:
    - |ABCDEFGH| threads
      - blockIdx.x = 0 0 0 0 0 0 0 0
      - blockIdx.y = 0 0 0 0 0 0 0 0
      - blockIdx.z = 0 0 0 0 0 0 0 0
    - |ABCD| |EFGH| threads
      - blockIdx.x = 0 0 0 0 1 1 1 1
      - blockIdx.y = 0 0 0 0 0 0 0 0
      - blockIdx.z = 0 0 0 0 0 0 0 0
- blockDim: number of threads in each dimension of a thead block. Notice that all the thread block in a grid have the same block size (dim3 type data)
- gridDim: number of blocks in x/y/z (dim3 type data)
```c
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
__global__ void print_details()
{
  printf("threadIdx.x/y/z: {%d,%d,%d} blockDim.x/y/z: {%d,%d,%d} griDim.x/y/z: {%d,%d,%d}\n", threadIdx.x,threadIdx.y,threadIdx.z,blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z);
}
int main ()
{
  int nx, ny;
  nx = 16; // total number of threads = 16*16 = 256
  ny = 16;
  dim3 block(8, 8);
  dim3 grid(nx/block.x, ny/block.y); // 2x2 block data
  print_details <<<grid,block >>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```  
- Demo:
```bash
$ nvcc ch07.cu
$ ./a.out |& grep "{7,6,0}"
threadIdx.x/y/z: {7,6,0} blockDim.x/y/z: {8,8,1} griDim.x/y/z: {2,2,1}
threadIdx.x/y/z: {7,6,0} blockDim.x/y/z: {8,8,1} griDim.x/y/z: {2,2,1}
threadIdx.x/y/z: {7,6,0} blockDim.x/y/z: {8,8,1} griDim.x/y/z: {2,2,1}
threadIdx.x/y/z: {7,6,0} blockDim.x/y/z: {8,8,1} griDim.x/y/z: {2,2,1}
```

### 8. Programming exercise 1
- Print value of threadIdx, blockIdx, gridDim for 3D grids which have 4 threads in all x/y/z and thread block size will be 2 threads in each dimension
  - Each block has 2x2x2
  - Grid has 2x2x2 blocks
```c
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
__global__ void print_details()
{
  printf("threadIdx.x/y/z: {%d,%d,%d} blockDim.x/y/z: {%d,%d,%d} griDim.x/y/z: {%d,%d,%d}\n", threadIdx.x,threadIdx.y,threadIdx.z,blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z);
}
int main ()
{
  int nx, ny, nz;
  nx = 4; // total number of threads = 4x4x4 = 64
  ny = 4;
  nz = 4;
  dim3 block(2,2,2);
  dim3 grid(nx/block.x, ny/block.y,nz/block.z); // 2x2x2 block data
  print_details <<<grid,block >>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```

### 9. Unique index calculation using threadIdx, blockId, and blockDim
- Using threadIdx, blockIdx, blockDim, gridDim, a unique index of each thread is found
  - threadIdx out of blockDim
  - blockIdx out of gridDim
- threadIdx is not unique as the same id exits in other blocks
  - Offset of blockIdx*blockDim
- Sample code with 1 block of 8 threads
```c
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
__global__ void unique_idx_calc_threadIdx(int *input)
{
  int tid = threadIdx.x;
  printf("threadIdx: %d value: %d\n", tid, input[tid]);
}
int main ()
{
  int array_size = 8;
  int array_byte_size = sizeof(int) * array_size;
  int h_data[] = {23,9,4,53,65,12,1,33};
  for (int i=0; i< array_size; i++)
  {
    printf("%d\n",h_data[i]);
  }
  printf("\n\n");
  int * d_data;
  cudaMalloc((void**)&d_data, array_byte_size);
  cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);
  dim3 block(8);
  dim3 grid(1);
  unique_idx_calc_threadIdx <<<grid,block >>>(d_data);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```
- Demo:
```bash
$ nvcc ch09.cu
$ ./a.out
23
9
4
53
65
12
1
33
threadIdx: 0 value: 23
threadIdx: 1 value: 9
threadIdx: 2 value: 4
threadIdx: 3 value: 53
threadIdx: 4 value: 65
threadIdx: 5 value: 12
threadIdx: 6 value: 1
threadIdx: 7 value: 33
```
- Sample code with 2 blocks and 4 threads each
  - Change block and grid as shown below
```c
  dim3 block(4);
  dim3 grid(2);
```
  - This yields:
```
threadIdx: 0 value: 23 
threadIdx: 1 value: 9
threadIdx: 2 value: 4
threadIdx: 3 value: 53
threadIdx: 0 value: 23  # duplicated
threadIdx: 1 value: 9   #
threadIdx: 2 value: 4   #
threadIdx: 3 value: 53  #
```
  - tid is not unique and wrong results are made
- Defining a unique global id (gid) using blockIdx and blockDim
```c
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
__global__ void unique_gid_calc_threadIdx(int *input)
{
  int tid = threadIdx.x;
  int gid = tid + blockIdx.x * blockDim.x;
  printf("gid: %d value: %d\n", gid, input[gid]);
};;
int main ()
{
  int array_size = 8;
  int array_byte_size = sizeof(int) * array_size;
  int h_data[] = {23,9,4,53,65,12,1,33};
  for (int i=0; i< array_size; i++)
  {
    printf("%d\n",h_data[i]);
  }
  printf("\n\n");
  int * d_data;
  cudaMalloc((void**)&d_data, array_byte_size);
  cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);
  dim3 block(4);
  dim3 grid(2);
  unique_gid_calc_threadIdx <<<grid,block >>>(d_data);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```
- Now correct results are found:
```bash
gid: 0 value: 23
gid: 1 value: 9
gid: 2 value: 4
gid: 3 value: 53
gid: 4 value: 65
gid: 5 value: 12
gid: 6 value: 1
gid: 7 value: 33
```

### 10. Unique index calculations for 2D grid 1
- A unique index = row offset + block offset + tid
- gid = gridDim.x\*blockDim.x\*blockIdx.y + blockIdx.x\*blockDim.x + threadIdx.x
```c
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
__global__ void unique_gid_calc_2d(int *input)
{
  int tid = threadIdx.x;
  int block_offset = blockIdx.x * blockDim.x;
  int row_offset = blockDim.x * gridDim.x * blockIdx.y;
  int gid = row_offset + block_offset + tid;
  printf("gid: %d value: %d\n", gid, input[gid]);
};
int main ()
{
  int array_size = 8;
  int array_byte_size = sizeof(int) * array_size;
  int h_data[] = {23,9,4,53,65,12,1,33};
  for (int i=0; i< array_size; i++)
  {
    printf("%d\n",h_data[i]);
  }
  printf("\n\n");
  int * d_data;
  cudaMalloc((void**)&d_data, array_byte_size);
  cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);
  dim3 block(4);
  dim3 grid(2);
  unique_gid_calc_2d <<<grid,block >>>(d_data);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```

### 11. Unique index calculations for 2D grid 2
- Memory access pattern will depend on the way we calculate the global index
- We prefer to calculate global indices in a way that threads with in same thread block access consecutive memory locations or consecutive elements in the array
```c
__global__ void unique_gid_calc_2d_2d(int *input)
{
  int tid = blockDim.x*threadIdx.y + threadIdx.x;
  int num_threads_in_a_block = blockDim.x*blockDim.y;
  int block_offset = blockIdx.x * num_threads_in_a_block;
  int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
  int row_offset = num_threads_in_a_row * blockIdx.y;
  int gid = row_offset + block_offset + tid;
  printf("gid: %d value: %d\n", gid, input[gid]);
};
```

### 12. Memory transfer b/w host and device
- cudaMemCpy(destination ptr, source ptr, size_in_byte, direction)
  - Direction: cudamemcpyhtod, cudamemcpydtoh, cudamemcpydtod
- Let's make thread block size as multiples of 32

|   C    | CUDA      |
|--------|-----------|
| malloc | cudaMalloc|
| memset | cudaMemset|
| free   | cudaFree  |

```c
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
__global__ void mem_trs_test(int * input)
{
  int tid = threadIdx.x;
  int block_offset = blockIdx.x * blockDim.x;
  int row_offset = blockDim.x * gridDim.x * blockIdx.y;
  int gid = row_offset + block_offset + tid;
  printf("gid: %d value: %d\n", gid, input[gid]);
};
int main ()
{
  int size = 128;
  int byte_size = size * sizeof(int);
  int *h_input;
  h_input = (int*)malloc(byte_size);
  time_t t;
  srand((unsigned)time(&t));
  for (int i=0;i<size;i++)
  {
    h_input[i] = (int)(rand() & 0xff);
  }
  int * d_input;
  cudaMalloc((void**)&d_input,byte_size); // same as cudaMalloc(&d_input...) ? what is the benefit of void**?
  cudaMemcpy(d_input,h_input,byte_size,cudaMemcpyHostToDevice);
  dim3 block(64);
  dim3 grid(2);
  mem_trs_test <<< grid,block>>> (d_input);
  cudaDeviceSynchronize();
  cudaFree(d_input);
  free(h_input);
  cudaDeviceReset();
  return 0;
}
```

### 13. Exercise 2
- Produce a random 64 element array and pass to device. 3D grid of 2x2x2 and each block as 2x2x2 threads
- https://stackoverflow.com/questions/11554280/cuda-global-unique-thread-index-in-a-3d-grid
```c
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
__global__ void gid_3d(int * input)
{
  int tid = threadIdx.x;
  int offset = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int gid = tid + offset*blockDim.x;
  printf("gid: %d value: %d\n", gid, input[gid]);
};
int main ()
{
  int size = 64;
  int byte_size = size * sizeof(int);
  int *h_input;
  h_input = (int*)malloc(byte_size);
  time_t t;
  srand((unsigned)time(&t));
  for (int i=0;i<size;i++)
  {
    h_input[i] = (int)(rand() & 0xff);
  }
  int * d_input;
  cudaMalloc(&d_input,byte_size);
  cudaMemcpy(d_input,h_input,byte_size,cudaMemcpyHostToDevice);
  dim3 block(2,2,2);
  dim3 grid(2,2,2);
  gid_3d <<< grid,block>>> (d_input);
  cudaDeviceSynchronize();
  cudaFree(d_input);
  free(h_input);
  cudaDeviceReset();
  return 0;
}
```
- Alternative:
```c
// ref: https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
// Ref not working anymore
__device__ int getGlobalIdx_3D_3D() {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x +
                gridDim.x * gridDim.y * blockIdx.z;
  int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z) +
                (threadIdx.z * (blockDim.x * blockDim.y)) +
                (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
```

### 14. Sum array example with validity check
```c
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
__global__ void sum_array_gpu(int *a, int *b, int *c, int size) 
{
  int tid = threadIdx.x;
  int offset = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int gid = tid + offset*blockDim.x;
  if (gid < size)
  {
    c[gid] = a[gid] + b[gid];
  }
};
void sum_array_cpu(int *a, int *b, int *c, int size)
{
  for (int i=0;i<size;i++)
  { 
    c[i] = a[i] + b[i];
  }
}
int main ()
{
  int size = 10000;
  int block_size = 128;
  int NO_BYTES = size * sizeof(int);
  // host pointers
  int *h_a, *h_b, *gpu_results, *cpu_results;
  // allocate memory for host pointers
  h_a = (int*) malloc(NO_BYTES);
  h_b = (int*) malloc(NO_BYTES);
  cpu_results = (int*) malloc(NO_BYTES);
  gpu_results = (int*) malloc(NO_BYTES);
  // initialize host pointer
  time_t t;
  srand((unsigned)time(&t));
  for(int i =0; i<size; i++) 
  {
    h_a[i] = (int)(rand() & 0xff);
    h_b[i] = (int)(rand() & 0xff);
  }
  memset(gpu_results,0,NO_BYTES); // initializes as zero
  // device pointer
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, NO_BYTES);
  cudaMalloc(&d_b, NO_BYTES);
  cudaMalloc(&d_c, NO_BYTES);
  // copy from host to device
  cudaMemcpy(d_a,h_a, NO_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b, NO_BYTES, cudaMemcpyHostToDevice);
  dim3 block(block_size);
  dim3 grid(size/block.x + 1);
  sum_array_gpu <<< grid,block>>> (d_a,d_b,d_c,size);
  cudaDeviceSynchronize();
  cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);
  sum_array_cpu(h_a,h_b,cpu_results,size); 
  int diff_sum=0;
  for (int i=0; i< size; i++)
  {
    //printf("%d %d\n", gpu_results[i], cpu_results[i]);
    diff_sum += gpu_results[i] - cpu_results[i];
  }
  printf("diff_sum = %d\n", diff_sum);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(gpu_results);
  free(cpu_results);
  cudaDeviceReset();
  return 0;
}
```

### 15. Sum array example with error handling
- Compile time errors: happens when program is built
- Run time errors: happens when program is run
- Error handling in CUDA:
```c
cudaError cuda_function(...)
cudaGetErrorString(error)
```
- Macro for cuda error check
```c
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}
...
gpuErrchk( cudaMalloc(&d_a, size*sizeof(int)) );
```

### 16. Sum array with timing
- CPU
```c
clock_t cpu_start, cpu_end;
cpu_start = clock();
...
cpu_end = clock();
printf("Sum array CPU wall time = %4.6f\n",
      (double)((double)(cpu_end-cpu_start)/CLOCKS_PER_SEC));
```
- Performance of a CUDA application
  - Execution time
  - Power consumption
  - Floor space
  - Cost of HW
- Trial and Error method
  - Running the CUDA program with different grid, block, shared memory, cache, memory access configurations and choose the best configuration based on the execution time

### Assignment 2: Extend sum array implementation to sum up 3 arrays
- In the assignment you have to implement array summation in GPU which can sum 3 arrays. You have to use error handling mechanisms, timing measuring mechanisms as well.Then you have to measure the execution time of you GPU implementations.

| block_size | cpu  | gpu  | host->device | device->host |
|------------|------|------|--------------|--------------|
|64          |0.0168|0.0027|0.031         | 0.010        |
|128         |0.0186|0.0027|0.031         | 0.010        |
|256         |0.0166|0.0028|0.031         | 0.010        |
|512         |0.0167|0.0027|0.031         | 0.010        |

### 17. Device properties
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

### 18. Summary
- Basic steps of a CUDA program
  1. Intialize memory in host
  2. Transfer the memory from host to device
  3. Launch the kernel from host
  4. Wait until kernel execution finish
  5. Transfer the memory from device to host
  6. Reclaim the memory
- Launching a kernel from host code, is asynchronous
  - Use of cudaDeviceSynchronize() is necessary

## Section 2: CUDA Execution model

### 19. Understanding the device better
- In every SM:
  - Warp scheduler
  - CUDA cores
  - Registers
  - Load/Store units
  - Special function units
- In CUDA
  - Thread blocks will execute in single SM. Multiple thread blocks can be executed on same SM depending on resource limitation in SM
  - One thread block cannot run on multiple SMs. If device cannot run single block on one SM, error will return for that kernel launch

### 20. All about warps
- Thread blocks are divided into smaller units called warps, which have 32 consecutive threads
  - Warps can be defined as the basic unit of execution in a SM
  - Once a thread block is scheduled to an SM, threads in the thread block are further partitioned into warps
  - All threads in a warp are executed in SMIT fashion
- CUDA execution model ref: https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores
- GT1030 has 1024 threads per block as max
  - This means 32 warps per block as max
  - 3 SM with 2048 threads each
    - Each SM has 128 CUDA cores
    - There could be 6 blocks with 1024 threads each, implying 192 warps
- Only 4 warps can run simultaneously
- A single thread block -> partitioned over warps -> now those threads run with a single instruction (SIMT)
  - Unit of warp allocation is multiples of 32. Could be waste if less than 32 threads are used
```c
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
__global__ void print_details_of_warps()
{
  int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x*blockDim.x*threadIdx.x;
  int warp_id = threadIdx.x/32;
  int gbid = blockIdx.y * gridDim.x + blockIdx.x;
  printf("tid: %d, bid.x: %d, bid.y: %d, gid: %d, warp_id: %d, gbid: %d\n", threadIdx.x, blockIdx.x, blockIdx.y,gid, warp_id,gbid);
};
int main ()
{
  dim3 block_size(42);
  dim3 grid_size(2,2);
  print_details_of_warps<<<grid_size,block_size>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```

### 21. Warp divergence
- Within a single warp, some threads have different instruction than others, this is a warp divergence
  - Significant penalty
  - Any if statement may yield divergence
  - Make conditional flow in units of warp size (=32)
    - Conditional check doesn't diverge when conditions are made in terms of warp size (=32)
- How to calculate branch efficiency
  - 100*((N. branches - N. divergent branches)/N. branches)
```c
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
```bash
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

### 22. Resources partitioning and latency hiding 1
- Local execution context of a warp mainly consists of the following resources
  - Program counters
  - Registers
  - Shared memory
- Set of 32-bit registers stored in a register file that are partitioned among threads, and a fixed amoutn of shared memory that is partitioned among thread blocks
- Warp categories in SM
  - Active blocks/warps: Resource have been allocated
  - Selected warp: actively executing
  - Stalled warp
  - Eligible warp: ready for execution but not currently executing
- Conditions to be an eligible warps
  - 32 CUDA cores must be available for execution
  - All arguments to the current instruction for that warp must be ready

### 23. Resources partitioning and latency hiding 2
- What is latency?
  - Number of clock cycles b/w instruction being issued and being completed
- Latency hiding
  - Switching from one execution context to another has no cost for SM
- Arithmic latency
  - Assume that we need 20 warps to hide context
  - Assume that 1 sm has 128 cores -> 4 warps in one SM
  - 4x20 = 80 to hide the latency per SM (?)
  - If we have 13 SMs then 13x80=1040 warps to hide the latency per device (?)
- Memory latency
  - Assume DRAM latency of Maxwell architecture as 350 cycles
  - GTX970 has bandwidth of 196GB/s
    - nvidia-smi -a -q -d CLOCk
  - 3.6GHz memory clock
  - 196/3.6 = 54Bytes/ cycle
  - 54*350 = 18900 bytes
  - 18900/4 = 4725 threads
  - 4725/32 = 148 warps
  - 146/13 = 12 warps per SM
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
- Categorizing CUDA applications
  - Bandwidth bound applications
  - Computation bound applications

### 24. Occupancy
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

### 25. Profile driven optimization
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

### 26. Parallel reduction as synchronization example
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
- Neighbored pair approach
```c
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}
//reduction neighbored pairs kernel
__global__ void redunction_neighbored_pairs(int * input,
	int * temp, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid > size)
		return;
	for (int offset = 1; offset <= blockDim.x/2; offset *= 2)
	{
		if (tid % (2 * offset) == 0) // only even index works. odd index idles
		{
			input[gid] += input[gid + offset];
      // offset will be 1, 2, 4, 8, 16, 32
      // for T0,T1,T2, .... T63
      // @offset = 1, T0 += T1, T2 += T3, T4 += T5, ....
      // @offset = 2, T0 += T2 (which added T3 before), ...
      // @offset = 4, T0 += T4 (which added T5 and T6 before) ...
      // @offset = 8, T0 += T8 (which added T9, T6, T5) ...
      // @offset = 16, T0 += T16, ....
      // @offset = 32, T0 += T32, ...
		}
		__syncthreads(); 
	}
	if (tid == 0)
	{
		temp[blockIdx.x] = input[gid];
	}
}
int reduction_cpu(int * input, const int size)
{
	int sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += input[i];
	}
	return sum;
}
int main(int argc, char ** argv)
{
	printf("Running neighbored pairs reduction kernel \n");
	int size = 1 << 27; //128 Mb of data - 134_217_728
	int byte_size = size * sizeof(int);
	int block_size = 128;
	int * h_input, *h_ref;
	h_input = (int*)malloc(byte_size);
  for(int i =0; i<size; i++) 
  {
    h_input[i] = (int)(rand() & 0xff);
  }
  //	//get the reduction result from cpu
	int cpu_result = reduction_cpu(h_input,size);
	dim3 block(block_size);   // 128
	dim3 grid(size/ block.x); // 2_097_152
	printf("Kernel launch parameters | grid.x : %d, block.x : %d \n",
		grid.x, block.x);
	int temp_array_byte_size = sizeof(int)* grid.x;
	h_ref = (int*)malloc(temp_array_byte_size);
	int * d_input, *d_temp;
	gpuErrchk(cudaMalloc((void**)&d_input,byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));
	gpuErrchk(cudaMemset(d_temp, 0 , temp_array_byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
		cudaMemcpyHostToDevice));
	redunction_neighbored_pairs << <grid, block >> > (d_input,d_temp, size);
	gpuErrchk(cudaDeviceSynchronize());
	cudaMemcpy(h_ref,d_temp, temp_array_byte_size,
		cudaMemcpyDeviceToHost);
	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}
	//validity check
	//compare_results(gpu_result, cpu_result);
  printf("diff=  %d\n", gpu_result - cpu_result);
	gpuErrchk(cudaFree(d_temp));
	gpuErrchk(cudaFree(d_input));
	free(h_ref);
	free(h_input);
	gpuErrchk(cudaDeviceReset());
	return 0;
}
```
- **Grid size is not necessarily to be the size of HW**
  - Block size is constrained by HW spec
  - Grid size is the size of model, not HW
- As shown above, warp divergence is found and 50% of threads idle (odd indices)

### 27. Parallel reduction reducing warp divergence example
- The code of section 26 idles a half of threads when the gpu sum starts
- From:
```c
	for (int offset = 1; offset <= blockDim.x/2; offset *= 2)	{
		if (tid % (2 * offset) == 0)		{
			input[gid] += input[gid + offset];		}
		__syncthreads();	}
```
- To:
```c
for (int offset = 1; offset <= blockDim.x /2 ; offset *= 2)	{
		int index = 2 * offset * tid;
		if (index < blockDim.x)		{
			input[index] += input[index + offset];		}
		__syncthreads();	}
```
  - This still wastes threads but index of the sum narrows to a warp
- Using interleaved pairs
```c
	for (int offset = blockDim.x/ 2; offset > 0; offset = offset/2)	{
		if (tid < offset)		{
			input[gid] += input[gid + offset];		}
		__syncthreads();	}
```

### 28. Parallel reduction with loop unrolling
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
### 29. Parallel reduction as warp unrolling
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

### 30. Reduction wtih complete unrolling

### 31. Performance comparison of reduction kernels

### 32. Dynamic parallelism
- New GPU kernels from the existing GPU kernel
- Can be recursive
- nvcc -arch=sm_61 -rdc=true 14_dynamic_parallelism.cu
  - An option -rdc=true is necessary to launch kernel from __device__ or __global__
  - Parent kernel waits until the child kernel completes

### 33. Reduction with dynamic parallelism
- nvcc -rdc=true -link common.o reduc.cu
- GPU version is 60x slower than CPU version. How to accelerate?

### 34. Summary

## Section 3: CUDA memory model

### 35. CUDA memory model
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

### 36. Different memory types in CUDA
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

### 37. Memory management and pinned memory
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

### 38. Zero copy memory
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

### 39. Unified memory
- Let CPU/GPU access the same memory address or pointer
- __device__ __managed__ int y;
- cudaMallocManaged()
- No malloc/free/copy function is required

### 40. Global memory access patterns
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

### 41. Global memory writes
- Offset is applied into the c[] array, as writing index

### 42. AOS vs SOA
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

### 43. Matrix transpose
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

### 45. Matrix transpose with diagonal coordinate
- Partition camping: memory requests are queued at some partitions while other partitions remain unused
- Avoid partition camping using diagonal coordinate system

## Section 4: CUDA shared memory and constant memory

### 47. Shared memory
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

### 53. Synchronization in CUDA
- __threadfence_block()
- __threadfence()


### 55. CUDA constant memory
- Can be adjusted from the host
- Must be initialized from the host
- cudaMemcpyToSymbol()

### 56. Matrix transpose with shared memory padding
### 57. Warp shuffle instructions
- Shuffling threads within the same warp

## Section 5: CUDA Streams

### 60. CUDA streams
- Launch multiple kernels, transferring memory b/w kernels by overlapping execution
- CUDA stream: a sequence of commands that execute in order
- Overlapping is the key to transfer memory within device
- NULL stream: default stream that kernel launches and data transfers use
  - Implicitly declared stream

### 61. Asynchronous functions
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

### 62. How to use CUDA streams
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

### 63. Overlapping memory transfer and kernel execution
- ? Kernels didn't overlap?

### 64
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

### 65. Explicit/implicit synchronization
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

### 66. CUDA events
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

### 67. Inter stream dependency
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

## Section 6: Performance Tuning with CUDA instruction level primitives

### 68. Not all instructions are created equally
- SAXPY
  - MAD() in CUDA has less numerical accuracy

### 69. Single vs Double FPE
- Round_to_zero
- Round_up
- Round_down
- Round_to_nearest
- Coping double precision from device/host may be larger than 2x more than single precision

### 70. Standard vs intrinsic functions
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

## Section 7: Parallel patterns and applications

### 72. Scan algorithm
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

### 73. Simple parallel scan
- CPU scan : N-1
- Naive scan: NlogN - (N-1)

### 74. Balanced tree model
- Reduction + down sweep (exclusive scan) phases
- Workload: 2*(N-1)

### 75. Efficient inclusive scan
- Reduction + down sweep (inclusive scan) phases
- Workload: 2*(N-1)

### 76. Parallel scan for large data


## Section 8: Bonus: Introduction to image processing with CUDA
