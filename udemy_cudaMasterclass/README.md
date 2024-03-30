## CUDA Programming Masterclass with C++
- Instructor: Kasun Liyanage
  - Github: https://github.com/kasunindikaliyanage/CUDAMasterclass/tree/master/CUDAMasterclass
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
- **Grid size is not necessarily the size of HW**
  - Block size is constrained by HW spec
  - Grid size is the size of model, not HW
- As shown above, warp divergence is found and 50% of threads idle (odd indices)
- In the iteration over offset, every warp in the block is used
  - See next section to minize the use of warp
- Profiling resuls
```bash
$ sudo nvprof --metrics gld_efficiency,sm_efficiency,achieved_occupancy,warp_execution_efficiency ./a.out 
Running neighbored pairs reduction kernel 
Kernel launch parameters | grid.x : 1048576, block.x : 128 
==314317== NVPROF is profiling process 314317, command: ./a.out
==314317== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "redunction_neighbored_pairs(int*, int*, int)" (done)
==314317== Profiling application: ./a.out
diff=  0==314317== Profiling result:
==314317== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GT 1030 (0)"
    Kernel: redunction_neighbored_pairs(int*, int*, int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.10%      25.10%      25.10%
          1                             sm_efficiency                   Multiprocessor Activity     100.00%     100.00%     100.00%
          1                        achieved_occupancy                        Achieved Occupancy    0.979304    0.979304    0.979304
          1                 warp_execution_efficiency                 Warp Execution Efficiency      78.15%      78.15%      78.15%
```

### 27. Parallel reduction reducing warp divergence example
- We are using even index number of threads only. However, we can achieve better performance by reducing warp divergence
  - For 128 thread blocks, there are 4 warps. We may use less number of warps in the iteration of offsets
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
- For interleave,
```bash
$ sudo nvprof --metrics gld_efficiency,sm_efficiency,achieved_occupancy,warp_execution_efficiency ./a.out 
Running neighbored pairs reduction kernel 
Kernel launch parameters | grid.x : 1048576, block.x : 128 
==314213== NVPROF is profiling process 314213, command: ./a.out
==314213== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "redunction_interleave(int*, int*, int)" (done)
==314213== Profiling application: ./a.out
diff=  0==314213== Profiling result:
==314213== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GT 1030 (0)"
    Kernel: redunction_interleave(int*, int*, int)
          1                            gld_efficiency             Global Memory Load Efficiency      86.15%      86.15%      86.15%
          1                             sm_efficiency                   Multiprocessor Activity     100.00%     100.00%     100.00%
          1                        achieved_occupancy                        Achieved Occupancy    0.977700    0.977700    0.977700
          1                 warp_execution_efficiency                 Warp Execution Efficiency      90.99%      90.99%      90.99%
```
- Note that warp execution efficiency increased significantly

### 28. Parallel reduction with loop unrolling
- What is loop unrolling
  - In loop unrolling, rather than writing the body of a loop once and using a loop to execute it repeatedly, the body is written in code multipe times
  - The number of copies made of the loop body is called the loop unrolling factor
- Thread block unrolling
```c
if ((index + 3 * blockDim.x) < size) {
  int a1 = input[index];
  int a2 = input[index +   blockDim.x];
  int a3 = input[index + 2*blockDim.x];
  int a4 = input[index + 3*blockDim.x];
  input[index] = a1 + a2 + a3 + a4;
}
__syncthreads();
```

### 29. Parallel reduction as warp unrolling
- We can avoid warp divergence by using technique called warp unrolling
```c
__global__ void reduction_kernel_warp_unrolling(int * int_array,
	int * temp_array, int size)
{
	int tid = threadIdx.x;
	//element index for this thread
	int index = blockDim.x * blockIdx.x  + threadIdx.x;
	//local data pointer
	int * i_data = int_array + blockDim.x * blockIdx.x ;
	for (int offset = blockDim.x/2; offset >= 64; offset = offset/2)
	{
		if (tid < offset)
		{
			i_data[tid] += i_data[tid + offset];
		}
		__syncthreads();
	}
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
	if (tid == 0)
	{
		temp_array[blockIdx.x] = i_data[0];
	}
}
```  
- volatile qualifier will disable optimization, preventing a register optimization
  - Ref: https://stackoverflow.com/questions/49163482/cuda-reduction-warp-unrolling-school

### 30. Reduction wtih complete unrolling
```c
__global__ void reduction_kernel_complete_unrolling(int * int_array,
	int * temp_array, int size)
{
	int tid = threadIdx.x;
	//element index for this thread
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	//local data pointer
	int * i_data = int_array + blockDim.x * blockIdx.x;
	if (blockDim.x == 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];
	__syncthreads();
	if (blockDim.x == 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];
	__syncthreads();
	if (blockDim.x == 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];
	__syncthreads();
	if (blockDim.x == 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];
	__syncthreads();
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
	if (tid == 0)
	{
		temp_array[blockIdx.x] = i_data[0];
	}
}
```
- Using template would be able to remove unnecessary code when compiled
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "cuda_common.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_SIZE 1024
template<unsigned int iblock_size>
__global__ void reduction_gmem_benchmark(int * input, int * temp, int size)
{
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;
	//manual unrolling depending on block size
	if (iblock_size >= 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];
	__syncthreads();
	if (iblock_size >= 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];
	__syncthreads();
	if (iblock_size >= 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];
	__syncthreads();
	if (iblock_size >= 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];
	__syncthreads();
	//unrolling warp
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
	if (tid == 0)
	{
		temp[blockIdx.x] = i_data[0];
	}
}
template<unsigned int iblock_size>
__global__ void reduction_smem(int * input, int * temp, int size)
{
	__shared__ int smem[BLOCK_SIZE];
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;
	smem[tid] = i_data[tid];
	__syncthreads();
	// in-place reduction in shared memory   
	if (blockDim.x >= 1024 && tid < 512)
		smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256)
		smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128)
		smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)
		smem[tid] += smem[tid + 64];
	__syncthreads();
	//unrolling warp
	if (tid < 32)
	{
		volatile int * vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}
	if (tid == 0)
	{
		temp[blockIdx.x] = smem[0];
	}
}
template<unsigned int iblock_size>
__global__ void reduction_smem_complete_unroll(int * input, int * temp, int size)
{
	__shared__ int smem[BLOCK_SIZE];
	// set thread ID   
	unsigned int tid = threadIdx.x;
	// global index, 4 blocks of input data processed at a time   
	unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	// unrolling 4 blocks   
	int tmpSum = 0;
	// boundary check   
	if (idx + 3 * blockDim.x <= size)
	{
		int a1 = input[idx];
		int a2 = input[idx + blockDim.x];
		int a3 = input[idx + 2 * blockDim.x];
		int a4 = input[idx + 3 * blockDim.x];
		tmpSum = a1 + a2 + a3 + a4;
	}
	smem[tid] = tmpSum;
	__syncthreads();
	// in-place reduction in shared memory   
	if (blockDim.x >= 1024 && tid < 512)
		smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256)
		smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128)
		smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)
		smem[tid] += smem[tid + 64];
	__syncthreads();
	//unrolling warp
	if (tid < 32)
	{
		volatile int * vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}
	if (tid == 0)
	{
		temp[blockIdx.x] = smem[0];
	}
}
int main(int argc, char ** argv)
{
	printf("Running parallel reduction with complete unrolling kernel \n");
	int kernel_index = 0;
	if (argc >1)
	{
		kernel_index = 1;
	}
	int size = 1 << 22;
	int byte_size = size * sizeof(int);
	int block_size = BLOCK_SIZE;
	int * h_input, *h_ref;
	h_input = (int*)malloc(byte_size);
	initialize(h_input, size);
	int cpu_result = reduction_cpu(h_input, size);
	dim3 block(block_size);
	dim3 grid((size / block_size));
	int temp_array_byte_size = sizeof(int)* grid.x;
	h_ref = (int*)malloc(temp_array_byte_size);
	int * d_input, *d_temp;
	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));
	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
		cudaMemcpyHostToDevice));
	if (kernel_index == 0)
	{
		printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);

		switch (block_size)
		{
		case 1024:
			reduction_smem_complete_unroll <1024> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 512:
			reduction_smem_complete_unroll <512> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 256:
			reduction_smem_complete_unroll <256> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 128:
			reduction_smem_complete_unroll <128> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 64:
			reduction_smem_complete_unroll <64> << < grid, block >> > (d_input, d_temp, size);
			break;
		}
	}
	else if (kernel_index == 1)
	{
		printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);
		switch (block_size)
		{
		case 1024:
			reduction_smem <1024> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 512:
			reduction_smem <512> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 256:
			reduction_smem <256> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 128:
			reduction_smem <128> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 64:
			reduction_smem <64> << < grid, block >> > (d_input, d_temp, size);
			break;
		}
	}
	else
	{
		grid.x = grid.x / 4;
		printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);

		switch (block_size)
		{
		case 1024:
			reduction_smem_complete_unroll <1024> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 512:
			reduction_smem_complete_unroll <512> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 256:
			reduction_smem_complete_unroll <256> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 128:
			reduction_smem_complete_unroll <128> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 64:
			reduction_smem_complete_unroll <64> << < grid, block >> > (d_input, d_temp, size);
			break;
		}
	}
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}
	compare_results(gpu_result, cpu_result);
	gpuErrchk(cudaFree(d_input));
	gpuErrchk(cudaFree(d_temp));
	free(h_input);
	free(h_ref);
	gpuErrchk(cudaDeviceReset());
	return 0;
}
```

### 31. Performance comparison of reduction kernels
- Naive neighbored pairs approach
- Interleaved pair approach
- Data block unrolling
- Warp unrolling
- Completely unrolling

### 32. Dynamic parallelism
- New GPU kernels from the existing GPU kernel
  - Size of blocks/grids can change on run-time
  - Can be recursive
- Grid launches in a device thread are visible across a thread block
- Execution of a thread block is not considered complete until all child grids created by all threads in the block have completed
- When a parent launches a child grid, the child is not guaranteed to begin execution until the parent thread block explicitly synchronizes on the child
- Example:
  - Parent kernel will be launched from host with one thread block of 16 threads
  - In each grid, first grid block in the thread block has to launch child grid which has a half of elements in the parent grid
  - Child grid's minimum element count is 1
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
__global__ void dynamic_parallelism_check(int size, int depth)
{
	printf(" Depth : %d - tid : %d \n", depth, threadIdx.x);
	if (size == 1)
		return;
	if (threadIdx.x == 0)
	{
		dynamic_parallelism_check << <1, size / 2 >> > (size / 2, depth + 1);
	}
}
int main(int argc, char** argv)
{
	dynamic_parallelism_check << <1, 16 >> > (16,0);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}
```
- `nvcc -arch=sm_61 -rdc=true 14_dynamic_parallelism.cu`
  - An option `-rdc=true` is necessary to launch kernel from `__device__` or `__global__`
  - Parent kernel waits until the child kernel completes

### 33. Reduction with dynamic parallelism
```c
__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata,
	unsigned int isize)
{
	int tid = threadIdx.x;
	int *idata = g_idata + blockIdx.x*blockDim.x;
	int *odata = &g_odata[blockIdx.x];
	 //stop condition
	if (isize == 2 && tid == 0)
	{
		g_odata[blockIdx.x] = idata[0] + idata[1];
		return;
	}
	 //nested invocation
	int istride = isize >> 1;
	if (istride > 1 && tid < istride)
	{
		 //in place reduction
		idata[tid] += idata[tid + istride];
	}
	 //sync at block level
	__syncthreads();
	 //nested invocation to generate child grids
	if (tid == 0)
	{
		gpuRecursiveReduce << <1, istride >> > (idata, odata, istride);
		cudaDeviceSynchronize();
	}
	 //sync at block level again
	__syncthreads();
}
// ##################################################################
__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata,
	unsigned int isize)
{
	int tid = threadIdx.x;
	int *idata = g_idata + blockIdx.x*blockDim.x;
	int *odata = &g_odata[blockIdx.x];
  if (isize == 2 && tid == 0)
	{
		g_odata[blockIdx.x] = idata[0] + idata[1];
		return;
	}
	int istride = isize >> 4;
	if (istride > 1 && tid < istride)
	{
    idata[tid] += idata[tid + istride*8];
    idata[tid] += idata[tid + istride*4];
    idata[tid] += idata[tid + istride*2];
    idata[tid] += idata[tid + istride]; // This is wrong
	}
	//__syncthreads();
	if (tid == 0)
	{
		gpuRecursiveReduce2 << <1, istride >> > (idata, odata, istride);
		cudaDeviceSynchronize();
	}
	 //sync at block level again
	__syncthreads();
}
```
- nvcc -rdc=true -link common.o reduc.cu
- GPU version is 60x slower than CPU version. How to accelerate?
  - Too many launches of kernel - how to reduce the number of kernel launches?

### 34. Summary
- Warp execution
- Resource partition and latency hiding
- Thread block is scheduled to a single SM
  - Multiple thread blocks can reside in a single SM at a given time
- Optimizing a CUDA program based on CUDA execution model
- **Warp is the basic unit of execution in a CUDA program**
  - If the set of threads execute different instruction than other part of the warp, warp divengence occurs
- Shared memory is a memory shared by all the threads in a thread block
  - Registers will be local to each thread
- Latency of an arithmetic and memory instructions
- Occupancy is a good measurement of how much thread blocks/warp is efficient
- Synchronization b/w threads with in thread block using `__syncthread()` function

## Section 3: CUDA memory model

### 35. CUDA memory model
- gld_efficiency : global memory load efficiency
- gld_throughput : global memory load throughput
- gld_transactions : global memory load transactions
- gld_transactions_per_request : how many memory transactions needed for one memory request
```bash
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
- Locality: applications access a relatively small and localized portion of their address space at any point-in-time
  - Temporal locality
  - Spatial locality

### 36. Different memory types in CUDA
- Registers: fastest. Thread-private. Max 255 registers per thread
  - Share the lifetime with the kernel
  - In nvcc, using `--ptxas-options=-v` shows the number of registers
```c
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
- `nvcc --ptxas-options=-v 2_register_usage.cu`
```bash
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z19register_usage_testPii' for 'sm_52'
ptxas info    : Function properties for _Z19register_usage_testPii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 4 registers, 332 bytes cmem[0]
```
  - gid, x1,x2,x3,x4 are allocated and 5 registers may be expected but 4 registers are reported due to optimization
  - Register spills: If more registers are used than HW limit, it will spill over to local memory, resulting in performance penalty
- Local Memory: local arrays with indices. High latency memory access
- Shared memory: `__shared__` L1 cache and shared memory uses the same on-chip memory

### 37. Memory management and pinned memory
- Host
  - malloc
  - free
- Device
  - cudaMalloc
  - cudaFree
  - cudaMemCpy  
- Pinned memory
  - Host allocated host memory is pageable and GPU cannot access safely
  - Pageable memory must be pinned before copy
  - cudaMallocHost() will pin the memory
  - cudaFreeHost() to deallocate memory
  - Instead of `float *h_a = (float *)malloc(nbytes);...;free(h_a);`,
  ```c
	float *h_a;
	cudaMallocHost((float **)&h_a, nbytes);
  ...
  cudaFreeHost(h_a);
  ```
- `nvprof --print-gpu-trace ./a.out`
- Pageable:
```bash
  Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
438.61ms  85.106ms                    -               -         -         -         -  128.00MB  1.4688GB/s    Pageable      Device  GeForce GT 1030         1         7  [CUDA memcpy HtoD]
523.72ms  78.946ms                    -               -         -         -         -  128.00MB  1.5834GB/s      Device    Pageable  GeForce GT 1030         1         7  [CUDA memcpy DtoH]
```
- Pinned:
```bash
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
373.13ms  84.588ms                    -               -         -         -         -  128.00MB  1.4777GB/s      Pinned      Device  GeForce GT 1030         1         7  [CUDA memcpy HtoD]
457.76ms  78.664ms                    -               -         -         -         -  128.00MB  1.5890GB/s      Device      Pinned  GeForce GT 1030         1         7  [CUDA memcpy DtoH]
```
  - Pinned memory reduces 438ms to 373ms
-  Pinned memory management will be more expensive than regular pageable memory in host

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
```c
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "common.h"
__global__ void sumArrays(int *A, int *B, int *C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) C[i] = A[i] + B[i];
}
__global__ void sumArraysZeroCopy(int *A, int *B, int *C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) C[i] = A[i] + B[i];
}
int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaSetDevice(dev);
	// get device properties
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	// check if support mapped memory
	if (!deviceProp.canMapHostMemory)
	{
		printf("Device %d does not support mapping CPU host memory!\n", 
dev);
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
	// set up data size of vectors
	int power = 22;
	if (argc > 1) power = atoi(argv[1]);
	int nElem = 1 << power;
	size_t nBytes = nElem * sizeof(int);
	// part 1: using device memory
	// malloc host memory
	int *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (int *)malloc(nBytes);
	h_B = (int *)malloc(nBytes);
	hostRef = (int *)malloc(nBytes);
	gpuRef = (int *)malloc(nBytes);
	// initialize data at host side
	initialize(h_A, nElem,INIT_ONE_TO_TEN);
	initialize(h_B, nElem);
	memset(gpuRef, 0, nBytes);
	// malloc device global memory
	int *d_A, *d_B, *d_C;
	cudaMalloc((int**)&d_A, nBytes);
	cudaMalloc((int**)&d_B, nBytes);
	cudaMalloc((int**)&d_C, nBytes);
	// transfer data from host to device
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	// set up execution configuration
	int iLen = 512;
	dim3 block(iLen);
	dim3 grid((nElem + block.x - 1) / block.x);
	sumArrays << <grid, block >> >(d_A, d_B, d_C, nElem);
	cudaDeviceSynchronize();
	// copy kernel result back to host side
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	// free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	// free host memory
	free(h_A);
	free(h_B);
	// part 2: using zerocopy memory for array A and B
	// allocate zerocpy memory
	cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocMapped);
	// initialize data at host side
	initialize(h_A, nElem, INIT_ONE_TO_TEN);
	initialize(h_B, nElem, INIT_ONE_TO_TEN);
	memset(gpuRef, 0, nBytes);
	// get the mapped device pointer
	cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0);
	cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0);
	// execute kernel with zero copy memory
	sumArraysZeroCopy << <grid, block >> >(d_A, d_B, d_C, nElem);
	cudaDeviceSynchronize();
	// copy kernel result back to host side
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	// free  memory
	cudaFree(d_C);
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	free(hostRef);
	free(gpuRef);
	// reset device
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
```
- Build and test:
```bash
$ nvcc -c common.cpp
$ nvcc -link common.o 4_zero_copy_memory.cu
$ nvprof --print-gpu-trace ./a.out
$ sudo nvprof --print-gpu-trace ./a.out 
==612722== NVPROF is profiling process 612722, command: ./a.out
==612722== Profiling application: ./a.out
==612722== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
388.70ms  10.609ms                    -               -         -         -         -  16.000MB  1.4728GB/s    Pageable      Device  NVIDIA GeForce          1         7  [CUDA memcpy HtoD]
399.43ms  10.610ms                    -               -         -         -         -  16.000MB  1.4727GB/s    Pageable      Device  NVIDIA GeForce          1         7  [CUDA memcpy HtoD]
410.04ms  1.1724ms           (8192 1 1)       (512 1 1)         8        0B        0B         -           -           -           -  NVIDIA GeForce          1         7  sumArrays(int*, int*, int*, int) [114]
411.23ms  9.8538ms                    -               -         -         -         -  16.000MB  1.5857GB/s      Device    Pageable  NVIDIA GeForce          1         7  [CUDA memcpy DtoH]
453.56ms  21.159ms           (8192 1 1)       (512 1 1)         8        0B        0B         -           -           -           -  NVIDIA GeForce          1         7  sumArraysZeroCopy(int*, int*, int*, int) [123]
474.74ms  9.8549ms                    -               -         -         -         -  16.000MB  1.5855GB/s      Device    Pageable  NVIDIA GeForce          1         7  [CUDA memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
SrcMemType: The type of source memory accessed by memory operation/copy
DstMemType: The type of destination memory accessed by memory operation/copy
```
- Zero copy memory is way slower than regular cuda memory but could be an alternative when required memory is larger than GPU memory
- Must sync b/w host and device before use

### 39. Unified memory
- Let CPU/GPU access the same memory address or pointer
- `__device__ __managed__ int y;`
- `cudaMallocManaged()`
- No malloc/free/copy function is required
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "cuda_common.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
__global__ void test_unified_memory(float* a, float* b, float *c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
		c[gid] = a[gid] + b[gid];
}
int main(int argc, char** argv)
{
	printf("Runing 1D grid \n");
	int size = 1 << 22;
	int block_size = 128;
	if (argc > 1)
		block_size = 1 << atoi(argv[1]);
	printf("Entered block size : %d \n", block_size);
	unsigned int byte_size = size * sizeof(float);
	printf("Input size : %d \n", size);
	float * A, *B, *ref, *C;
	cudaMallocManaged((void **)&A, byte_size);
	cudaMallocManaged((void **)&B, byte_size);
	cudaMallocManaged((void **)&ref, byte_size);
	C = (float*)malloc(byte_size);
	if (!A)
		printf("host memory allocation error \n");
	for (size_t i = 0; i < size; i++)
	{
		A[i] = i % 10;
		A[i] = i % 7;
	}
	sum_array_cpu(A, B, C, size);
	dim3 block(block_size);
	dim3 grid((size + block.x - 1) / block.x);
	printf("Kernel is lauch with grid(%d,%d,%d) and block(%d,%d,%d) \n",
		grid.x, grid.y, grid.z, block.x, block.y, block.z);
	test_unified_memory << <grid, block >> > (A, B, ref, size);
	gpuErrchk(cudaDeviceSynchronize());
	compare_arrays(ref, C, size);
	free(C);
	return 0;
}
```
- Build and profiling:
```bash
$ nvcc -c common.cpp 
$ nvcc -link common.o 5_sum_array_with_unified_memory.cu 
$ sudo nvprof --unified-memory-profiling off ./a.out
Runing 1D grid 
Entered block size : 128 
Input size : 4194304 
==616673== NVPROF is profiling process 616673, command: ./a.out
Kernel is lauch with grid(32768,1,1) and block(128,1,1) 
Arrays are same 
==616673== Profiling application: ./a.out
==616673== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  26.084ms         1  26.084ms  26.084ms  26.084ms  test_unified_memory(float*, float*, float*, int)
      API calls:   84.29%  141.15ms         3  47.050ms  9.3220us  141.13ms  cudaMallocManaged
                   15.58%  26.092ms         1  26.092ms  26.092ms  26.092ms  cudaDeviceSynchronize
                    0.07%  111.65us        97  1.1510us     109ns  43.152us  cuDeviceGetAttribute
                    0.04%  71.305us         1  71.305us  71.305us  71.305us  cudaLaunchKernel
                    0.01%  18.852us         1  18.852us  18.852us  18.852us  cuDeviceGetName
                    0.00%  6.1030us         1  6.1030us  6.1030us  6.1030us  cuDeviceGetPCIBusId
                    0.00%  1.7640us         2     882ns     167ns  1.5970us  cuDeviceGetCount
                    0.00%     452ns         1     452ns     452ns     452ns  cuDeviceTotalMem
                    0.00%     283ns         2     141ns     105ns     178ns  cuDeviceGet
                    0.00%     196ns         1     196ns     196ns     196ns  cuDeviceGetUuid
```
- `sudo nvprof --unified-memory-profiling per-process-device ./a.out` not working. How to profile unified-memory?

### 40. Global memory access patterns
- Aligned memory access : First address is an even multiple of the cache granularity
- Coalesced memory access : 32 threads in a warp access a continuous chunk of memory
- Uncached memory (skipping L1 cache) may be fine-grained, and may be useful for mis-aligned or non-coalesced memory access
- `nvcc -Xptxas -dlcm=ca 6_misaligned_read.cu`
  - `-dlcm=ca` : default, enabling L1 available
  - `-dlcm=cg` : L2 only
- Mis-aligned test using given offset
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
__global__ void misaligned_read_test(float* a, float* b, float *c, int size, int offset)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int k = gid + offset;
	if (k < size)
		c[gid] = a[k]+ b[k];
	//c[gid] = a[gid];
}
int main(int argc, char** argv)
{
	printf("Runing 1D grid \n");
	int size = 1 << 25;
	int block_size = 128;
	unsigned int byte_size = size * sizeof(float);
	int offset = 0;
	if (argc > 1)
		offset = atoi(argv[1]);
	printf("Input size : %d \n", size);
	float * h_a, *h_b, *h_ref;
	h_a = (float*)malloc(byte_size);
	h_b = (float*)malloc(byte_size);
	h_ref = (float*)malloc(byte_size);
	if (!h_a)
		printf("host memory allocation error \n");
	for (size_t i = 0; i < size; i++)
	{
		h_a[i] = i % 10;
		h_b[i] = i % 7;
	}
	dim3 block(block_size);
	dim3 grid((size + block.x - 1) / block.x);
	printf("Kernel is lauch with grid(%d,%d,%d) and block(%d,%d,%d) \n",
		grid.x, grid.y, grid.z, block.x, block.y, block.z);
	float *d_a, *d_b, *d_c;
	cudaMalloc((void**)&d_a, byte_size);
	cudaMalloc((void**)&d_b, byte_size);
	cudaMalloc((void**)&d_c, byte_size);
	cudaMemset(d_c, 0, byte_size);
	cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice);
	misaligned_read_test << <grid, block >> > (d_a, d_b, d_c, size, offset);
	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_c, byte_size, cudaMemcpyDeviceToHost);
	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);
	free(h_ref);
	free(h_b);
	free(h_a);
}
```
- Build and test:
```bash
$ nvcc -Xptxas -dlcm=ca 6_misaligned_read.cu 
$ sudo nvprof --metrics gld_efficiency,gld_transactions ./a.out 
Runing 1D grid 
Input size : 33554432 
Kernel is lauch with grid(262144,1,1) and block(128,1,1) 
==625116== NVPROF is profiling process 625116, command: ./a.out
==625116== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "misaligned_read_test(float*, float*, float*, int, int)" (done)
==625116== Profiling application: ./a.out
==625116== Profiling result:
==625116== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GT 1030 (0)"
    Kernel: misaligned_read_test(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                          gld_transactions                  Global Load Transactions    33554434    33554434    33554434
```
  - As offset size increases, gld_efficiency decreases from 100% to 80%
  - Difference of using -dlcm=ca & -dlcm=cg is not clear. Same results.

### 41. Global memory writes
- Offset is applied into the c[] array, as writing index
```c
__global__ void misaligned_write_test(float* a, float* b, float *c, int size, int offset)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int k = gid + offset;

	if (k < size)
		c[k] = a[gid] + b[gid];
}
```
- 80% of gld_efficiency

### 42. AOS vs SOA
- AOS
```c
struct abc
{
  float x;
  float y;
}
struct abc myA[N];
```
- SOA
```c
struct abc
{
  float x[N];
  float y[N];
}
struct abc myA;
```
- `sudo /usr/local/cuda-11.1/bin/nvprof --metrics gld_efficiency,gld_transactions ./a.out`
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
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
__global__ void copy_row(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[iy * nx + ix] = mat[iy * nx + ix];
	}
}
__global__ void copy_column(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[ix * ny + iy] = mat[ix * ny + iy];
	}
}
__global__ void transpose_read_row_write_column(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[ix * ny + iy] = mat[iy * nx + ix];
	}
}
__global__ void transpose_read_column_write_row(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < nx && iy < ny)
	{
		transpose[iy * nx + ix] = mat[ix * ny + iy];
	}
}
__global__ void transpose_unroll4_row(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int ti = iy * nx + ix;
	int to = ix * ny + iy;
	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		transpose[to]						= mat[ti];
		transpose[to + ny*blockDim.x]		= mat[ti + blockDim.x];
		transpose[to + ny * 2 * blockDim.x] = mat[ti + 2 * blockDim.x];
		transpose[to + ny * 3 * blockDim.x] = mat[ti + 3 * blockDim.x];
	}
}
__global__ void transpose_unroll4_col(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int ti = iy * nx + ix;
	int to = ix * ny + iy;
	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		transpose[ti] = mat[to];
		transpose[ti + blockDim.x] = mat[to + blockDim.x*ny];
		transpose[ti + 2 * blockDim.x] = mat[to + 2 * blockDim.x*ny];
		transpose[ti + 3 * blockDim.x] = mat[to + 3 * blockDim.x*ny];
	}
}
__global__ void transpose_diagonal_row(int * mat, int * transpose, int nx, int ny)
{
	int blk_x = blockIdx.x;
	int blk_y = (blockIdx.x + blockIdx.y) % gridDim.x;
	int ix = blockIdx.x * blk_x + threadIdx.x;
	int iy = blockIdx.y * blk_y + threadIdx.y;
	if (ix < nx && iy < ny)
	{
		transpose[ix * ny + iy] = mat[iy * nx + ix];
	}
}
int main(int argc, char** argv)
{
	//default values for variabless
	int nx = 1024;
	int ny = 1024;
	int block_x = 128;
	int block_y = 8;
	int kernel_num = 0;
	if (argc > 1)
		kernel_num = atoi(argv[1]);
	int size = nx * ny;
	int byte_size = sizeof(int*) * size;
	printf("Matrix transpose for %d X % d matrix with block size %d X %d \n",nx,ny,block_x,block_y);
	int * h_mat_array = (int*)malloc(byte_size);
	int * h_trans_array = (int*)malloc(byte_size);
	int * h_ref = (int*)malloc(byte_size);
	//initialize matrix with integers between one and ten
	initialize(h_mat_array,size ,INIT_ONE_TO_TEN);
	//matirx transpose in CPU
	mat_transpose_cpu(h_mat_array, h_trans_array, nx, ny);
	int * d_mat_array, *d_trans_array;
	cudaMalloc((void**)&d_mat_array, byte_size);
	cudaMalloc((void**)&d_trans_array, byte_size);
	cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice);
	dim3 blocks(block_x, block_y);
	dim3 grid(nx/block_x, ny/block_y);
	void(*kernel)(int*, int*, int, int);
	char * kernel_name;
	switch (kernel_num)
	{
	case 0:
		kernel = &copy_row;
		kernel_name = "Copy row   ";
		break;
	case 1 :
		kernel = &copy_column;
		kernel_name = "Copy column   ";
		break;
	case 2 :
		kernel = &transpose_read_row_write_column;
		kernel_name = " Read row write column ";
		break;
	case 3:
		kernel = &transpose_read_column_write_row;
		kernel_name = "Read column write row ";
		break;
	case 4:
		kernel = &transpose_unroll4_row;
		kernel_name = "Unroll 4 row ";
		break;
	case 5:
		kernel = &transpose_unroll4_col;
		kernel_name = "Unroll 4 col ";
		break;
	case 6:
		kernel = &transpose_diagonal_row;
		kernel_name = "Diagonal row ";
		break;
	}
	printf(" Launching kernel %s \n",kernel_name);
	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	kernel <<< grid, blocks>> > (d_mat_array, d_trans_array,nx, ny);
	cudaDeviceSynchronize();
	gpu_end = clock();
	print_time_using_host_clock(gpu_start, gpu_end);
	//copy the transpose memroy back to cpu
	cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost);
	//compare the CPU and GPU transpose matrix for validity
	//compare_arrays(h_ref, h_trans_array, size);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
```
- `sudo /usr/local/cuda-11.1/bin/nvprof --metrics  gld_efficiency,gst_efficiency  ./a.out`
  - copy_row() shows 100% while copy_column() yields 12.5%, which is non-coalesced
- copy_row(): well-coalesced (not matrix transpose. Just copy)
```bash
$ sudo /usr/local/cuda-11.1/bin/nvprof --metrics  gld_efficiency,gst_efficiency  ./a.out
    Kernel: copy_row(int*, int*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
```
- copy_column(): non-coalesced (not matrix transpose. Just copy). gld/gst_efficiency are 12.5%
```bash
$ sudo /usr/local/cuda-11.1/bin/nvprof --metrics  gld_efficiency,gst_efficiency  ./a.out 1
    Kernel: copy_column(int*, int*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      12.50%      12.50%      12.50%
          1                            gst_efficiency            Global Memory Store Efficiency      12.50%      12.50%      12.50%
```
- read_row_write_column: reading is coalesced while write is not. gst_efficency is 12.5%
```bash
    Kernel: transpose_read_row_write_column(int*, int*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency      12.50%      12.50%      12.50%
```
- read_column_write_row: reading is non-coalesced while write is. gld_efficiency is 12.5%
```bash
    Kernel: transpose_read_column_write_row(int*, int*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      12.50%      12.50%      12.50%
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
```

### 44. Matrix transpose with unrolling

### 45. Matrix transpose with diagonal coordinate
- Partition camping: memory requests are queued at some partitions while other partitions remain unused
- Avoid partition camping using diagonal coordinate system

### 46. Summary
- Different types of memories in CUDA memory model
  - Pinned
  - Zero copy
  - Unified
- Global memory access pattern
  - Aligned
  - Coalesced
- Global memory writes does not utilize L1 cache
- AOS vs SOA

## Section 4: CUDA shared memory and constant memory

### 47. Intorduction to CUDA shared memory
- Mis-aligned/non-coalesced memory access can have benefit from using shared memory
- Shared memory usage
  - An intra-block thread communication channel
  - A program managed cache for global memory data
  - Scratch pad memory for transforming data to improve global memory access patterns
- A fixed amount of shared memory is allocated to each thread block when it starts executing
- This shared memory address space is shared by **all threads in a thread block**
- Its contents have the same lifetime as the thread block where it was created
- Optimizing memory access
  - Counting on L1 cache to store repeatedly access memory
  - Store repeatedly access memory explicitly in shared memory
    - Buffer for global memory per SM
```c
    #include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#define SHARED_ARRAY_SIZE 128
__global__ void smem_static_test(int * in, int * out, int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int smem[SHARED_ARRAY_SIZE];
	if (gid < size)
	{
		smem[tid] = in[gid];
		out[gid] = smem[tid];
	}
}
__global__ void smem_dynamic_test(int * in, int * out, int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int smem[];
	if (gid < size)
	{
		smem[tid] = in[gid];
		out[gid] = smem[tid];
	}
}
int main(int argc, char ** argv)
{
	int size = 1 << 22;
	int block_size = SHARED_ARRAY_SIZE;
	bool dynamic = false;
	if (argc > 1)
	{
		dynamic = atoi(argv[1]);
	}
	//number of bytes needed to hold element count
	size_t NO_BYTES = size * sizeof(int);
	// host pointers
	int *h_in, *h_ref, *d_in, *d_out;
	// allocate memory for host size pointers
	h_in = (int *)malloc(NO_BYTES);
	h_ref = (int *)malloc(NO_BYTES);
	initialize(h_in, size, INIT_ONE_TO_TEN);
	cudaMalloc((int **)&d_in, NO_BYTES);
	cudaMalloc((int **)&d_out, NO_BYTES);
	// kernel launch parameters
	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);
	cudaMemcpy(d_in, h_in, NO_BYTES, cudaMemcpyHostToDevice);
	if (!dynamic)
	{
		printf("Static smem kernel \n");
		smem_static_test << <grid, block >> > (d_in, d_out, size);
	}
	else
	{
		printf("Dynamic smem kernel \n");
		smem_dynamic_test << <grid, block, sizeof(int)*  SHARED_ARRAY_SIZE >> > (d_in, d_out, size);
    // dynamic shared memory requires one more argument when launching the kernel
	}
	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_out, NO_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);
	free(h_in);
	free(h_ref);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
```
- Build and test
```bash
$ nvcc -link common.o  1_intro_smem.cu 
$ sudo nvprof ./a.out 
==663507== NVPROF is profiling process 663507, command: ./a.out
Static smem kernel 
==663507== Profiling application: ./a.out
==663507== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.91%  10.613ms         1  10.613ms  10.613ms  10.613ms  [CUDA memcpy HtoD]
                   46.35%  9.8565ms         1  9.8565ms  9.8565ms  9.8565ms  [CUDA memcpy DtoH]
                    3.75%  796.62us         1  796.62us  796.62us  796.62us  smem_static_test(int*, int*, int)
      API calls:   76.92%  215.68ms         2  107.84ms  135.07us  215.55ms  cudaMalloc
                   14.86%  41.668ms         1  41.668ms  41.668ms  41.668ms  cudaDeviceReset
                    7.17%  20.101ms         2  10.050ms  9.6484ms  10.452ms  cudaMemcpy
                    0.67%  1.8709ms         1  1.8709ms  1.8709ms  1.8709ms  cudaDeviceSynchronize
                    0.30%  850.10us         2  425.05us  166.88us  683.22us  cudaFree
                    0.05%  128.68us        97  1.3260us     117ns  48.605us  cuDeviceGetAttribute
                    0.01%  41.508us         1  41.508us  41.508us  41.508us  cudaLaunchKernel
                    0.01%  22.443us         1  22.443us  22.443us  22.443us  cuDeviceGetName
                    0.00%  13.120us         1  13.120us  13.120us  13.120us  cuDeviceGetPCIBusId
                    0.00%  2.4290us         3     809ns     126ns  1.4370us  cuDeviceGetCount
                    0.00%  1.3880us         2     694ns     243ns  1.1450us  cuDeviceGet
                    0.00%     470ns         1     470ns     470ns     470ns  cuDeviceTotalMem
                    0.00%     207ns         1     207ns     207ns     207ns  cuDeviceGetUuid
```

### 48. Shared memory access modes and memory banks
- Bank conflict
  - When multiple addresses in a shared memory request fall into the same memory bank, a bank conflict occurs, causing the request to be replayed
  - HW splits a request with a bank conflict into as many separate conflict-free transactions as necessary
- Parallel access of memory banks
  - When multiple addresses accessed by a warp fall into multiple banks, ideally all 32 banks then parallel access of shared memory occurs. This pattern implies that some of the addresses canbe serviced in a single memory transaction
- Sequential access
  - When multiple threads access different memory address in the same bank  
- Broadcast access
  - When multiple threads access the same memory address in the same bank
- Shared memory access modes
  - 32bit
  - 64bit
- bank index = (byte address/ 4bytes/bank)%32banks

### 49. Row major and Column major access to shared memory
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#define BDIMX 32
#define BDIMY 32
__global__ void setRowReadCol(int * out)
{
	__shared__ int tile[BDIMY][BDIMX];
	int idx = threadIdx.y * blockDim.x + threadIdx.x;
	//store to the shared memory
	tile[threadIdx.y][threadIdx.x] = idx;
	//waiting for all the threads in thread block to reach this point
	__syncthreads();
	//load from shared memory
	out[idx] = tile[threadIdx.x][threadIdx.y];
}
__global__ void setColReadRow(int * out)
{
	__shared__ int tile[BDIMY][BDIMX];
	int idx = threadIdx.y * blockDim.x + threadIdx.x;
	//store to the shared memory
	tile[threadIdx.x][threadIdx.y] = idx;
	//waiting for all the threads in thread block to reach this point
	__syncthreads();
	//load from shared memory
	out[idx] = tile[threadIdx.y][threadIdx.x];
}
__global__ void setRowReadRow(int * out)
{
	__shared__ int tile[BDIMY][BDIMX];
	int idx = threadIdx.y * blockDim.x + threadIdx.x;
	//store to the shared memory
	tile[threadIdx.y][threadIdx.x] = idx;
	//waiting for all the threads in thread block to reach this point
	__syncthreads();
	//load from shared memory
	out[idx] = tile[threadIdx.y][threadIdx.x];
}
int main(int argc, char **argv)
{
	int memconfig = 0;
	if (argc > 1)
	{
		memconfig = atoi(argv[1]);
	}
	if (memconfig == 1)
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	}
	else
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	}
	cudaSharedMemConfig pConfig;
	cudaDeviceGetSharedMemConfig(&pConfig);
	printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");
	// set up array size 2048
	int nx = BDIMX;
	int ny = BDIMY;
	bool iprintf = 0;
	if (argc > 2) iprintf = atoi(argv[1]);
	size_t nBytes = nx * ny * sizeof(int);
	// execution configuration
	dim3 block(BDIMX, BDIMY);
	dim3 grid(1, 1);
	printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,block.y);
	// allocate device memory
	int *d_C;
	cudaMalloc((int**)&d_C, nBytes);
	int *gpuRef = (int *)malloc(nBytes);
	cudaMemset(d_C, 0, nBytes);
	setColReadRow << <grid, block >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	if (iprintf)  printData("set col read col   ", gpuRef, nx * ny);
	cudaMemset(d_C, 0, nBytes);
	setRowReadRow << <grid, block >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	if (iprintf)  printData("set row read row   ", gpuRef, nx * ny);
	cudaMemset(d_C, 0, nBytes);
	setRowReadCol << <grid, block >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	if (iprintf)  printData("set row read col   ", gpuRef, nx * ny);
	// free host and device memory
	cudaFree(d_C);
	free(gpuRef);
	// reset device
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
```
- Build and test:
```bash
$ nvcc -c common.cpp
$ nvcc -link common.o sm.cu 
$ sudo nvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request  ./a.out 
==669784== NVPROF is profiling process 669784, command: ./a.out
with Bank Mode:4-Byte <<< grid (1,1) block (32,32)>>>
==669784== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "setColReadRow(int*)" (done)
Replaying kernel "setRowReadRow(int*)" (done)
Replaying kernel "setRowReadCol(int*)" (done)
==669784== Profiling application: ./a.out
==669784== Profiling result:
==669784== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GT 1030 (0)"
    Kernel: setColReadRow(int*)
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   32.000000   32.000000   32.000000
    Kernel: setRowReadCol(int*)
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.000000   32.000000   32.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
    Kernel: setRowReadRow(int*)
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
```

### 50. Static and Dynamic shared memory
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#define BDIMX 32
#define BDIMY 32
__global__ void setRowReadColDyn(int * out)
{
	extern __shared__ int tile[];
	int row_index = threadIdx.y * blockDim.x + threadIdx.x;
	int col_index = threadIdx.x * blockDim.y + threadIdx.y;
	tile[row_index] = row_index;
	__syncthreads();
	out[row_index] = tile[col_index];
}
int main(int argc, char **argv)
{
	cudaSharedMemConfig pConfig;
	cudaDeviceGetSharedMemConfig(&pConfig);
	printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");
	// set up array size 2048
	int nx = BDIMX;
	int ny = BDIMY;
	bool iprintf = 0;
	if (argc > 1) iprintf = atoi(argv[1]);
	size_t nBytes = nx * ny * sizeof(int);
	// execution configuration
	dim3 block(BDIMX, BDIMY);
	dim3 grid(1, 1);
	printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
		block.y);
	// allocate device memory
	int *d_C;
	cudaMalloc((int**)&d_C, nBytes);
	int *gpuRef = (int *)malloc(nBytes);
	cudaMemset(d_C, 0, nBytes);
	setRowReadColDyn << <grid, block, sizeof(int) * (nx*ny) >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	// free host and device memory
	cudaFree(d_C);
	free(gpuRef);
	// reset device
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
```

### 51. Shared memory padding
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#define BDIMX 32
#define BDIMY 32
#define IPAD 1
__global__ void setRowReadColPad(int * out)
{
	__shared__ int tile[BDIMY][BDIMX + IPAD];
	int idx = threadIdx.y * blockDim.x + threadIdx.x;
	//store to the shared memory
	tile[threadIdx.y][threadIdx.x] = idx;
	//waiting for all the threads in thread block to reach this point
	__syncthreads();
	//load from shared memory
	out[idx] = tile[threadIdx.x][threadIdx.y];
}
__global__ void setRowReadColDynPad(int * out)
{
	extern __shared__ int tile[];
	int row_index = threadIdx.y * (blockDim.x+ IPAD) + threadIdx.x;
	int col_index = threadIdx.x * (blockDim.x + IPAD) + threadIdx.y;
	tile[row_index] = row_index;
	__syncthreads();
	out[row_index] = tile[col_index];
}
int main(int argc, char **argv) 
{
	cudaSharedMemConfig pConfig;
	cudaDeviceGetSharedMemConfig(&pConfig);
	printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");
	// set up array size 2048
	int nx = BDIMX;
	int ny = BDIMY;
	bool iprintf = 0;
	if (argc > 1) iprintf = atoi(argv[1]);
	size_t nBytes = nx * ny * sizeof(int);
	// execution configuration
	dim3 block(BDIMX, BDIMY);
	dim3 grid(1, 1);
	printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
		block.y);
	// allocate device memory
	int *d_C;
	cudaMalloc((int**)&d_C, nBytes);
	int *gpuRef = (int *)malloc(nBytes);
	cudaMemset(d_C, 0, nBytes);
	setRowReadColPad << <grid, block >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	cudaMemset(d_C, 0, nBytes);
	setRowReadColDynPad << <grid, block, sizeof(int) * ((nx + IPAD)*ny) >> > (d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	// free host and device memory
	cudaFree(d_C);
	free(gpuRef);
	// reset device
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
```
- Build and test
```bash
sudo nvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request  ./a.out 
==672818== NVPROF is profiling process 672818, command: ./a.out
with Bank Mode:4-Byte <<< grid (1,1) block (32,32)>>>
==672818== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "setRowReadColPad(int*)" (done)
Replaying kernel "setRowReadColDynPad(int*)" (done)
==672818== Profiling application: ./a.out
==672818== Profiling result:
==672818== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GT 1030 (0)"
    Kernel: setRowReadColDynPad(int*)
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
    Kernel: setRowReadColPad(int*)
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
```
- Now all of load/store transaction per request is 1

### 52. Parallel reduction with shared memory
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "cuda_common.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_SIZE 1024
template<unsigned int iblock_size>
__global__ void reduction_gmem_benchmark(int * input,int * temp, int size)
{
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;
	//manual unrolling depending on block size
	if (iblock_size >= 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];
	__syncthreads();
	if (iblock_size >= 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];
	__syncthreads();
	if (iblock_size >= 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];
	__syncthreads();
	if (iblock_size >= 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];
	__syncthreads();
	//unrolling warp
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
	if (tid == 0)
	{
		temp[blockIdx.x] = i_data[0];
	}
}
template<unsigned int iblock_size>
__global__ void reduction_smem(int * input, int * temp, int size)
{
	__shared__ int smem[BLOCK_SIZE];
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;
	smem[tid] = i_data[tid];
	__syncthreads();
	//manual unrolling depending on block size
	if (iblock_size >= 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];
	__syncthreads();
	if (iblock_size >= 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];
	__syncthreads();
	if (iblock_size >= 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];
	__syncthreads();
	if (iblock_size >= 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];
	__syncthreads();
	//unrolling warp
	if (tid < 32)
	{
		volatile int * vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}
	if (tid == 0)
	{
		temp[blockIdx.x] = i_data[0];
	}
}
int main(int argc, char ** argv)
{
   printf("Running parallel reduction with complete unrolling kernel \n");
	int kernel_index = 0;
	if (argc >1)
	{
		kernel_index = 1;
	}
	int size = 1 << 22;
	int byte_size = size * sizeof(int);
	int block_size = BLOCK_SIZE;
	int * h_input, *h_ref;
	h_input = (int*)malloc(byte_size);
	initialize(h_input, size);
	int cpu_result = reduction_cpu(h_input, size);
	dim3 block(block_size);
	dim3 grid((size / block_size));
	int temp_array_byte_size = sizeof(int)* grid.x;
	h_ref = (int*)malloc(temp_array_byte_size);
	int * d_input, *d_temp;
	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));
	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
		cudaMemcpyHostToDevice));
	if (kernel_index == 0)
	{
		printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);
		switch (block_size)
		{
		case 1024:
			reduction_gmem_benchmark <1024> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 512:
			reduction_gmem_benchmark <512> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 256:
			reduction_gmem_benchmark <256> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 128:
			reduction_gmem_benchmark <128> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 64:
			reduction_gmem_benchmark <64> << < grid, block >> > (d_input, d_temp, size);
			break;
		}
	}
	else
	{
		printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);
		switch (block_size)
		{
		case 1024:
			reduction_smem <1024> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 512:
			reduction_smem <512> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 256:
			reduction_smem <256> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 128:
			reduction_smem <128> << < grid, block >> > (d_input, d_temp, size);
			break;
		case 64:
			reduction_smem <64> << < grid, block >> > (d_input, d_temp, size);
			break;
		}
	}
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}
	compare_results(gpu_result, cpu_result);
	gpuErrchk(cudaFree(d_input));
	gpuErrchk(cudaFree(d_temp));
	free(h_input);
	free(h_ref);
	gpuErrchk(cudaDeviceReset());
	return 0;
}
```
- Build and test
```bash
$ sudo nvprof --metrics gld_transactions,gst_transactions ./a.out 
Running parallel reduction with complete unrolling kernel 
==676199== NVPROF is profiling process 676199, command: ./a.out
Kernel launch parameters || grid : 4096, block : 1024 
==676199== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "void reduction_gmem_benchmark<unsigned int=1024>(int*, int*, int)" (done)
GPU result : 18874356 , CPU result : 18874356 
GPU and CPU results are same 
==676199== Profiling application: ./a.out
==676199== Profiling result:
==676199== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GT 1030 (0)"
    Kernel: void reduction_gmem_benchmark<unsigned int=1024>(int*, int*, int)
          1                          gld_transactions                  Global Load Transactions     4734978     4734978     4734978
          1                          gst_transactions                 Global Store Transactions      593920      593920      593920
$ sudo nvprof --metrics gld_transactions,gst_transactions ./a.out 1
Running parallel reduction with complete unrolling kernel 
==676227== NVPROF is profiling process 676227, command: ./a.out 1
Kernel launch parameters || grid : 4096, block : 1024 
==676227== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "void reduction_smem<unsigned int=1024>(int*, int*, int)" (done)
GPU result : 262140 , CPU result : 18874356 
GPU and CPU results are different 
==676227== Profiling application: ./a.out 1
==676227== Profiling result:
==676227== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GT 1030 (0)"
    Kernel: void reduction_smem<unsigned int=1024>(int*, int*, int)
          1                          gld_transactions                  Global Load Transactions     6045698     6045698     6045698
          1                          gst_transactions                 Global Store Transactions      495616      495616      495616
```
- With GT1030, reduction in transactions is not significant but higher reduction is found in better performing GPU cards

### 53. Synchronization in CUDA
- Ref: https://stackoverflow.com/questions/5241472/cuda-substituting-syncthreads-instead-of-threadfence-difference
- `__threadfence_block()` stalls current thread until all writes to shared memory are visible to other threads from the same block. It prevents the compiler from optimising by caching shared memory writes in registers. It does not synchronise the threads and it is not necessary for all threads to actually reach this instruction.
- `__threadfence()` stalls current thread until all writes to shared and global memory are visible to all other threads.
- `__syncthreads()` must be reached by all threads from the block (e.g. no divergent if statements) and ensures that the code preceding the instruction is executed before the instructions following it, for all threads in the block.

### 54. Matrix transpose with shared memory
- 4 memory access or indices required
  - Index to access input array in row major format
    - ix = blockIdx.x * blockDim.x + threadIdx.x
    - iy = blockIdx.y * blockDim.y + threadIdx.y
    - index = iy*nx + ix (nx,ny are matrix size)
  - Index to store memory to shared memory in row major format
    - tile[threadIdx.y][threadIdx.x]
  - Index to load memory from shared memory in column major format
    - 1d_index = threadIdx.y*blockDim.x + threadIdx.x
    - i_row = 1d_index/blockDim.y
    - i_col = 1d_index%blockDim.y
    - tile[i_col][i_row]
  - Index to store memory to output array in row major format
    - out_ix = blockIdx.y*blockDim.y + i_col
    - out_iy = blockIdx.x*blockDim.x + i_row
    - out_index = iy*ny + ix
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "cuda_common.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BDIMX 64
#define BDIMY 8
#define IPAD 2
__global__ void transpose_read_raw_write_column_benchmark(int * mat, 
	int* transpose, int nx, int ny)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	if (ix < nx && iy < ny)
	{
		//read by row, write by col
		transpose[ix * ny + iy] = mat[iy * nx + ix];
	}
}
__global__ void transpose_smem(int * in, int* out, int nx, int ny)
{
	__shared__ int tile[BDIMY][BDIMX];
	//input index
	int ix, iy, in_index;
	//output index
	int i_row, i_col, _1d_index, out_ix, out_iy, out_index;
	//ix and iy calculation for input index
	ix = blockDim.x * blockIdx.x + threadIdx.x;
	iy = blockDim.y * blockIdx.y + threadIdx.y;
	//input index
	in_index = iy * nx + ix;
	//1D index calculation fro shared memory
	_1d_index = threadIdx.y * blockDim.x + threadIdx.x;
	//col major row and col index calcuation
	i_row = _1d_index / blockDim.y;
	i_col = _1d_index % blockDim.y;
	//coordinate for transpose matrix
	out_ix = blockIdx.y * blockDim.y + i_col;
	out_iy = blockIdx.x * blockDim.x + i_row;
	//output array access in row major format
	out_index = out_iy * ny + out_ix;
	if (ix < nx && iy < ny)
	{
		//load from in array in row major and store to shared memory in row major
		tile[threadIdx.y][threadIdx.x] = in[in_index];
		//wait untill all the threads load the values
		__syncthreads();
		out[out_index] = tile[i_col][i_row];
	}
}
__global__ void transpose_smem_pad(int * in, int* out, int nx, int ny)
{
	__shared__ int tile[BDIMY][BDIMX + IPAD];
	//input index
	int ix, iy, in_index;
	//output index
	int i_row, i_col, _1d_index, out_ix, out_iy, out_index;
	//ix and iy calculation for input index
	ix = blockDim.x * blockIdx.x + threadIdx.x;
	iy = blockDim.y * blockIdx.y + threadIdx.y;
	//input index
	in_index = iy * nx + ix;
	//1D index calculation fro shared memory
	_1d_index = threadIdx.y * blockDim.x + threadIdx.x;
	//col major row and col index calcuation
	i_row = _1d_index / blockDim.y;
	i_col = _1d_index % blockDim.y;
	//coordinate for transpose matrix
	out_ix = blockIdx.y * blockDim.y + i_col;
	out_iy = blockIdx.x * blockDim.x + i_row;
	//output array access in row major format
	out_index = out_iy * ny + out_ix;
	if (ix < nx && iy < ny)
	{
		//load from in array in row major and store to shared memory in row major
		tile[threadIdx.y][threadIdx.x] = in[in_index];
		//wait untill all the threads load the values
		__syncthreads();
		out[out_index] = tile[i_col][i_row];
	}
}
__global__ void transpose_smem_pad_unrolling(int * in, int* out, int nx, int ny)
{
	__shared__ int tile[BDIMY * (2 * BDIMX + IPAD)];
	//input index
	int ix, iy, in_index;
	//output index
	int i_row, i_col, _1d_index, out_ix, out_iy, out_index;
	//ix and iy calculation for input index
	ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
	iy = blockDim.y * blockIdx.y + threadIdx.y;
	//input index
	in_index = iy * nx + ix;
	//1D index calculation fro shared memory
	_1d_index = threadIdx.y * blockDim.x + threadIdx.x;
	//col major row and col index calcuation
	i_row = _1d_index / blockDim.y;
	i_col = _1d_index % blockDim.y;
	//coordinate for transpose matrix
	out_ix = blockIdx.y * blockDim.y + i_col;
	out_iy = 2 * blockIdx.x * blockDim.x + i_row;
	//output array access in row major format
	out_index = out_iy * ny + out_ix;
	if (ix < nx && iy < ny)
	{
		int row_idx = threadIdx.y * (2 * blockDim.x + IPAD) + threadIdx.x;
		//load from in array in row major and store to shared memory in row major
		tile[row_idx] = in[in_index];
		tile[row_idx+ BDIMX] = in[in_index + BDIMX];
		//wait untill all the threads load the values
		__syncthreads();
		int col_idx = i_col * (2 * blockDim.x + IPAD) + i_row;
		out[out_index] = tile[col_idx];
		out[out_index + ny* BDIMX] = tile[col_idx + BDIMX];
	}
}
int main(int argc, char** argv)
{
	//default values for variabless
	int nx = 1024;
	int ny = 1024;
	int block_x = BDIMX;
	int block_y = BDIMY;
	int kernel_num = 0;
	//set the variable based on arguments
	if (argc > 1)
		nx = 1 << atoi(argv[1]);
	if (argc > 2)
		ny = 1 << atoi(argv[2]);
	if (argc > 3)
		block_x = 1 << atoi(argv[3]);
	if (argc > 4)
		block_y = 1 <<atoi(argv[4]);
	int size = nx * ny;
	int byte_size = sizeof(int*) * size;
	printf("Matrix transpose for %d X % d matrix with block size %d X %d \n",nx,ny,block_x,block_y);
	int * h_mat_array = (int*)malloc(byte_size);
	int * h_trans_array = (int*)malloc(byte_size);
	int * h_ref = (int*)malloc(byte_size);
	initialize(h_mat_array,size ,INIT_ONE_TO_TEN);
	//matirx transpose in CPU
	mat_transpose_cpu(h_mat_array, h_trans_array, nx, ny);
	int * d_mat_array, *d_trans_array;
	gpuErrchk(cudaMalloc((void**)&d_mat_array, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_trans_array, byte_size));
	gpuErrchk(cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(d_trans_array, 0, byte_size));
	dim3 blocks(block_x, block_y);
	dim3 grid(nx/block_x, ny/block_y);
	printf("Launching smem kernel \n");
	transpose_smem <<< grid, blocks>> > (d_mat_array,d_trans_array,nx, ny);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost));
	compare_arrays(h_ref, h_trans_array,size);
	printf("Launching benchmark kernel \n");
	cudaMemset(d_trans_array,0, byte_size);
	transpose_read_raw_write_column_benchmark << < grid, blocks >> > (d_mat_array, d_trans_array, nx, ny);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost));
	compare_arrays(h_ref, h_trans_array, size);
	printf("Launching smem padding kernel \n");
	cudaMemset(d_trans_array, 0, byte_size);
	transpose_smem_pad << < grid, blocks >> > (d_mat_array, d_trans_array, nx, ny);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost));
	compare_arrays(h_ref, h_trans_array, size);
	printf("Launching smem padding and unrolling kernel \n");
	cudaMemset(d_trans_array, 0, byte_size);
	grid.x = grid.x / 2;
	transpose_smem_pad_unrolling << < grid, blocks >> > (d_mat_array, d_trans_array, nx, ny);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost));
	compare_arrays(h_ref, h_trans_array, size);
	cudaFree(d_trans_array);
	cudaFree(d_mat_array);
	free(h_ref);
	free(h_trans_array);
	free(h_mat_array);
	gpuErrchk(cudaDeviceReset());
	return EXIT_SUCCESS;
}
```    
- Profiling:
```bash
$ sudo nvprof --metrics gld_transactions,gst_transactions,shared_load_transactions_per_request,shared_store_transactions_per_request  ./a.out 
Matrix transpose for 1024 X  1024 matrix with block size 64 X 8 
==735163== NVPROF is profiling process 735163, command: ./a.out
Launching smem kernel 
==735163== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "transpose_smem(int*, int*, int, int)" (done)
Arrays are same 
Launching benchmark kernel 
Replaying kernel "transpose_read_raw_write_column_benchmark(int*, int*, int, int)" (done)
Arrays are same al events
Launching smem padding kernel 
Replaying kernel "transpose_smem_pad(int*, int*, int, int)" (done)
Arrays are same al events
Launching smem padding and unrolling kernel 
Replaying kernel "transpose_smem_pad_unrolling(int*, int*, int, int)" (done)
Arrays are same al events
==735163== Profiling application: ./a.out
==735163== Profiling result:
==735163== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GT 1030 (0)"
    Kernel: transpose_smem_pad(int*, int*, int, int)
          1                          gld_transactions                       Global Load Transactions      524290      524290      524290
          1                          gst_transactions                      Global Store Transactions      262144      262144      262144
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    2.000000    2.000000    2.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
    Kernel: transpose_smem(int*, int*, int, int)
          1                          gld_transactions                       Global Load Transactions      524290      524290      524290
          1                          gst_transactions                      Global Store Transactions      262144      262144      262144
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    8.000000    8.000000    8.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
    Kernel: transpose_smem_pad_unrolling(int*, int*, int, int)
          1                          gld_transactions                       Global Load Transactions      524290      524290      524290
          1                          gst_transactions                      Global Store Transactions      262144      262144      262144
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    2.000000    2.000000    2.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
    Kernel: transpose_read_raw_write_column_benchmark(int*, int*, int, int)
          1                          gld_transactions                       Global Load Transactions      524290      524290      524290
          1                          gst_transactions                      Global Store Transactions     1048576     1048576     1048576
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
```
- As shared memory store is used, gst store decreases significantly

### 55. CUDA constant memory
- Special purpose memory used for data that is read-only from device and accessed uniformly by threads in a warp
  - Read only from devce
  - Read/write from host
  - Must be initialized from the host
- `__constant__`
- cudaMemcpyToSymbol()
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "cuda_common.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define c0 1
#define c1 2
#define c2 3
#define c3 4
#define c4 5
#define RADIUS 4
#define BDIM 128
//constant memory declaration
__constant__ int coef[9];
//stencil calculation in host side
void host_const_calculation(int * in, int * out, int size)
{
	for (int i = 0; i < size; i++)
	{

		if (i < RADIUS)
		{
			out[i] = in[i + 4] * c0
				+ in[i + 3] * c1
				+ in[i + 2] * c2
				+ in[i + 1] * c3
				+ in[i] * c4;
			if (i == 3)
			{
				out[i] += in[2] * c3;
				out[i] += in[1] * c2;
				out[i] += in[0] * c1;
			}
			else if (i == 2)
			{
				out[i] += in[1] * c3;
				out[i] += in[0] * c2;
			}
			else if (i == 1)
			{
				out[i] += in[0] * c3;
			}
		}
		else if ((i + RADIUS) >= size)
		{
			out[i] = in[i - 4] * c0
				+ in[i - 3] * c1
				+ in[i - 2] * c2
				+ in[i - 1] * c3
				+ in[i] * c4;
			if (i == size - 4)
			{
				out[i] += in[size - 3] * c3;
				out[i] += in[size - 2] * c2;
				out[i] += in[size - 1] * c1;
			}
			else if (i == size -3)
			{
				out[i] += in[size - 2] * c3;
				out[i] += in[size - 1] * c2;
			}
			else if (i == size - 2)
			{
				out[i] += in[size - 1] * c3;
			}
		}
		else
		{
			out[i] = (in[i - 4] + in[i + 4])*c0
				+ (in[i - 3] + in[i + 3])*c1
				+ (in[i - 2] + in[i + 2])*c2
				+ (in[i - 1] + in[i + 1])*c3
				+ in[i] * c4;
		}
	}
}
//setting up constant memory from host
void setup_coef_1()
{
	const int h_coef[] = { c0,c1,c2,c3,c4,c3,c2,c1,c0 };
	cudaMemcpyToSymbol(coef, h_coef, (9) * sizeof(float));
}
__global__ void constant_stencil_smem_test(int * in, int * out, int size)
{
	//shared mem declaration
	__shared__ int smem[BDIM + 2 * RADIUS];
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int bid = blockIdx.x;
	int num_of_blocks = gridDim.x;
	int value = 0;
	if (gid < size)
	{
		//index with offset
		int sidx = threadIdx.x + RADIUS;
		//load data to shared mem
		smem[sidx] = in[gid];
		if (bid != 0 && bid != (num_of_blocks - 1))
		{
			if (threadIdx.x < RADIUS)
			{
				smem[sidx - RADIUS] = in[gid - RADIUS];
				smem[sidx + BDIM] = in[gid + BDIM];
			}
		}
		else if (bid == 0)
		{
			if (threadIdx.x < RADIUS)
			{
				smem[sidx - RADIUS] = 0;
				smem[sidx + BDIM] = in[gid + BDIM];
			}
		}
		else
		{
			if (threadIdx.x < RADIUS)
			{
				smem[sidx - RADIUS] = in[gid - RADIUS];
				smem[sidx + BDIM] = 0;
			}
		}
		 //wait untill all the threads in block finish storing smem
		__syncthreads();
		value += smem[sidx - 4] * coef[0];
		value += smem[sidx - 3] * coef[1];
		value += smem[sidx - 2] * coef[2];
		value += smem[sidx - 1] * coef[3];
		value += smem[sidx - 0] * coef[4];
		value += smem[sidx + 1] * coef[5];
		value += smem[sidx + 2] * coef[6];
		value += smem[sidx + 3] * coef[7];
		value += smem[sidx + 4] * coef[8];
		out[gid] = value;
	}
}
int main(int argc, char ** argv)
{
	int size = 1 << 22;
	int byte_size = sizeof(int) * size;
	int block_size = BDIM;
	int * h_in, *h_out, *h_ref;
	h_in = (int*)malloc(byte_size);
	h_out = (int*)malloc(byte_size);
	h_ref = (int*)malloc(byte_size);
	initialize(h_in, size, INIT_ONE);
	int * d_in, *d_out;
	cudaMalloc((void**)&d_in, byte_size);
	cudaMalloc((void**)&d_out, byte_size);
	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
	cudaMemset(d_out, 0, byte_size);
	setup_coef_1();
	dim3 blocks(block_size);
	dim3 grid(size / blocks.x);
	constant_stencil_smem_test << < grid, blocks >> > (d_in, d_out, size);
	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
	host_const_calculation(h_in, h_out, size);
	compare_arrays(h_ref, h_out, size);
	cudaFree(d_out);
	cudaFree(d_in);
	free(h_ref);
	free(h_out);
	free(h_in);
	return 0;
}
```

### 56. Matrix transpose with shared memory padding
- Matrix transpose with shared memory + padding + unrolling
- The code is shown above at ch 54

### 57. Warp shuffle instructions
- The shutffle insruction allows threads to directy read another thread's register, while they are in the same warp
- Lower latency than shared memory and does not consume extra memory
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#define ARRAY_SIZE 32
__global__ void test_shfl_broadcast_32(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_sync(0xffffffff, x, 3, 32);
	out[threadIdx.x] = y;
}
__global__ void test_shfl_broadcast_16(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_sync(0xffffffff, x, 3, 16);
	out[threadIdx.x] = y;
}
__global__ void test_shfl_up(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_up_sync(0xffffffff, x, 3);
	out[threadIdx.x] = y;
}
__global__ void test_shfl_down(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_down_sync(0xffffffff, x, 3);
	out[threadIdx.x] = y;
}
//__global__ void test_shfl_shift_around(int * in, int *out, int offset)
//{
//	int x = in[threadIdx.x];
//	int y = __shfl_sync(0xffffffff, x, threadIdx.x + offset);
//	out[threadIdx.x] = y;
//}
__global__ void test_shfl_xor_butterfly(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_xor_sync(0xffffffff, x, 1, 32);
	out[threadIdx.x] = y;
}
int main(int argc, char ** argv)
{
	int size = ARRAY_SIZE;
	int byte_size = size * sizeof(int);
	int * h_in = (int*)malloc(byte_size);
	int * h_ref = (int*)malloc(byte_size);
	for (int i = 0; i < size; i++)
	{
		h_in[i] = i;
	}
	int * d_in, *d_out;
	cudaMalloc((int **)&d_in, byte_size);
	cudaMalloc((int **)&d_out, byte_size);
	dim3 block(size);
	dim3 grid(1);
	//broadcast 32
	printf("shuffle broadcast 32 \n");
	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
	test_shfl_broadcast_32 << < grid, block >> > (d_in, d_out);
	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
	print_array(h_in, size);
	print_array(h_ref, size);
	//broadcast 16
	printf("shuffle broadcast 16 \n");
	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
	test_shfl_broadcast_16 << < grid, block >> > (d_in, d_out);
	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
	print_array(h_in, size);
	print_array(h_ref, size);
	printf("\n");
	//up
	printf("shuffle up \n");
	cudaMemset(d_out, 0, byte_size);
	test_shfl_up << < grid, block >> > (d_in, d_out);
	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
	print_array(h_in, size);
	print_array(h_ref, size);
	printf("\n");
	//down
	printf("shuffle down \n");
	cudaMemset(d_out, 0, byte_size);
	test_shfl_down << < grid, block >> > (d_in, d_out);
	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
	print_array(h_in, size);
	print_array(h_ref, size);
	printf("\n");
	//shift around
	printf("shift around \n");
	cudaMemset(d_out, 0, byte_size);
	test_shfl_shift_around << < grid, block >> > (d_in, d_out, 2);
	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
	print_array(h_in, size);
	print_array(h_ref, size);
	printf("\n");
	//shuffle xor butterfly
	printf("shuffle xor butterfly \n");
	cudaMemset(d_out, 0, byte_size);
	test_shfl_xor_butterfly << < grid, block >> > (d_in, d_out);
	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
	print_array(h_in, size);
	print_array(h_ref, size);
	printf("\n");
	cudaFree(d_out);
	cudaFree(d_in);
	free(h_ref);
	free(h_in);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
```

### 58. Parallel reduction with warp shuffle instructions
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "cuda_common.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_SIZE 128
#define FULL_MASK 0xffffffff
template<unsigned int iblock_size>
__global__ void reduction_smem_benchmark(int * input, int * temp, int size)
{
	__shared__ int smem[BLOCK_SIZE];
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;
	smem[tid] = i_data[tid];
	__syncthreads();
	// in-place reduction in shared memory   
	if (blockDim.x >= 1024 && tid < 512)
		smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256)
		smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128)
		smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)
		smem[tid] += smem[tid + 64];
	__syncthreads();
	//unrolling warp
	if (tid < 32)
	{
		volatile int * vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 1];
	}
	if (tid == 0)
	{
		temp[blockIdx.x] = smem[0];
	}
}
template<unsigned int iblock_size>
__global__ void reduction_smem_warp_shfl(int * input, int * temp, int size)
{
	__shared__ int smem[BLOCK_SIZE];
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;
	smem[tid] = i_data[tid];
	__syncthreads();
	// in-place reduction in shared memory   
	if (blockDim.x >= 1024 && tid < 512)
		smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256)
		smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128)
		smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)
		smem[tid] += smem[tid + 64];
	__syncthreads();
	if (blockDim.x >= 64 && tid < 32)
		smem[tid] += smem[tid + 32];
	__syncthreads();
	int local_sum = smem[tid];
	//unrolling warp
	if (tid < 32)
	{
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 16);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 8);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 4);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 2);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 1);
	}
	if (tid == 0)
	{
		temp[blockIdx.x] = local_sum;
	}
}
int main(int argc, char ** argv)
{
	printf("Running parallel reduction with complete unrolling kernel \n");
	int kernel_index = 0;
	if (argc > 1)
	{
		kernel_index = 1;
	}
	int size = 1 << 25;
	int byte_size = size * sizeof(int);
	int block_size = BLOCK_SIZE;
	int * h_input, *h_ref;
	h_input = (int*)malloc(byte_size);
	initialize(h_input, size);
	int cpu_result = reduction_cpu(h_input, size);
	dim3 block(block_size);
	dim3 grid((size / block_size));
	printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);
	int temp_array_byte_size = sizeof(int)* grid.x;
	h_ref = (int*)malloc(temp_array_byte_size);
	int * d_input, *d_temp;
	printf(" \nreduction with shared memory\n ");
	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));
	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,cudaMemcpyHostToDevice));
	reduction_smem_benchmark <1024> << < grid, block >> > (d_input, d_temp, size);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}
	compare_results(gpu_result, cpu_result);
	//warp shuffle implementation
	printf(" \nreduction with warp shuffle instructions \n ");
	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));
	reduction_smem_warp_shfl <1024> << < grid, block >> > (d_input, d_temp, size);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
	gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}
	compare_results(gpu_result, cpu_result);
	gpuErrchk(cudaFree(d_input));
	gpuErrchk(cudaFree(d_temp));
	free(h_input);
	free(h_ref);
	gpuErrchk(cudaDeviceReset());
	return 0;
}
```

### 59. Summary
- Shared memory
  - Share memory banks
  - Access modes
  - Bank conflicts
  - Statically and dynamically declared shared memory
  - Shared memory padding

## Section 5: CUDA Streams

### 60. Introduction to CUDA streams and events
- Launch multiple kernels, transferring memory b/w kernels by overlapping execution
- CUDA stream: a sequence of commands that execute in order
- Overlapping is the key to transfer memory within device
- Synchronous vs asynchronous function calls
  - Synchronous functions block the host thread until they complete
  - Asynchronous functions return control to the host immediately after being called
- NULL stream: default stream that kernel launches and data transfers use
  - Implicitly declared stream
- Tasks that can operate concurrently
  - Computation on the host
  - Computation on the device
  - Memory transfer from the host to the device
  - Memory transfer from the device to the host
  - Memory transfer within the memory of the device
  - Memory transfer among devices

### 61. How to use CUDA asynchronous functions
- cudaMemCpyAsync()
  - Host pointers should be pinned memory
  - Stream is assigned
- 3_simple_cuda_stream_modified.cu produced different results than the lecture. Two kernel executions actually don't overlap
```c
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
__global__ void stream_test_modified(int* in, int * out, int size)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid < size)
	{
		//THIS FOR LOOP IS ONLY FOR VISUALIZING PURPOSE
		for (int i = 0; i < 25; i++)
		{
			out[gid] = in[gid] + (in[gid] - 1) * (gid % 10);
		}
	}
}
int main(int argc, char ** argv)
{
	int size = 1 << 22;
	int byte_size = size * sizeof(int);
	//initialize host pointer
	int* h_in, *h_ref, *h_in2, *h_ref2;
	cudaMallocHost((void**)&h_in,byte_size);
	cudaMallocHost((void**)&h_ref, byte_size);
	cudaMallocHost((void**)&h_in2, byte_size);
	cudaMallocHost((void**)&h_ref2, byte_size);
	initialize(h_in, INIT_RANDOM);
	initialize(h_in2, INIT_RANDOM);
	//allocate device pointers
	int * d_in, *d_out, *d_in2, *d_out2;
	cudaMalloc((void**)&d_in, byte_size);
	cudaMalloc((void**)&d_out, byte_size);
	cudaMalloc((void**)&d_in2, byte_size);
	cudaMalloc((void**)&d_out2, byte_size);
	cudaStream_t str,str2;
	cudaStreamCreate(&str);
	cudaStreamCreate(&str2);
	//kernel launch
	dim3 block(128);
	dim3 grid(size / block.x);
	//transfer data from host to device
	cudaMemcpyAsync(d_in, h_in, byte_size, cudaMemcpyHostToDevice,str);
	stream_test_modified << <grid, block,0,str >> > (d_in, d_out, size);
	cudaMemcpyAsync(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost,str);
	cudaMemcpyAsync(d_in2, h_in2, byte_size, cudaMemcpyHostToDevice, str2);
	stream_test_modified << <grid, block, 0, str2 >> > (d_in2, d_out2, size);
	cudaMemcpyAsync(h_ref2, d_out2, byte_size, cudaMemcpyDeviceToHost, str2);
	cudaStreamSynchronize(str);
	cudaStreamDestroy(str);
	cudaStreamSynchronize(str2);
	cudaStreamDestroy(str2);
	cudaDeviceReset();
	return 0;
}
```

### 62. How to use CUDA streams
- Objective: overlap kernel executions so we can reduce the overhead of memory transfer
  - Launch multiple kernels
    - Concurrent kernel executions
  - Perform asynchronous memory transfer
- Default stream will execute kernel functions one by one
  - No parallel execution
- concurrent.cu
  - std::cout not working in the device
```c
#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
using std::cout;
using std::endl;
__global__ void simple_kernel() {
  printf("Hello from the kernel\n");
}
int main(int argc, char** argv) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if (deviceProp.concurrentKernels == 0)
    cout <<"> GPU does not support concurrent kernel execution\n";
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
  return 0;
}
```
  - Kernels are overlapping

### 63. Overlapping memory transfer and kernel execution
- ? Kernels didn't overlap?
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
void sumArraysOnHostx(int *A, int *B, int *C, const int N)
{
	for (int idx = 0; idx < N; idx++)
		C[idx] = A[idx] + B[idx];
}
__global__ void sum_array_overlap(int * a, int * b, int * c, int N)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < N)
	{
		c[gid] = a[gid] + b[gid];
	}
}
int main()
{
	int size = 1 << 25;
	int block_size = 128;
	//number of bytes needed to hold element count
	size_t NO_BYTES = size * sizeof(int);
	int const NUM_STREAMS = 8;
	int ELEMENTS_PER_STREAM = size / NUM_STREAMS;
	int BYTES_PER_STREAM = NO_BYTES / NUM_STREAMS;
	// host pointers
	int *h_a, *h_b, *gpu_result, *cpu_result;
	//allocate memory for host size pointers
	cudaMallocHost((void**)&h_a,NO_BYTES);
	cudaMallocHost((void**)&h_b, NO_BYTES);
	cudaMallocHost((void**)&gpu_result, NO_BYTES);
	cpu_result = (int *)malloc(NO_BYTES);
	//initialize h_a and h_b arrays randomly
	initialize(h_a, INIT_ONE_TO_TEN);
	initialize(h_b, INIT_ONE_TO_TEN);
	//summation in CPU
	sumArraysOnHostx(h_a, h_b, cpu_result, size);
	int *d_a, *d_b, *d_c;
	cudaMalloc((int **)&d_a, NO_BYTES);
	cudaMalloc((int **)&d_b, NO_BYTES);
	cudaMalloc((int **)&d_c, NO_BYTES);
	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		cudaStreamCreate(&streams[i]);
	}
	//kernel launch parameters
	dim3 block(block_size);
	dim3 grid(ELEMENTS_PER_STREAM/block.x + 1);
	int offset = 0;
	for (int  i = 0; i < NUM_STREAMS; i++)
	{
		offset = i * ELEMENTS_PER_STREAM;
		cudaMemcpyAsync(&d_a[offset], &h_a[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice,streams[i]);
		cudaMemcpyAsync(&d_b[offset], &h_b[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice,streams[i]);
		sum_array_overlap << <grid, block, 0, streams[i] >> > (&d_a[offset], &d_b[offset], &d_c[offset], size);
		cudaMemcpyAsync(&gpu_result[offset], &d_c[offset], BYTES_PER_STREAM, cudaMemcpyDeviceToHost,streams[i]);
	}
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
	cudaDeviceSynchronize();
	//validity check
	compare_arrays(cpu_result, gpu_result, size);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(gpu_result);
	free(cpu_result);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
```

### 64. Stream synchronization and blocking behaviours of NULL stream
- NULL stream can block non-null stream
- Non-NULL streams
  - blocking : can be blocked by NULL stream
  - non-blocking : cannot be blocked by NULL stream
- Streams created by cudaStreamCreate() are blocking streams
1. For blocking streams only
```c
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
2. For blocking streams + default stream
```c
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
3. For one blocking stream + one nonblocking stream + default stream
```c
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
4. For nonblocking streams + default stream
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

### 66. CUDA events and timing with CUDA events
- CUDA event : a marker in CUDA stream
  - Can sync stream execution
  - Can Monitor device progress
  - Can measure the execution time of the kernel
- cudaEventCreate()/cudaEventDestroy()
- cudaEventRecord() will queue an event
- cudaEventElapsedTime() will return the elapsed time in msec
```c
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
```c
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

### 68. Introduction to different types of instructions in CUDA
- Not all instructions are created equally
- MAD(Multiply-Add operations)
  - double v = x \* y + z
  - MAD optimization has less numerical accuracy
   
### 69. Floating point operations
- CUDA follows IEEE standard754
- Single vs Double FPE
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
```asm
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
```asm
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
```asm
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

### Section 71. Atomic functions
- Atomic instructions
  - A single uninterruptable operation with no interference from other threads
  - When completed, it is not affected by the access from other threads
```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
__global__ void incr(int *ptr)
{
	/*int temp = *ptr;
	temp = temp + 1;
	*ptr = temp;*/
	atomicAdd(ptr,1);
}
int main()
{
	int value = 0;	
	int SIZE = sizeof(int);
	int ref = -1;
	int *d_val;
	cudaMalloc((void**)&d_val, SIZE);
	cudaMemcpy(d_val, &value, SIZE, cudaMemcpyHostToDevice);
	incr << <1, 32 >> > (d_val);
	cudaDeviceSynchronize();
	cudaMemcpy(&ref,d_val,SIZE, cudaMemcpyDeviceToHost);
	printf("Updated value : %d \n",ref);
	cudaDeviceReset();
	return 0;
}
```
- Very low performance

## Section 7: Parallel patterns and applications

### 72. Scan algorithm introduction
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

### 74. Work efficient parallel exclusive scan
- Balanced tree model
- Reduction + down sweep (exclusive scan) phases
- Workload: 2*(N-1)

### 75. Work efficient parallel inclusive scan
- Reduction + down sweep (inclusive scan) phases
- Workload: 2*(N-1)
```c
#include "scan.cuh"
#define BLOCK_SIZE 512
void inclusive_scan_cpu(int *input, int *output, int size)
{
	output[0] = input[0];
	for (int i = 1; i < size; i++)
	{
		output[i] = output[i - 1] + input[i];
	}
}
__global__ void naive_inclusive_scan_single_block(int *input, int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < size)
	{
		for (int stride = 1; stride <= tid; stride *= 2)
		{
			input[gid] += input[gid - stride];
			__syncthreads();
		}
	}
}
__global__ void efficient_inclusive_scan_single_block(int *input,int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < size)
	{
		for (int stride = 1; stride <= tid; stride *= 2)
		{
			input[gid] += input[gid - stride];
			__syncthreads();
		}
	}
}
__global__ void efficient_inclusive_scan(int *input, int * aux ,int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < size)
	{
		for (int stride = 1; stride <= tid; stride *= 2)
		{
			input[gid] += input[gid - stride];
			__syncthreads();
		}
	}
}
__global__ void sum_aux_values(int *input,  int *aux, int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < size)
	{
		for (int i = 0; i < blockIdx.x; i++)
		{
			input[gid] += aux[i];
			__syncthreads();
		}
	}
}
int main(int argc, char**argv)
{
	printf("Scan algorithm execution starterd \n");
	int input_size = 1 << 10;
	if (argc > 1)
	{
		input_size = 1 << atoi(argv[1]);
	}
	const int byte_size = sizeof(int) * input_size;
	int * h_input, *h_output, *h_ref, *h_aux;
	clock_t cpu_start, cpu_end, gpu_start, gpu_end;
	h_input = (int*)malloc(byte_size);
	h_output = (int*)malloc(byte_size);
	h_ref = (int*)malloc(byte_size);
	initialize(h_input, input_size, INIT_ONE);
	cpu_start = clock();
	inclusive_scan_cpu(h_input, h_output, input_size);
	cpu_end = clock();
	int *d_input, *d_aux;
	cudaMalloc((void**)&d_input, byte_size);
	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
	dim3 block(BLOCK_SIZE);
	dim3 grid(input_size/ block.x);
	int aux_byte_size = block.x * sizeof(int);
	cudaMalloc((void**)&d_aux , aux_byte_size);
	h_aux = (int*)malloc(aux_byte_size);
	naive_inclusive_scan_single_block << <grid, block >> > (d_input, input_size);
	cudaDeviceSynchronize();
	cudaMemcpy(h_aux, d_aux, aux_byte_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost);
	print_arrays_toafile(h_ref, input_size, "input_array.txt");
	for (int i = 0; i < input_size; i++)
	{
		for (int j = 0; j < i / BLOCK_SIZE ; j++)
		{
			h_ref[i] += h_aux[j];
		}
	}
	print_arrays_toafile(h_aux,grid.x, "aux_array.txt");
	//sum_aux_values << < grid, block >> > (d_input, d_aux, input_size);
	//cudaDeviceSynchronize();
	//cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost );
	//print_arrays_toafile_side_by_side(h_ref, h_output, input_size, "scan_outputs.txt");
	compare_arrays(h_ref, h_output, input_size);
	gpuErrchk(cudaDeviceReset());
	return 0;
}
```

### 76. Parallel scan for large data sets

### 77. Parallel Compact algorithm
- Compaction
```c
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include "common.h"
#include "cuda_common.cuh"
#define BLOCK_SIZE 64
__global__ void scan_for_compact(int * input, int * output_index_array,int* auxiliry_array, int input_size)
{
	int idx = threadIdx.x;
	int gid = blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ int local_input[BLOCK_SIZE];
	if (input[gid] >0)
	{
		local_input[idx] = 1;
	}
	else
	{
		local_input[idx] = 0;
	}
	__syncthreads();
	// reduction phase
	// this can be optimized check wether global memory access for "d" or calculation here is better
	int d = ceilf(log2f(BLOCK_SIZE));
	int denominator = 0;
	int offset = 0;
	//reduction should happen per block
	for (int i = 1; i <= d; i++)
	{
		denominator = 1 << i;
		offset = 1 << (i - 1);
		if (((idx + 1) % denominator) == 0)
		{
			local_input[idx] += local_input[idx - offset];
		}
		__syncthreads();
	}
	////end of reduction phase
	//// start of  down-sweep phase
	if (idx == (BLOCK_SIZE - 1))
	{
		local_input[idx] = 0;
	}
	int temp = 0;
	int sawp_aux = 0;
	for (int i = d; i > 0; i--)
	{
		temp = 1 << i;
		if ((idx != 0) && (idx + 1) % temp == 0)
		{
			sawp_aux = local_input[idx];
			local_input[idx] += local_input[idx - (temp / 2)];
			local_input[idx - (temp / 2)] = sawp_aux;
		}
		__syncthreads();
	}
	//can this be add to if condition at the begining of the down sweep phase 
	if (idx == (BLOCK_SIZE - 1))
	{
		auxiliry_array[blockIdx.x] = local_input[idx];
		//printf("%d \n", auxiliry_array[blockIdx.x]);
	}
	output_index_array[gid] = local_input[idx];
}
__global__ void scan_summation_for_compact(int * output_index_array, int * auxiliry_array, int input_size)
{
	int idx = threadIdx.x;
	int gid = blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ int local_input[BLOCK_SIZE];
	local_input[idx] = output_index_array[gid];
	__syncthreads();
	for (int i = 0; i < blockIdx.x; i++)
	{
		local_input[idx] += auxiliry_array[i];
	}
	output_index_array[gid] = local_input[idx];
	//printf("gid : %d, value : %d \n", gid, output_index_array[gid]);
}
__global__ void compact_1D_array( int * input, int * output, int * output_index_array, int array_size)
{
	int gid = blockDim.x*blockIdx.x + threadIdx.x;	
	//TO DO handle when gid ==0
	//this is very unefficient in memory management
	if (gid > 0 && gid < array_size)
	{
		if (output_index_array[gid] != output_index_array[gid - 1])
		{
			//printf("gid : %d , index :%d , value : %d, prev_value : %d \n",gid, output_index_array[gid], input[gid], input[gid-1]);
			output[output_index_array[gid]] = input[gid-1];
		}
	}
}
void run_compact()
{
	int input_size = 1 << 7;
	int input_byte_size = input_size * sizeof(int);
	dim3 block(BLOCK_SIZE);
	dim3 grid(input_size / block.x);
	int aux_byte_size = sizeof(int)*grid.x;
	int* h_input, *h_ref, *h_aux_ref, *h_output;
	h_input = (int*)malloc(input_byte_size);
	h_ref = (int*)malloc(input_byte_size);
	h_aux_ref = (int*)malloc(aux_byte_size);
	for (int i = 0; i < input_size; i++)
	{
		if (i % 5 == 0)
		{
			h_input[i] = i;
		}
		else
		{
			h_input[i] = 0;
		}
	}
	int * d_input, *d_output_index_array, *d_aux, *d_sum_input, *d_sum_aux, *d_output;
	gpuErrchk(cudaMalloc((int**)&d_input, input_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_output_index_array, input_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_aux, aux_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_sum_input, input_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_sum_aux, aux_byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, input_byte_size, cudaMemcpyHostToDevice));
	scan_for_compact << <grid, block >> > (d_input, d_output_index_array,d_aux, input_size);
	gpuErrchk(cudaDeviceSynchronize());
	//gpuErrchk(cudaMemcpy(d_sum_input, d_output_index_array, input_byte_size, cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(d_sum_aux, d_aux, aux_byte_size, cudaMemcpyDeviceToDevice));
	scan_summation_for_compact << <grid, block >> > (d_output_index_array, d_sum_aux, input_size);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_output_index_array, input_byte_size, cudaMemcpyDeviceToHost));
	int compact_output_size = h_ref[input_size - 1];
	int compact_output_byte_size = sizeof(float)*compact_output_size;
	h_output = (int*)malloc(compact_output_byte_size);
	gpuErrchk(cudaMalloc((int**)&d_output, compact_output_byte_size));
	compact_1D_array << <grid, block >> > (d_input, d_output, d_output_index_array, input_size);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_output, d_output, compact_output_byte_size, cudaMemcpyDeviceToHost));
	for (int i=0;i<compact_output_size;i++)
	{
		printf("%d \n",h_output[i]);
	}
	cudaFree(d_sum_input);
	cudaFree(d_sum_aux);
	cudaFree(d_input);
	cudaFree(d_aux);
	free(h_input);
	free(h_aux_ref);
	free(h_ref);
}
//int main()
//{
//	run_compact();
//	system("pause");
//	return 0;
//}
```
## Section 8: Bonus: Introduction to image processing with CUDA
