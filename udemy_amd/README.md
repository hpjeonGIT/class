## Complete AMD GPU Programming for Beginners 12+ Hours
- Scientific Programming School

## Section 1: Introduction

### 1. Introduction
- ROCm ecosystem
  - Frameworks (Tensorflow/Pytorch)
  - Libraries (MIOpen/Blas/RCCL), programming model (HIP)
- ROCm system runtimes
  - HSA Runtime: low level device manipulation
- ROCm programming frameworks
  - HIP
  - OpenCL
- ROCm libraries
  - MIOpen
  - MIOpenGEMM
  - rocBloas, hipBlas
  - rocSparse, hipSparse
  - rocFFT
  - rocRAND
  - RCCL
- ROCm development tools
  - Assembler/disassembler
  - roc-prof, roc-tracer
  - ROCr debug agent
  - rocm-smi
- ROCm multi-versioning support

### 2. Slurm

### 3. AMD GPUs
- A group of 64 threads  in a wavefront (warp)
- A single Compute Unit can hold up to 40 wavefronts
- A workgroup is composed of 1-16 wavefronts
- Per GPU, shader engine: 1-4
- Per SE, CU: 9-16

### 4. AMD Concepts
- Block dispatching
  - GPU's command processor
    - Breaksdown kernels to blocks
    - Dispatches blocks to compute units
  - Block executes on Compute Units
    - Threads from one block execute on the same CU
    - One CU can execute multiple blocks
    - A kernel can have more blocks than the CU can fit
- Memory Access Coalescing
  - Combine memroy accesses to the same cache line
  - Increase effective memory throughput
- Use a following macro
```cpp
#define HIP_ASSERT(x)  (assert(x) == hipSuccess)
```

## Section 2: AMD Architecture

### 5. GPU Architectures
- Graphics Core Next (GCN): 2012-2019
  - Vector Memory instruction
    - Retires befor data return
    - Next instruction can be issued immediately
    - waitcnt maintains dependency
  - GCN microarchitecture
    - Controlling group
      - Command Processor
        - An embedded micro processor
        - Handles requests from the driver
          - Kernel launching
          - Memory copy
          - Cache flushing
        - Not exposed into user control
      - DMA engine
        - Processes memory copy commands
      - Asynchronos Compute Engine (ACE)
        - Kernel dispatching
        - Enables concurrent kernel execution with multiple ACEs
      - Compute Unit: Wavefront patching
      - Work-groups execution
        - Simultaneously vs concurrently
          - Cannot sync all the work-items in a kernel
          - Not all work-groups in a kernel execute simultaneously
        - Workgroups to CU mapping
          - Non-deterministic
          - Do not that a certain work group to CU mapping
      - Compute Unit
        - Instruction is round-robin over SIMDs
        - One per wavefront          
    - Shader group
    - Memory Group
- RDNA: for gaming
- CDNA: Since MI100, for HPC

## Section 3: HIP Programming

### 6. HIP Programming Basics
- At host, use STL or vector
- At device still hipMalloc/hipFree

### 7. Kernels HIP
- `__device__`
  - Executed on GPU
  - Cannot access the grid/thread coordinates (threadIdx). Must be delivered as arguments
  - Can be used with `__noinline__` and `__forceinline__`
    - HIP currently forces all device functions to be inlined
- `__global__`
  - Always return `void`
  - Declares a function as a kernel
    - To be launched by the host on the device using <<<>>>
- `__host__`
  - Default for host functions        
- `__host__ __device__` is not same to `__global__`
- Memory Hierarchy
  - Thread (work-item)
    - threadIdx is a 3component vector of type dim3
    - Readonly
    - 0 <= threadIdx.x/y/z < blockDim.x/y/z
  - Warp (Wavefront)
    - warpSize
      - Number of threads available in a single CU
      - 32 for RDNA
      - 64 for GC/CDNA
      - Block size from kernel launch is advised to be multipe of this
        - 256 is commonly recommended
```cpp
int threadIdx = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*trehadIdx.y + threadIdx.x
int blockIdx = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x
```
  - Block (Workgroup)
  - Grid (NDRange)

### 8. HIP Memory Models
- GPU programs almost always have bottlenecks related to memory
- Data locality
  - Temporal locality: useful data tends to continue to be used
  - Spatial locality: useful data tends to be followed by more useful data
- Global Memory
  - Resides on GPU DRAM
  - Highest latency but largest memory
- Registers
  - In the warp/wavefront
  - Lowest latency but smallest memory
  - Private to each thread
  - 64KB per warp
  - Max number is 255
    - Must avoid memory leak as it spills to global memory
- Shared memory
  - Low latency, small size
  - Private to each block
  - Lasts for the lifetime of a block
  - Thread sync is necessary
  - Max 64kb per block in GCN/RDNA
- Constant memory
  - Read-only
  - Declared with `__constant__` keyword
- To debug kernel apps
  - Use printf()
- Querying compilation environment
  - AMD target: `__HIP_PLATFORM_HCC__`, `__HIP__`, `__HIPCC__`
  - NVCC for Nvidia target: `__HIP_PLATFORM_NVCC__`, `__NVCC__`, `__CUDACC__`
  - `__HIP_DEVICE__COMPILE__` is defined when the code is compiled for device 
  - hipDeviceProp_t: device name, memsize, shared mem size, warpSize, ...
  - hipDeviceArch_t: 32/64bit atomics, warp vote instruction, ...
  - Architectural features on device code
    - `__HIP_ARCH_HaS_*__` such as `__HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__`
- Sample code:
```cpp
#include <hip/hip_runtime_api.h>
#include <iostream>
int main()
{
 hipDeviceProp_t devP;
 hipGetDeviceProperties(&devP, 0);
 std::cout << "name = " << devP.name << std::endl;
 std::cout << "Global mem = " << devP.totalGlobalMem << std::endl;
 std::cout << "Shared mem per block = " << devP.sharedMemPerBlock << std::endl;
 std::cout << "MaxThreads per bloc = " << devP.maxThreadsPerBlock << std::endl;
 std::cout << "Clock Rate = " << devP.clockRate << std::endl;
 std::cout << "warpSize = " << devP.warpSize << std::endl;
 return 0;
}
```  
- Demo:
```bash
$ hipcc query_hipDevice.cpp
$ ./a.out 
name = NVIDIA GeForce GT 1030
Global mem = 2089943040
Shared mem per block = 49152
MaxThreads per bloc = 1024
Clock Rate = 1468000
warpSize = 32
```
- Full query code is at: https://github.com/ROCm/rocm-examples/blob/develop/HIP-Basic/device_query/main.cpp

### 9. HIP Matrix and Syncs
- Thread synchronization and coordination
- Atomic operations
  - Race conditions can be avoided
  - For collaboration or synchronizawtion
    - Collaboration: threads work together to aggregate their work
  - Can be used with both shared memory and global memory
    - Device-specific
- Memory fence
  - Not synchronization
  - `__threadfence_block()`: stalls current thread until all writes by this thread to shared/global memory are visible to other threads **from the same block**
  - `__threadfence()`: stalls current thread until all writes to shared/global memory are visible to all the threads in the block
    - `__threadfence_block()` + global memory write ordering
- Intr-block synchronization
  - `__syncthreads()` syncs the execution of threads inside a block
- There is no explicit way of syncing all threads inside a kernel
  - May launch a new kernel
- Reduction in HIP
  - Operation must be:
    - Associative: (A\Omega B) \Omega C = A\Omega (B\Omega C)
    - Commutative: A|Omega B = B\Omega A
    - Has a null/identity operator: A\Omega o = A
  - Operation: addition, multiplication, min, max
  - Tree based approach
    - Warp-level using `__shfl`
    - Block level using shared memory
```cpp
// warp-level sum
int offset;
for (offset = warpSize/2; offset > 0; offset /=2) data += __shfl_down(data,offset);
```
  - Using shared memory
    - First thread in the warp writes the warp's results into shared memory
```cpp
extern __shared__ scalar_t data_s[];
if (threadId.x %warpSize == 0) data_s[threadId.x/warpSize] = data;
__syncthreads();
for (offset = blockDim.x/(warpSize*2); offset > 0 ; offset /=2) {
  if (threadIdx.x < offset)  data_s[threadIdx.x] += data_s[threadIdx.x + offset];
  __syncthreads();
}
...
// writing block level results  to global memory
if (threadIdx.x == 0) resuls[blockIdx.x] = data_s[0];
```
- Concurrency and Overlapping Operations in HIP
  - HIP has blocking/non-blocking APIs
  - **Stream** defines the ordering of operations
  - HIP Streams
    - Sequence of device operations
    - Scheduling work from the host in a stream is non-blocking
    - Operations from the same stream cannot overlap
  - Default stream
    - All HIP operations are placed in the default (null or zero) stream by default
- HIP Events
  - An abstraction to monitor the progress of operatoins
    - `hipEvet_t`
  - May sync streams

### 10. HIP tools
- rocminfo
  - May not work with Nvidia GPU
  - clinfo as a legacy tool
```bash
$ rocminfo
ROCk module is NOT loaded, possibly no GPU devices
$ clinfo 
Number of platforms:				 2
  Platform Profile:				 FULL_PROFILE
  Platform Version:				 OpenCL 3.0 CUDA 12.6.32
  Platform Name:				 NVIDIA CUDA
  Platform Vendor:				 NVIDIA Corporation
  Platform Extensions:				 cl_khr_global_int32
...
```
- rocprof
  - --hip-trace: HIP runtime trace
  - --hsa-trace: ROCr runtime trace
  - --kfd-trace: KFD Driver trace
  - rocprof --hip-trace ./a.out
  - Will get results.json
  - In chrome, `chrome://tracing` then drag the json file
  - --list-basic: HW counters
  - --list-derive: calculated
  - Metrics
    - SQ_WAVES: number of wavefronts
    - SQ_INSTS_VALU: Number of VALU instruction
    - TCC_HIT: L2 cache hit count
    - TCC_EA_WRREQ: number o write request sent from L2 cache to DRAM
  - GPU profiling
    - input.txt: list of metrics to profile
```txt
pmc: GPUBusy, Wavefronts, TCC_EA_WRREQ_sum, TCC_EA_RDREQ_sum
range: 0:6
gpu: 0 1 2 3 
kernel: copy_kernel
```
      - rocprof -i input.txt -o output.csv ./a.exe
- Debugger
  - rocgdb
  - -g is required when compiled: hipcc -g main.cpp -o a.exe      
  - rocgdb main
  - rocgdb --args main arg1 arg2
  - Commands
    - CPU only: print/p, backtrace/bt
    - Useful commands
      - info/i register
      - disassemble
      - list/l
      - step/s
      - next/n

### 11. HIP Performance Tuning

## Section 4: Nvidia CUDA to AMD HIP conversion

### 12. Hipify
- kernelName <<<grid,block>>> vs hipLaunchKernelGGL(kernel, grid, block)
- hipify-perl
  - hipify-perl file.cu
  - hipify-perl --print-stats file.cu
  - hipify-perl --inplace file.cu
    - f.prehip: original cuda code
    - file.cu: hip code (why?)
    - do not use this option
- Modifying makefiles
  - nvcc -> hipcc
  - cublas -> hipblas
  - /opt/rocm/lib
  - rocm lib path and include path
- hipify-clang
  - Converts CUDA into AST then generates HIP code
- Block size
  - When threads run out of registers
  - Workarounds
    - Reduce the work group size
    - Kernel code must use `__attribute__((amdgpu_flat_work_group_size(<min>,<max>)))` when kernel is launched with a work group size > 256

## Section 5: AMD HIP Libraries

### 13. Libraries
- Using hipBlas/hipSparse will offload to CUDA on Nvidia GPU
  - More portability
- On AMD GPUs, use ROCm libraries

### 14. Advanced Libraries
- Matrix multplication sample:
  - Simple: 1.6s
  - Tiling: 0.76sec
  - rocBLAS: 0.096sec

## Section 6: AMD GPU Multiscaling

### 15. GPU Multiscaling
- HIP Device APIs
  - hipGetDeviceCount
  - hipGetDeviceProperties
  - hipSetDevice
- Multi-GPU programming
  - Intra-Stream
  - Inter-Stream
  - Stream-based multi-GPU programming
    - Create one stream for each GPU
    - Enqueue tasks to the streams
    - Synchronize streams
- RCCL: Broadcast and AllReduce

## Section 7: AMD HIP Programming codes

### 16. Code examples
