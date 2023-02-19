## Summary
- Title: Function Acceleration on FPGA with Vitis-Part 1: Fundamental
- Instructor: Mohammad Hosseinbady

## Section 1: Prologue

1. Introduction
- High Level Synthesis (HLS)
- Xilinx/Vitis community
  - ZYBOZ7 ($309 as of Jan 2023)
  - Or emulation
- C/C++/OpenCL

2. Course structure 
- Xilinx ZYNQ based system
- Vitis high-level synthesis user guide from Xilinx
- https://highlevel-synthesis.com

## Section 2: Embedded Systems

3. Introduction

4. Definition
- Traditional Computing platform
- Cloud computing platform
- Cloud & fog computing platform
  - Fog or Edge: b/w Cloud and end devices
- Embedded systems
  - Embedded GPU: Jetson
  - Embedded ASIC
  - Embedded FPGAs
- This course focuses on using Zynq FPGA-based embedded system targeting edge computing
  - Download Zynq-700 SoC technical reference manual & Zynq ultrascale+ device

5. FPGA Role
- FPGA applications
  - End devices: low response time
  - Edge Devices: accelerating computing power
  - Cloud: high end FPGA to accelerate database power and reduce power consumption
- Xilinx FPGA platforms
  - Zynq/Zynq ultrascale+
    - PYNQ, Edge AI platform, OpenCV, BLAS, ...
      - Vitis Unified SW platform
- End devices: HLS/HDL
- Edge devices/Cloud: HLS
- HLS can be used to describe and design algorithms and logic circuits on end-devices, edge platform and cloud servers
- This course focuses on accelerating functions and algorithms on FPGA-based embedded systems targeting edge platforms
- The complete advanced HLS desgin flow consists of three layers
  - Underlying embedded system hardware platform
  - HLS hardware and software libraries
  - Unified development platform
- Download **Vitis unified software platform documentation**

6. HLS Role
- HLS Driver
  - Compared to HDL, can reduce debug time
  - Existing libraries for ML, vision, ...

7. Zynq
- Programmable Logic (PL)
- Processing system (PS)
  - Cortex A9, NEON, FPU, On-chip memeory, memory subsystem
- DDR memory
- 4 high performance memory port in PL: HP0, HP1, HP2, HP3
  - Are NOT cache-coherent
  - Each HP memory port has 64bits
  - We can choose the design clock frequency for the FPGA accelerator
- Burst data transfer
  - float array A with size of 2048
  - Frequency of 100MHz = 1/100MHz = 10 ns
  - Execution time = 2048*10ns = 20.48 microsec
  - 4 bytes are read on each block cycle then bandwitdh usage = 4*100MB/s = 400MBps 

8. Zynq MPSoC
- Zynq ultrascale
  - Has cache coherent port of HPC0, HPC1
  - Ports have 128bits
  - Ultra96v2

9. Exercises
- May use HP0 and HP1 at Zynq ultrascale MPSoC with 200MHz frequency. What would be the maximum memory bandwidth?
  - Each HP port of Zynq MPSoC has 128bits of data then two ports will provide 256 bits. Therefore 256/8 (in bytes) * 200MHz = 6.4 GBytes/sec
- What is the upper bound of the memory utilization for reading data from DDR memory in Zynq 7000 at the frequency of 150MHz?
  - Zynq 7000 has 4 HP ports for HLS and each port has 64bits then 4 HP ports provide 64*4= 256 bits. Therefore maximum bandwidth = 256/8 * 150 MHz = 4.8GB/sec

## Section 3: Lab structure

10. Introduction
- Determine the required computers and FPGA boards
- Overview of design

11. Definition
- Lab environment
  - Linux desktop to run Vitis
  - HW/SW emulation
  - HW: Zynq MP (Zybo-27-20) or Zynq Ultrascale+ MPSoC (Ultra96v2)
- SW environment
  - Vitis unified SW platform
- Vitis component
  - Xilinx embedded system hardware
  - Vitis target platform
  - Vitis drivers and runtime (XRT)
  - Vitis core development kit: compilers, analyzers, debuggers
  - Vitis accelerated libraries: OpenCV, Blas, AI/ML, Fintech

12. Design flow
- Program Structure
  - General form
    - Top C-main program (host program)
    - HLS-C Kernel
    - OpenCL API among them
- Execution environments
  - Software emulation
    - QEMU emulates
    - Can check the functionality of applications
  - Hardware emulation
    - QEMU emulates
    - Can check the cycle accuracy of the generated HDL code. Can evaluate the hardware efficiency and analyze the memory transactions to find possible bottlenecks
  - Actual hardware

13. Exercises

## Section 4: Hardware/software setup

14. Introduction
  - VirtualBox for Ubuntu

15. Setup Structure

16. VirtualBox

17. Xilinx Vitis
- Download Vitis unified software platform 2020.2

18. ZCU102 Board -- Vitis platform
- Zynq FPGA Boards
  - Zync SoC
    - Zybo-Z7-20
    - Zynq-7000 SoC ZC702
  - Zynq Ultrascale+ MPSoC
    - Ultra96V2
    - ZCU102
- Need to download the driver and image file
- Xilinx.com->SW development-> Download -> Vitis Embedded Platforms Archive -> 2020.2
- Download  ZCU102 Base 2020.2 (ZIP - 23.68 MB)and  ZYNQMP common image (TAR/GZIP - 1.26 GB) from xilinx.com
  - For 2020.2 or compatible one with Vitis suite
  - Unpack at certain location like ~/hw/vitis/vitis-platform/zcu102
  ```bash
  cd xilinx-zynqmp-common-v2020.2
  mkdir linux_files
  ./sdk.sh
  Enter target directory for SDK (default: /opt/petalinux/2020.2): /.../vitis/vitis-platforms/zcu102/xilinx-zynqmp-common-v2020.2/linux_files
  export PLATFORM_REPO_PATHS=~/hw/vitis/vitis-platform
  ```
  - Restart vitis if necessary
- From GUI, select platform and choose zcu102, creating vector_addition application then in Domain tab, set:
  - Sysroot path: /.../hw/vitis/vitis-platforms/zcu102/xilinx-zynqmp-common-v2020.2/linux_files/sysroots/aarch64-xilinx-linux
  - Root FS: /.../hw/vitis/vitis-platforms/zcu102/xilinx-zynqmp-common-v2020.2/rootfs.ext4
  - Kernel image: /.../hw/vitis/vitis-platforms/zcu102/xilinx-zynqmp-common-v2020.2/Image
- In Templates tab, select Vector Addtition then click Finish

19. Ultra96v2 Board -- Vitis Platform
- Requires Zynq Ultrascale+ MPSoC, AES-ACC-U96-JTAG, 64GB SD disk
- https://avnet.me//Zedsupport

20. ZyboZ7-20Board -- Vitis platform

## Section 5: Vitis-DesignFlow

21. Introduction
- Objectives
  - Create a Vitis project
  - Compile the projects for SW and HW emulation
  - Run the applications in the SW/HW emulators and on the actual FPGA based embedded system

22. Definition
- Hardware/software partitioning
- Offloads computing intesive part into FPGA
- Design Flow
  - Kernel synthesis: Vitis-HLS (v++)
  - Hardware integration: Vivado (v++ linkage)
  - Host program compilation(gcc)
  - Can run on SW/HW emulation or actual FPGA

23. Vitis Project
- vector_addition_kernels->src->vector_addition_kernel.cpp
```cpp
//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Demonstrate Vector Add Kernel
//
#define BUFFER_SIZE 256
#define DATA_SIZE 4096 
//TRIPCOUNT identifier
const unsigned int c_len = DATA_SIZE / BUFFER_SIZE;
const unsigned int c_size = BUFFER_SIZE;
/*
    Vector Addition Kernel Implementation 
    Arguments:
        in1   (input)     --> Input Vector1
        in2   (input)     --> Input Vector2
        out_r   (output)    --> Output Vector
        size  (input)     --> Size of Vector in Integer
*/
extern "C" {
void krnl_vadd(const unsigned int *in1, // Read-Only Vector 1
          const unsigned int *in2, // Read-Only Vector 2
          unsigned int *out_r,     // Output Result
          int size                 // Size in integer
) {
    unsigned int v1_buffer[BUFFER_SIZE];   // Local memory to store vector1
    //Per iteration of this loop perform BUFFER_SIZE vector addition
    for (int i = 0; i < size; i += BUFFER_SIZE) {
       #pragma HLS LOOP_TRIPCOUNT min=c_len max=c_len
        int chunk_size = BUFFER_SIZE;
        //boundary checks
        if ((i + BUFFER_SIZE) > size)
            chunk_size = size - i;
        read1: for (int j = 0; j < chunk_size; j++) {
           #pragma HLS LOOP_TRIPCOUNT min=c_size max=c_size
            v1_buffer[j] = in1[i + j];
        }
        //Burst reading B and calculating C and Burst writing 
        // to  Global memory
        vadd_writeC: for (int j = 0; j < chunk_size; j++) {
           #pragma HLS LOOP_TRIPCOUNT min=c_size max=c_size
            //perform vector addition
            out_r[i+j] = v1_buffer[j] + in2[i+j];
        }
    }
}
}
```
- vector_addition->src->host.cpp
```cpp
#include <stdlib.h>
#include <fstream>
#include <iostream>
//#include "vadd.h"
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>
static const int DATA_SIZE = 4096;
static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";
int main(int argc, char* argv[]) {
    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}
    char* xclbinFilename = argv[1];    
    // Compute the size of array in bytes
    size_t size_in_bytes = DATA_SIZE * sizeof(int);    
    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary    
    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;
    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
	    if (devices.size()){
		    device = devices[0];
		    found_device = true;
		    break;
	    }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device " 
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE; 
    }
    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    // Load xclbin 
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);    
    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);
    // This call will get the kernel object from program. A kernel is an 
    // OpenCL function that is executed on the FPGA. 
    cl::Kernel krnl_vector_add(program,"krnl_vadd");    
    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device. 
    cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, size_in_bytes);
    cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, size_in_bytes);
    cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY, size_in_bytes);
    //set the kernel Arguments
    int narg=0;
    krnl_vector_add.setArg(narg++,buffer_a);
    krnl_vector_add.setArg(narg++,buffer_b);
    krnl_vector_add.setArg(narg++,buffer_result);
    krnl_vector_add.setArg(narg++,DATA_SIZE);
    //We then need to map our OpenCL buffers to get the pointers
    int *ptr_a = (int *) q.enqueueMapBuffer (buffer_a , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes);
    int *ptr_b = (int *) q.enqueueMapBuffer (buffer_b , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes);
    int *ptr_result = (int *) q.enqueueMapBuffer (buffer_result , CL_TRUE , CL_MAP_READ , 0, size_in_bytes);
    //setting input data
    for(int i = 0 ; i< DATA_SIZE; i++){
	    ptr_a[i] = 10;
	    ptr_b[i] = 20;
    }
    // Data will be migrated to kernel space
    q.enqueueMigrateMemObjects({buffer_a,buffer_b},0/* 0 means from host*/);
    //Launch the Kernel
    q.enqueueTask(krnl_vector_add);
    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    q.enqueueMigrateMemObjects({buffer_result},CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();
    //Verify the result
    int match = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        int host_result = ptr_a[i] + ptr_b[i];
        if (ptr_result[i] != host_result) {
            printf(error_message.c_str(), i, host_result, ptr_result[i]);
            match = 1;
            break;
        }
    }
    q.enqueueUnmapMemObject(buffer_a , ptr_a);
    q.enqueueUnmapMemObject(buffer_b , ptr_b);
    q.enqueueUnmapMemObject(buffer_result , ptr_result);
    q.finish();
    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl; 
    return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
}
```
- From GUI, Explorer->vector_addition_system_kernels->vector_addition_kernels.prj, **add Hardware Functions**
  - Click Add Hardware Functions icon then it will find the function  name from the kernel code
- Now build project
- Assistant tab->vector_addition_system will show the status of building
  - Took 2m:50sec
  - Double click Assistent->vector_addition_kernels->Emulation-SW->krnl_vadd-> Compile summary then Vitis Analyzer will open

24. Software Emulation
- To ensure functional correctness in the of the host program and kernels
- Takes the c-based kernel code and compiles it with gcc
- Can debug with gdb
- In Project settings, select Target: Software Emulation
- Build project
- GUI->Browser->Project name->Right mousebutton->Run as -> Launch SW emulation

25. Hardware Emulation
- Can check the functional correctness of the RTL code synthesized from C,C++ or OpenCL kernel code
- Can gain the detailed visibility into internal activity of the kernels
- Can get initial performance estimates for the application
- Will be slower than SW emulation
- DDR memory model and memory interface generator (MIG) will not show exact performance but approximate
- In Project settings, select Target: Hardware Emulation
  - If it was built with SW emulation before, make sure to clean the project
- Build project
- GUI->Browser->Project name->Right mousebutton->Run as -> Launch HW emulation
  - Emulation Console is qemu prompt
  - `shutdown -r now`
  - Rerun with the option of Launch Emulator in GUI mode to display waveforms
  - vivado window will open to show waveform 
- PATCH
  - When following error message is found
```
bad lexical cast: source type value could not be interpreted as target
    while executing
"rdi::set_property core_revision 2302122003 {component component_1}"
    invoked from within
"set_property core_revision $Revision $core"
``` 
  - Year-Month-Day-HOUR-MIN as long integer
  - Larger than 32bit integer limit
  - https://support.xilinx.com/s/article/76960?language=en_US

26. Actual FPGA Hardware
- Target as hardware build
- Takes longer than SW/HW emulation build

27. Exercises

## Section 6: Host program

28. Introduction
- Objectives
  - Understanding the program models in Vitis
  - Understanding the OpenCL concepts in the context of HLS
  - Learn how to write a host program in Vitis

29. Programming Model
- Heterogeneous computing using OpenCL
- Host -> OpenCL API -> XRT <- AXI4 <- Kernel
- Memory space
  - Global memory is available to both of Host on CPU and Kernel on FPGA
  - Kernel get access to the global memory through HP & HPC port
- Kernel language
  - C/C++
  - OpenCL C
  - RLT(HDLs)
- Kernel Execution Modes
  - Sequential Mode: Host program runs Kernel code then exits. Runs another kernel code when necessary through sequential order
  - Pipelined Mode: Host program may launch multiple Kernel codes, not waiting for the early one finished
  - Free-Running Mode: Host and Kernel codes run parallel, communicating each other
![kernelExecutionModes](./kernelExecutionModes.png)
- In this course, sequential mode only

30. OpenCL Concepts
- A host cannot control the kernel directly. Instead it should send some commmands to a queue, and it is the OpenCL runtime to read the queue, fetch the command and order the kernel to do something
- Attached to each device, there must be a context that contains all the related programming objects such as the command queue
- After defining and setting up a context, the host can send commands and then the OpenCL runtime will execute those commands

31. Host Structure
- Setting up the environment
  - Target plaform
  - Devices
  - Create Context
  - Define command queue
  - Define Program
- Core commands
  - Setting up the kernels
  - Buffer definitions
  - Data setting
  - Kernel arguments
  - Kernel execution on FPGA
  - Event synchronization
- Post processing
  - FPGA cleanup
  - Deallocate objects

32. Host code
- OpenCL -> OpenCL C API -> OpenCL C++ API
- Required macro and headers
```cpp
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>
```
- Settubg up the environment
  1. Platform and device
  ```cpp
  std::vector<cl::Device> devices;
  cl::Device device;
  std::vector<cl::Platform> platforms;
  bool found_device = false;
  cl::Platform::get(&platforms);
  for(size_t i=0; (i<platforms.size()) & (found_device==false); i++) {
    cl::Platform platform = platforms[i];
    std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
    if (devices.size()) {
      device = devices[0];
      found_device = true;
      break;
    }
  }
  if (found_device == false) {
    std::cout << "Error: unable to find target device " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    return EXIT_FAILURE;
  }
  ```
  2. Context: contains command queue, memory objects, kernel program object
  ```cpp
  cl::Context context(device);
  ```
  3. Command queues
  ```cpp
  cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
  ```
  4. Program
  ```cpp
  // Load xclbin
  std::cout << "Loading: '" << xclbinFilename << "'\n";
  std::ifstream bin_file(xclbinFilename, std::ifstream;binary);
  bin_file.seekg(0, bin_file.end);
  unsigned nb = bin_file.tellg();
  bin_file.seekg(0,bin_file.beg);
  char *buf = new char [nb];
  bin_file.read(buf, nb);
  // Creating program from binary file
  cl::Program::Binaries bins;
  bins.push_back({buf.nb});
  devices.resize(1);
  cl::Program program(context,devices,bins);
  ```
- Executing commands in the FPGA
  1. Setting up the kernels
  ```cpp
  cl::Kernel krnl_vector_add(program,"krnl_vadd");
  ```
  2. Buffer transfer to/from the FPGA
  ```cpp
  cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, size_in_bytes);
  cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, size_in_bytes);
  cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY, size_in_bytes);
  ```
    - A single buffer cannot be larger than 4GB, yet to maximize throughput from the host to global memory. Xilinx also recommends to keep the buffer size at least 2MB if available (UG1393 v2020.2 March 22, 2021)
  3. Data setting
  ```cpp
  int *ptr_a = (int*) q.enqueueMapBuffer(buffer_a, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes);
  int *ptr_b = (int*) q.enqueueMapBuffer(buffer_b, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes);
  int *ptr_result = (int*) q.enqueueMapBuffer(buffer_result, CL_TRUE, CL_MAP_READ, 0, size_in_bytes);
  ```
  4. Kernel Arguments
  ```cpp
  int narg=0;
  krnl_vector_add.setArg(narg++,buffer_a);
  krnl_vector_add.setArg(narg++,buffer_b);
  krnl_vector_add.setArg(narg++,buffer_result);
  krnl_vector_add.setArg(narg++,DATASIZE);
  ```
  5. Kernel execution on FPGA
    - Migrating input data from host to kernel
    - Invoking the kernel
    - Migrating the output data from kernel to host
  ```cpp
  q.enqueueMigrateMemObjects({buffer_a,buffer_b},0 /* 0 means from host */);
  q.enqueueTask(krnl_vector_add);
  q.enqueueMigrateMemObjects({buffer_result},CL_MIGRATE_MEM_OBJECT_HOST);
  ```
  6. Even synchronization
    - All OpenCL enqueue-based API calls are asynchronous
  ```cpp
  q.finish();
  ```
- Post processing
  1. FPGA cleanup
  ```cpp
  q.enqueueUnmapMemObject(buffer_a, ptr_a);
  q.enqueueUnmapMemObject(buffer_b, ptr_b);
  q.enqueueUnmapMemObject(buffer_result, ptr_result);
  q.finish();
  ```
- Quiz:
  - For a given kernel fuction, define buffer objects
  ```cpp
  extern "C" {
    void spmv_kernel(
      float        *values,
      unsigned int *col_indices,
      unsigned int *row_indices,
      float        *x,
      float        *y,
      unsigned      n,
      unsigned      m,
      unsigned      nnz)
  }
  ```
  - input to kernel
    - values: size of nnz
    - col_indices: size of m
    - row_ptr : size of n
    - x: size of m
  - output from kernel
    - y: size of n
  - Buffer code
  ```cpp
  OCL_CHECK(err,buffer_values     =cl::Buffer(context,CL_MEM_READ_ONLY, nnz*sizeof(float),     nullptr,&err));
  OCL_CHECK(err,buffer_col_indices=cl::Buffer(context,CL_MEM_READ_ONLY, m*sizeof(unsigned_int),nullptr,&err));
  OCL_CHECK(err,buffer_row_indices=cl::Buffer(context,CL_MEM_READ_ONLY, n*sizeof(unsigned_int),nullptr,&err));
  OCL_CHECK(err,buffer_x          =cl::Buffer(context,CL_MEM_READ_ONLY, m*sizeof(float),       nullptr,&err));
  OCL_CHECK(err,buffer_y          =cl::Buffer(context,CL_MEM_WRITE_ONLY,n*sizeof(float),       nullptr,&err));
  ```

33. Exercises

## Section 7: Scaling Example

34. Introduction
- Objectives
  - Studying the memory access pattern of a kernel
  - Debugging an application on emulators and the actual hardware
  - For a vector X, we calculate Y = alpha*X + beta
  
35. Definition
- Scaling equation: y = alpha*x + beta
- 3 inputs of alpha, beta, x
  - How to handle array or pointer data?
- 1 output of y
- Memory latency
  - ~around 100 cycles
  - After addres, when data come
- The processor system sends the scalar values to the accelerator, so the processor is the master and the accelerator is the slave
- During arrays data transaction, the accelerator is the master and the memory is the slave. Note that the memory controllers are located in the PS system such that th ePS acts as a bridge b/w PL and memory

36. Kernel Execution Model
![loopExecution](./loopExecution.png)
- Sequential mode is slow
- Parallel mode is not feasible
![pipelined](./pipelined.png)
- Pipelined mode
  - Initial Interval is the latency
  ![pipelined2](./pipelined2.png)
  - We have to make pipelined as much as possible

37. Kernel code
- Vitis-HLS applies the pipeline optimization by default to all possible for-loops in a kernel code
- Vitis-HLS uses a single HP port for communication b/w kernel and memory subsystem
- Use `extern "C"` for the mix of C/C++ code
```cpp
extern "C" {
  void scaling_kernel(
    float *x, float*y,
    float alpha, float beta,
    int n) {
      for (int i=0;i<n;i++) {
        y[i] = alpha*x[i] + beta;
      }
    }
}
```

38. Burst Data Transfer
- Kernel timing b/w memory read and write in one cycle
  - How can this be done?
- Latency in regular memory read/write
![mem1](./mem1.png)  
- Latency in Burst transfer
![mem2](./mem2.png)  
  - Data received by streaming along pipeline
- Common memory overhead through AXI communication
![mem3](./mem3.png)  
- AXI Burst protocol
![mem4](./mem4.png)  
  - Streaming is the basic idea in Burst data transfer
- The AXI protocol provides burst data transfer for reading/writing a contiguous block of data
- The burst data transfer provides a streaming data communication b/w kernel in the FPGA and DDR memory

39. Host code
- Host code structure
  - Setting up the environment
    - Target platform
    - Devices
    - Context
    - Command queue
    - Program
  - Core commands
    - Setting up the kernels
    - Buffer definitions
    - Data setting
    - Kernel Arguments
    - Kernel execution on FPGA
    - Event Synchronization
  - Post processing
    - FPGA cleanup
- Kernel & Buffers
```cpp
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>
#include <sys/time.h>
#include <time.h>
double getTimestamp()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + tv.tv_sec*1e6;
}
double hardware_start;
double hardware_end;
double hardware_time;
int main(int argc, char* argv[]) {
	unsigned int n =   (1024*1024);
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}
    char* xclbinFilename = argv[1];
    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;
    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
	    if (devices.size()){
		    device = devices[0];
		    found_device = true;
		    break;
	    }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device "
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE;
    }
    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);
    // 1. Kernel & Buffers
    cl::Kernel krnl_scaling(program,"scaling_kernel");
    cl::Buffer buffer_x(context,  CL_MEM_READ_ONLY,  n*sizeof(float));
    cl::Buffer buffer_y(context,  CL_MEM_WRITE_ONLY, n*sizeof(float));
    // 2. Kernel arguments
    float alpha = 1.3;
    float beta = 47.83;
    //set the kernel Arguments
    int narg=0;
    krnl_scaling.setArg(narg++,buffer_x);
    krnl_scaling.setArg(narg++,buffer_y);
    krnl_scaling.setArg(narg++,alpha);
    krnl_scaling.setArg(narg++,beta);
    krnl_scaling.setArg(narg++,n);
    // 3. Input Data Setting
    float *ptr_x = (float *) queue.enqueueMapBuffer (buffer_x , CL_TRUE , CL_MAP_WRITE , 0, n*sizeof(float));
    float *ptr_y = (float *) queue.enqueueMapBuffer (buffer_y , CL_TRUE , CL_MAP_READ , 0, n*sizeof(float));
    for (unsigned int i = 0; i < n; i++) {
    	ptr_x[i] = rand()/(1.0*RAND_MAX);
    }
    hardware_start = getTimestamp();
    // 4. Kernel execution
    queue.enqueueMigrateMemObjects({buffer_x},0/* 0 means from host*/);
    queue.enqueueTask(krnl_scaling);
    queue.enqueueMigrateMemObjects({buffer_y},CL_MIGRATE_MEM_OBJECT_HOST);
    queue.finish();
    hardware_end = getTimestamp();
    hardware_time = (hardware_end-hardware_start)/1000;
	std::cout << "Exeution time running kernel in hardware 1: "
        		    << hardware_time << " msec " << std::endl;
    // 5. Evaluation
    //Verify the result
    int match = 0;
    for (unsigned int i = 0; i < n; i++) {
    	float y_sw = alpha*ptr_x[i]+beta;
		float diff = fabs(y_sw-ptr_y[i]);
		if(diff > 0.0001 || diff != diff){
			std::cout << "error occurs at " << i
					  << " with value y_hw = " << ptr_y[i]
					  << ", should be y_sw = " << y_sw
  					  << std::endl;
            match = 1;
            break;
        }
    }
    queue.enqueueUnmapMemObject(buffer_x , ptr_x);
    queue.enqueueUnmapMemObject(buffer_y , ptr_y);
    queue.finish();
    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
}
```
- During the OpenCL buffere memory, pay attention to the direction of the dataflow and choose a proper argument describing the direction
- After sending the kernel execution command, the host code must wait for the results

40. Lab: Executing
- Design Flow
  - Create a project
  - Add kernel code
  - Add host code
  - Introduce the kernel to the project
  - SW emulation
  - HW emulation
  - Build HW configuration
- At GUI
  - A new project
  - Add src code for kernel/host
  - At  *_kernels.prj, add HW functions
  - Build project as SW emulation
  - Took > 90sec for SW emulation
- Image file is locatd at vector_addition_system/Emulation-SW/package/sd_card.img
  - This can be mounted in Linux system

41. Lab: Debugging
- Emulation SW debugging
  - At GUI, right mouse button of the project -> Debug As -> Launch SW emulator
- Emulation HW debugging
  - You can see the timing diagram and data exchange b/w host and kernel codes
- Hardware debugging
- Ref: ug1393-vitis-application-acceleration document
  - https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration

42. Exercises

## Section 8: Image Thresholding example

43. Introduction
- Objectives
  - Reads an image file
  - Studying the burst data transaction in a kernel
  - Working with data files in Vitis
  - How to use the external SW library in HLS

44. Definition
- Image thresholding
  - output_pixel = maxval if input_pixel > threshold or 0 otherwise
  - Grey image to black/white
- Kernel pseudo-code
  - input_image/output_image through AXI Master    
  - AXIlite for threshold and maxVal arguments
- Summary
  - Image thresholding has two pointers (image files) and two scalar arguments (threshold,maxVal)
  - The port interface for the pointer arguments is master AXI, and the port interface for the scalar arguments is AXI-lite

45. Kernel code
```cpp
extern "C" {
  void image_thresholding_kernel(
    unsigned char *input_image,
    unsigned char *output_image,
    unsigned int n,
    unsigned int m,
    unsigned int threshold,
    unsigned int maxVal)
  {
    unsigned char input_pixel;
    unsighed char output_pixel;
    for (unsigned int i=0; i<n*m; i++) {
      input_pixel = input_image[i];
      output_pixel = (input_pixel > threshold) ? maxVal :0;
      output_image[i] = output_pixel;
    }
  }
}
```
- HP0 port will channel the kernel point arguments b/w DDR memory and FPGA
- Kernel timing diagram
![threshold_kernel1](./threshold_kernel1.png)  
- Kernel execution model
![threshold_kernel2](./threshold_kernel2.png)  
- AXI Burst
  - The memory access in the loop
    - Must be a monotonically increasing order of access
    - Must be consecutive in memory - one next to another with no gaps or overlap and in forward order
- Burst analysis example
![threshold_kernel3](./threshold_kernel3.png)  
- Summary
  - A pipeline micro-architecture can implement image thresholding kernel
  - Array indices in burst loop must be monotonically increasing without gap

46. Host code
- Using OpenCV library
```cpp
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <stdlib.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace cv;
int main(int argc, char* argv[])
{
  int status = 0;
  Mat src_image;
  Mat grey_image;
  src_image=imread("data/test_image.jpg");
  if (!src_image.data) {
    cout << "Could not open image" << endl;
    return 0;
  };
  unsigned int DATA_SIZE = src_image.rows * src_image.cols;
  size_t size_in_bytes = DATA_SIZE * sizeof(unsigned char);
  cvtColor(src_image, grey_image, cv::COLOR_BGR2GRAY);
  Mat dst, dst_golden;
  dst = grey_image.clone();
  dst_golden = grey_image.clone();
  unsigned int  threshold_value = 128;
  unsigned int max_binary_value = 255;
  std::cout << " size_in_bytes = '" << size_in_bytes << "'\n";
  if(argc != 2) {
  	std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
    return EXIT_FAILURE;
  }
  char* xclbinFilename = argv[1];
  std::vector<cl::Device> devices;
  cl::Device device;
  std::vector<cl::Platform> platforms;
  bool found_device = false;
  cl::Platform::get(&platforms);
  for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
  	cl::Platform platform = platforms[i];
    std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
    if ( platformName == "Xilinx"){
     	devices.clear();
     	platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
      if (devices.size()){
        device = devices[0];
        found_device = true;
        break;
      }
    }
  }
  if (found_device == false){
    std::cout << "Error: Unable to find Target Device "
             << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    return EXIT_FAILURE;
  }
  cl::Context context(device);
  cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
  std::cout << "Loading: '" << xclbinFilename << "'\n";
  std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
  bin_file.seekg (0, bin_file.end);
  unsigned nb = bin_file.tellg();
  bin_file.seekg (0, bin_file.beg);
  char *buf = new char [nb];
  bin_file.read(buf, nb);
  cl::Program::Binaries bins;
  bins.push_back({buf,nb});
  devices.resize(1);
  cl::Program program(context, devices, bins);
  cl::Kernel krnl_image_thresholding(program,"image_thresholding_kernel");
  cl::Buffer buffer_in(context,  CL_MEM_READ_ONLY, size_in_bytes);
  cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY, size_in_bytes);
  int narg=0;
  krnl_image_thresholding.setArg(narg++, buffer_in);
  krnl_image_thresholding.setArg(narg++, buffer_out);
  krnl_image_thresholding.setArg(narg++, grey_image.cols);
  krnl_image_thresholding.setArg(narg++, grey_image.rows);
  krnl_image_thresholding.setArg(narg++, threshold_value);
  krnl_image_thresholding.setArg(narg++, max_binary_value);
  unsigned char *ptr_in = (unsigned char  *) q.enqueueMapBuffer (buffer_in ,  CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes);
  unsigned char *ptr_out = (unsigned char *) q.enqueueMapBuffer (buffer_out , CL_TRUE , CL_MAP_READ  , 0, size_in_bytes);
  for (unsigned int i = 0; i< size_in_bytes; i++) {
  	ptr_in[i] = grey_image.data[i];
  }
  dst.data =  ptr_out;
  q.enqueueMigrateMemObjects({buffer_in},0/* 0 means from host*/);
  q.enqueueTask(krnl_image_thresholding);
  q.enqueueMigrateMemObjects({buffer_out},CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  imwrite("grey_threshold.jpg", dst);
  threshold( grey_image, dst_golden, threshold_value, max_binary_value, THRESH_BINARY );
  imwrite("grey_threshold-golden.jpg", dst_golden);
  for (int i = 0; i < grey_image.rows*grey_image.cols; i++) {
	if (dst.data[i] != dst_golden.data[i]) {
		std::cout << " Error at " << i
				  << " hardware dst.data = " << dst.data
               	  << " dst_golden = " << dst_golden.data
                   << std::endl;
		status = -1;
        break;
	}
  }
  cout << "Bye thresholding image" << endl;
  return status;
}
```

47. Emulation - lab
- C/C++Build -> Settings -> Tool Settings -> GCC Host Linker (Arm) -> Libraries, Add opencv_imgcodecs, opencv_highgui, opencv_core, opencv_imgproc

48. Hardware - lab

49. Exercises

## Section 9: Linear Relationship Accelerator Example

50. Introduction
- Objectives
  - Describing a kernel code that requires parallel DDR memory accesses
  - Handling the error codes return by OpenCL APIs

51. Definition
- C = alpha * A + beta * B + gamma
  - A, B, C: n-vectors
  - alpha, beta, gamma: scalar
- Kernel Execution Model
![linear_memport0](./linear_memport0.png) 
  - First execution model: when there are two memory ports. A[i] and B[i] can be read in parallel
  - Second execution model: when there is one memory port only. Computation is pipelined
- When there are two ports
  ![linear_memport1](./linear_memport1.png) 
  - No. cycles = (n-1)*II + l1 = n + l1 - 1
- When there is one ports
  ![linear_memport2](./linear_memport2.png) 
  - No. cycles = (n-1)*II + l2 = 2*n + l2 - 2
  - Note that No. cycles is 2x of the case having two ports
- Summary
  - Providing enough memory ports can improve the performance of an application on FPGA
  - The burst data transfer protocol must support a pipelined implementation of for-loop for providing maximum performance

52. Kernel code
```cpp
extern "C" {
  void linear_relationship_kernel(
    float *A, float *B, float *C, 
    float alpha, float beta, float gamma,
    unsigned int n) {
      for (unsigned int i=0;i<n;i++) {
        C[i] = alpha*A[i] + beta*B[i] + gamma;
      }
  }
}
```
- Synthesis report will show:
  - Interval: 2
  - Unable to schedule bus request on port 'gmem' due to limited memory ports
![linear_memport3](./linear_memport3.png) 
- We use HLS Interface Pragma to separate memory port
```cpp
extern "C" {
  void linear_relationship_kernel(
    float *A, float *B, float *C, 
    float alpha, float beta, float gamma,
    unsigned int n) {
#pragma HLS INTERFACE m_axi port=A bundle=gmem_0
#pragma HLS INTERFACE m_axi port=B bundle=gmem_1
      for (unsigned int i=0;i<n;i++) {
        C[i] = alpha*A[i] + beta*B[i] + gamma;
      }
  }
}
```
- Actual mapping of gmem_0/1 into HP0/1 is done at linker
  - `connectivity.sp linear_relationship_kernel_1.m_axi_gmem_0:HP0 --conectivity.sp linear_relationship_kernel_1.m_axi_gmem_1:HP1`
  - 10x performance differences

53. Host
- OpenCL error codes
  - `cl_int err;`
  - More than 70 error codes
- We use OpenCL API Check macro, OCL_CHECK()
- host code:
```cpp
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>
//OCL_CHECK doesn't work if call has templatized function call
#define OCL_CHECK(error,call)                                        \
    call;                                                            \
    if (error != CL_SUCCESS) {                                       \
      std::cout << __FILE__ << ": " << __LINE__ << " Error calling " \
      #call ", error code is: " << error << std::endl;               \
      exit(EXIT_FAILURE);                                            \
    }
#define DATA_SIZE (1024*1024)
int main(int argc, char* argv[]) {
    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}
    char* xclbinFilename = argv[1];
    // Compute the size of array in bytes
    size_t size_in_bytes = DATA_SIZE * sizeof(float);
    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;
    cl_int err;
    OCL_CHECK(err, err = cl::Platform::get(&platforms));
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        OCL_CHECK(err, std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
        if ( platformName == "Xilinx"){
            devices.clear();
            OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
	    if (devices.size()){
		    device = devices[0];
		    found_device = true;
		    break;
	    }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device "
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE;
    }
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, device, cl::QueueProperties::Profiling, &err));
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    cl::Kernel krnl_linear_relationship;
    OCL_CHECK(err, krnl_linear_relationship = cl::Kernel(program,"linear_relationship_kernel", &err));
    cl::Buffer buffer_A;
    cl::Buffer buffer_B;
    cl::Buffer buffer_C;
    OCL_CHECK(err, buffer_A = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes, nullptr, &err));
    OCL_CHECK(err, buffer_B = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes, nullptr, &err));
    OCL_CHECK(err, buffer_C = cl::Buffer(context, CL_MEM_WRITE_ONLY, size_in_bytes, nullptr, &err));
    float alpha = 1.34;
    float beta  = 2.45;
    float gamma = 3.45;
    //set the kernel Arguments
    int narg=0;
    OCL_CHECK(err, err = krnl_linear_relationship.setArg(narg++,buffer_A));
    OCL_CHECK(err, err = krnl_linear_relationship.setArg(narg++,buffer_B));
    OCL_CHECK(err, err = krnl_linear_relationship.setArg(narg++,buffer_C));
    OCL_CHECK(err, err = krnl_linear_relationship.setArg(narg++,alpha));
    OCL_CHECK(err, err = krnl_linear_relationship.setArg(narg++,beta));
    OCL_CHECK(err, err = krnl_linear_relationship.setArg(narg++,gamma));
    OCL_CHECK(err, err = krnl_linear_relationship.setArg(narg++,DATA_SIZE));
    //We then need to map our OpenCL buffers to get the pointers
    float *ptr_A = (float *) queue.enqueueMapBuffer (buffer_A , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes);
    float *ptr_B = (float *) queue.enqueueMapBuffer (buffer_B , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes);
    float *ptr_C = (float *) queue.enqueueMapBuffer (buffer_C , CL_TRUE , CL_MAP_READ , 0, size_in_bytes);
    //setting input data
    for(int i = 0 ; i< DATA_SIZE; i++){
	    ptr_A[i] = rand()/(1.0*RAND_MAX);
	    ptr_B[i] = rand()/(1.0*RAND_MAX);
    }
    OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_A,buffer_B},0/* 0 means from host*/));
    OCL_CHECK(err, err = queue.enqueueTask(krnl_linear_relationship));
    OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_C},CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = queue.finish());
    //Verify the result
    int match = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        float host_result = alpha*ptr_A[i] + beta*ptr_B[i]+gamma;
        double diff = fabs(ptr_C[i]-host_result);
        if(diff > 0.0001 || diff != diff){
            std::cout << "Error at " << i
            		  << " C hardware is " << ptr_C[i]
					  << " but C golden is " << host_result
					  << std::endl;
            match = 1;
            break;
        }
    }
    OCL_CHECK(err, err = queue.enqueueUnmapMemObject(buffer_A , ptr_A));
    OCL_CHECK(err, err = queue.enqueueUnmapMemObject(buffer_B , ptr_B));
    OCL_CHECK(err, err = queue.enqueueUnmapMemObject(buffer_C , ptr_C));
    OCL_CHECK(err, err = queue.finish());
    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
}
```

54. Lab - Emulator

55. Lab - Hardware

56. Exercises

## Section 10: Loop Optimisation

57. Introduction

58. Definition
- Loop optimization
  - Unrolling
  - Pipelining
  - Flattening
  - Merging
![loop_optimization](./loop_optimization.png)
- Memory optimization
  - BRAM (on FPGA)
  - System Memory (DDR)

59. Loop latency
- How to measure loop performance?
- Latency optimization
  - `#pragma HLS LATENCY max=<> min=<>`
  - If Vitis HLS cannot meet the maximum latency then it relaxes the constraint, trying to achieve the best possible result
  - If the minimum latency constraint is set and Vitis HLS produces a lower latency then it padds dummy clock to meet the minimum latency
- Iteration latency is the number of clock cycles required to execute one loop iteration
- Loop latency is the number of clock cycles required to finish a loop execution
![loop_latency](./loop_latency.png)

60. Loop Unrolling
- When data is on BRAM
  - BRAM has 2 ports only
  - Can be unrolled 2x only (?)
    - Can be improved by memory partitioning (?)
- UNROLL directive
  - `#pragma HLS UNROLL factor=<>`

61. Array Paritioning
- Partitioning arrays
  - `#pragma HLS_ARRAY_PARTITION variable=<> dim=<> factor=<> <type>`
  - type = Block, Cycle, Complete
![block_partitioning](./block_partitioning.png)
![cyclic_partitioning](./cyclic_partitioning.png)
![complete_partitioning](./complete_partitioning.png)
- The cyclic memory partitioning is in line with partial unrolling while complete memory partitioning is in line with the complete memory unrolling
- Multi-diemsional arrays
  - Use `dim` to specify which dimension would be partitioned
  ![multi_partitioning](./multi_partitioning.png)

62. Loop Merge
![loop_merge](./loop_merge.png)
- Loop Merge pragma
  - `#pragma HLS loop_merge`
```cpp
  #pragma HLS loop_merge
  int i;
  loop_1: for(...) { 
  ... 
  }
```
  - Will merge all for loops
```cpp
  int i;
  loop_1: for(...) { 
  #pragma HLS loop_merge
  ... 
  }
```
  - Will merge sub for loops inside of the outer loops
- Loop merging restrictions
  - If loop bounds are all variables, they must have the same value
  - If loop bounds are constants, the maximum constant value is used as the bound of the merged loop
  - Loops with both variable bound and constant bound cannot be merged
  - The code b/w loops to be merged cannot have side effects: multiple execution of this code must generate the same results
    - a = b is allowed
    - a = a + 1 is not allowed

63. Loop Pipeline
- Concurrent operation
- Function pipeline is not discussed in this class
- Latency can be reduced to Initial Interval (II)
![loop_pipelining](./loop_pipelining.png)
```cpp
for (int i=0; i<n ;i++) {
#pragma HLS PIPELINE
  z[i] = x[i] * y[i];
}
```
  - Can assign II (`II=2`) or disable using `OFF`
  - Inner loops in a pipelined loop are unrolled automatically

64. Nested loops: Flattening
- Pipelining a loop causes any loops nested inside the pipelined loop to get unrolled
- If the inner loop can be fully unrolled, applying the pipeline pragma to the outer loops can improve the performance
- Loop flattening is a technique to apply pipeline optimization technique to nested loops

65. Data Dependencies
- The **loop-carried dependency** is the data dependency among loop iterations. And it is usually the main source of high initiation-interval in a pipelined loop

66. Memory Dependencies
- BRAM provides 2 ports
   - II in pipelining
   - Or partition memory
    - When we partition an array into four parts, we save that into four different BRAMs with their own memory ports. Therefore, if we partition an array into four parts, we will have 8 memory ports
provides 2*4=8 ports.
![memory_dependency](./memory_dependency.png)
- Vitis automatically applies the array partitioning whenever it is possible

67. Exercises

## Section 11: Matrix-Vector Multplication (mxv)

68. Introduction
- Objectives
  - Writing a Vitis Kernel to describe the matrix-vector multiplication
  - Improves the performance by optimizaing the pipelined loops
  - Improves the memory bandwidth by using the burst data protocol to exchange data b/w DDR and FPGA

69. Definition
![mxv_complexity](./mxv_complexity.png)

70. Kernel Access Pattern
- Hint
  - Make a temporary storage
  - To enable burst data transfer, use temporary array using BRAM
- base code:
```cpp
void mxv_kernel (float *A, float *x, float *y, 
                 unsigned int n, unsigned int m) {
  for (int i=0; i<n ;i++) {
    y[i] = 0;
    for (int j=0;j<m;j++) {
      y[i] += A[i*m+j]*x[j];
    }
  }
}
```
  - A: no burst, x: no burst, y: no burst
- modification 1:
```cpp
void mxv_kernel (float *A, float *x, float *y, 
                 unsigned int n, unsigned int m) {
  for (int i=0; i<n ;i++) {
    float y_tmp = 0;
    for (int j=0;j<m;j++) {
      y_tmp += A[i*m+j]*x[j];
    }
    y[i] = y_tmp;
  }
}
```
  - A: no burst, x: no burst, y: Burst
- modification 2:
```cpp
void mxv_kernel (float *A, float *x, float *y, 
                 unsigned int n, unsigned int m) {
  float x_local[M_MAX];
  for (int i=0; i<m;i++) x_local[i] = x[i];
  for (int i=0; i<n ;i++) {
    float y_tmp = 0;
    for (int j=0;j<m;j++) {
      y_tmp += A[i*m+j]*x_local[j];
    }
    y[i] = y_tmp;
  }
}
```
  - A: Burst, x: Burst, y: no burst
  - But matrix A doesn't take the benefit from outer loop burst (?)
- modification 3:
```cpp
void mxv_kernel (float *A, float *x, float *y, 
                 unsigned int n, unsigned int m) {
  float x_local[M_MAX];
  float y_local[M_MAX];
  for (int i=0; i<m;i++) x_local[i] = x[i];
  float y_tmp = 0;
  for (int p=0; p<n*m ;p++) {
    int i = p/m;
    int j = p%m;
    y_tmp += A[i*m+j]*x_local[j];
    if (j == m-1) {
      y_local[i] = y_tmp;
      y_tmp = 0;
    }
  }
  for(int i=0; i<n;i++) y[i] = y_local[i];
}
```
  - A: Burst, x: Burst, y: no burst
  - Flattening for-loop
    - Now loop is pipelined
- Using local memory to hold input data or results temporarily can improve the DDR memory data transaction
- Flattening nested-loops manually can help improve the DDR memory utilization and performance

71. Kernel Loops
- Issue of the modification 2
  - Has large II
  - Solution:
    - We may keep the II constant and utilize parallelism inside the loop
    - Or we resolve the carried dependency on y_tmp
- First approach
![first_approach](./first_approach.png)
```cpp
#define M_MAX 10240
#define U 5
void mxv_kernel (float *A, float *x, float *y, 
                 unsigned int n, unsigned int m) {
  float x_local[M_MAX];
  float y_local[M_MAX];
  for (int i=0; i<m;i++) x_local[i] = x[i];
  float y_tmp = 0;
  for (int p=0; p<n*m/U ;p++) {
    int i = p*U/m;
    int j = p*U%m;
    float y_t = 0;
    for (int k=0;k<U;k++) y_t += A[p*U+k]*x_local[j+k];
    y_tmp += y_t;
    if (j + U > m-1) {
      y_local[i] = y_tmp;
      y_tmp = 0;
    }
  }
  for(int i=0; i<n;i++) y[i] = y_local[i];
}
```
- Second approach
  - Vitis reduce
![second_approach](./second_approach.png)
```cpp
void mxv_kernel (float *A, float *x, float *y, 
                 unsigned int n, unsigned int m) {
  float x_local[M_MAX];
  float y_local[M_MAX];
  for (int i=0; i<m;i++) x_local[i] = x[i];
  double y_total = 0; // due to frequent update, double is recommended
  double y_break = 0;
  for (int p=0; p<n*m ;p++) {
    int i = p/m;
    int j = p%m;
    y_total += A[i*m+j]*x_local[j];
    if (j == m-1) {
      y_local[i] = y_total - y_break;
      y_break    = y_total;
    }
  }
  for(int i=0; i<n;i++) y[i] = y_local[i];
}
```
- Utilising parallel threads inside a loop may cancel the negative impact of the high loop initiation interval
- Reducing the number of updates on the variable that causes a carried loop dependency in a pipeline structure can reduce the loop high initiation interval

72. Lab
- Q: vitis_hls vs. vitis?
  - vitis_hls can be used for 1) Vivado IP for HW design and 2) Vitis kernels for Vitis application acceleration development flow
  - Briefly, it can test kernel code
    - No host code runs at vitis_hls. Instead it runs Test Bench
- Testing Vitis kernel code
  - vitis_hls -> Create project -> Top Function as mxv_kernel -> Next x2 -> Part Selection (select HW) -> Flow Target as Vitis Kernel Flow Target -> Source
  - In the Explore:
    - mxv_kernel_test -> Source -> Add mxv_kernel.cpp
    ```cpp
    extern "C" {
    void mxv_kernel(
        float *A, float *x, float *y,
        unsigned int n, unsigned m) {

      L1: for (int i = 0; i < n; i++) {
        y[i] = 0;
        L2: for (int j = 0; j < m; j++) {
          y[i] += A[i*m+j]*x[j];
        }
      }
    }
    }
    ```
    - Test Bench -> Add maxv_tb.cpp
    ```cpp
    #include <stdlib.h>
    #include <fstream>
    #include <iostream>
    #include <math.h>
    extern "C" {
    void mxv_kernel(
        float *A, float *x, float *y,
        unsigned int n, unsigned m);
    }
    int main() {
      unsigned int n =  512;
      unsigned int m =  1024*5;
      float *A = (float *)malloc(n*m*sizeof(float));
      float *x = (float *)malloc(m*sizeof(float));
      float *y_hw = (float *)malloc(n*sizeof(float));
        float *y_sw = (float *)malloc(n*sizeof(float));
        for (unsigned int i = 0; i < n*m; i++) {
          A[i] = rand()/(1.0*RAND_MAX);
        }
        for (unsigned int i = 0; i < m; i++) {
          x[i] = rand()/(1.0*RAND_MAX);
        }
      for (unsigned int i = 0; i < n; i++) {
        y_sw[i] = 0;
        for (unsigned int j = 0; j < m; j++) {
          y_sw[i] += A[i*m+j]*x[j];
        }
      }
      mxv_kernel(A, x, y_hw, n, m);
        int match = 0;
        for (unsigned int i = 0; i < n; i++) {
        float diff = fabs(y_sw[i]-y_hw[i]);
        if(diff > 0.01 || diff != diff){
          std::cout << "error occurs at " << i
                << " with value y_hw = " << y_hw[i]
                << ", should be y_sw = " << y_sw[i]
                  << std::endl;
                match = 1;
                break;
            }
        }
        std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
        return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
    }
    ```
    - Click Run `C simulation`. When successful run `C synthesis`
    - Now check Performace & Resource Estimates
    ![mxv_naive](./mxv_naive.png)
      - Note that L1 loop is not pipelined
      - L2 is pipelined with II=2
    - Right Top of GUI -> Analysis -> Can find sheduler of each operation/step
    ![naive_L1_scheduler](./naive_L1_scheduler.png)
    - Also check synthesis (Right Top of GUI) -> Console
    ```txt
    WARNING: [HLS 200-885] Unable to schedule bus request on port 'gmem' (mxv_kernel_test/mxv_kernel.cpp:12) due to limited memory ports. Please consider using a memory core with more ports or partitioning the array. Resolution: ...
    ```
  - Testing update 1 kernel code
  ```cpp
  extern "C" {
  void mxv_kernel(
      float *A, float *x, float *y,
      unsigned int n, unsigned m) {

      L1: for (int i = 0; i < n; i++) {
      float y_tmp = 0;
          L2: for (int j = 0; j < m; j++) {
              y_tmp += A[i*m+j]*x[j];
          }
          y[i] = y_tmp;
      }
  }
  }
  ```
  ![mxv_01](./mxv_01.png)
    - Console message: WARNING: [HLS 200-960] Cannot flatten loop 'L1' (mxv_kernel_test/mxv_kernel.cpp:27:18) in function 'mxv_kernel' the outer loop is not a perfect loop because there is nontrivial logic in the loop latch.
  - Testing update 2 kernel code
  ```cpp
  #defien M_MAX 10240
  extern "C" {
  void mxv_kernel(
      float *A, float *x, float *y,
      unsigned int n, unsigned m) {

    float x_local[M_MAX];

      L1: for (int i = 0; i < m; i++) {
          x_local[i] = x[i];
      }

      L2: for (int i = 0; i < n; i++) {
          float y_tmp = 0;
          L3: for (int j = 0; j < m; j++) {
              y_tmp += A[i*m+j]*x_local[j];
          }
          y[i] = y_tmp;
      }
  }
  }
  ```
  ![mxv_02](./mxv_02.png)
    - Console message: WARNING: [HLS 200-960] Cannot flatten loop 'L2' (mxv_kernel_test/mxv_kernel.cpp:52:18) in function 'mxv_kernel' the outer loop is not a perfect loop because there is nontrivial logic before entering the inner loop.
    - Also note that there are 3 burst memory  (2 reads/1 write)
  - Testing update 3 kernel code
  ```cpp
  #define M_MAX 10240
  extern "C" {
  void mxv_kernel(
      float *A, float *x, float *y,
      unsigned int n, unsigned m) {
      float x_local[M_MAX];
      L1: for (int i = 0; i < m; i++) {
          x_local[i] = x[i];
      }
      float y_tmp = 0;
      L2: for (int i = 0; i < n; i++) {
          L3: for (int j = 0; j < m; j++) {
              y_tmp += A[i*m+j]*x_local[j];
              if (j == m-1) {
                  y[i] = y_tmp;
                  y_tmp = 0;
              }
          }
      }
  }
  }
  ```
  ![mxv_03](./mxv_03.png)
    - Console message: WARNING: [HLS 200-880] The II Violation in module 'mxv_kernel' (loop 'L3'): Unable to enforce a carried dependence constraint (II = 1, distance = 1, offset = 0) between 'store' operation ('y_tmp_write_ln80', mxv_kernel_test/mxv_kernel.cpp:80) of variable 'y_tmp', mxv_kernel_test/mxv_kernel.cpp:79 on local variable 'y_tmp' and 'load' operation ('y_tmp_load', mxv_kernel_test/mxv_kernel.cpp:79) on local variable 'y_tmp'.
  - Testing update 4 kernel code
  ```cpp
  #define M_MAX 10240
  extern "C" {
  void mxv_kernel(
      float *A, float *x, float *y,
      unsigned int n, unsigned m) {
      float x_local[M_MAX];
      float y_local[M_MAX];
      L1: for (int i = 0; i < m; i++) {
          x_local[i] = x[i];
      }
      float y_tmp = 0;
      L2: for (int p = 0; p < n*m; p++) {
          int i = p/m;
          int j = p%m;
          y_tmp += A[i*m+j]*x_local[j];
          if (j == m-1) {
              y_local[i] = y_tmp;
              y_tmp = 0;
          }
      }
      L3: for (int i = 0; i < n; i++) {
          y[i] = y_local[i];
          y_tmp = 0;
      }
  }
  }
  ``` 
  ![mxv_04](./mxv_04.png)
    - Console message: WARNING: [HLS 200-880] The II Violation in module 'mxv_kernel' (loop 'L2'): Unable to enforce a carried dependence constraint (II = 1, distance = 1, offset = 0) between 'store' operation ('y_tmp_write_ln110', mxv_kernel_test/mxv_kernel.cpp:110) of variable 'y_tmp', mxv_kernel_test/mxv_kernel.cpp:109 on local variable 'y_tmp' and 'load' operation ('y_tmp_load', mxv_kernel_test/mxv_kernel.cpp:109) on local variable 'y_tmp'.
  - Testing update 5 kernel code
  ```cpp
  #define M_MAX 10240
  #define II    5
  extern "C" {
  void mxv_kernel(
      float *A, float *x, float *y,
      unsigned int n, unsigned m) {
      float x_local[M_MAX];
      float y_local[M_MAX];
      L1: for (int i = 0; i < m; i++) {
          x_local[i] = x[i];
      }
      float y_tmp = 0;
      L2: for (int p = 0; p < (n*m)/II; p++) {
          int i = (p*II)/m;
          int j = (p*II)%m;
          float y_t = 0;
          L3: for (int k = 0; k < II; k++) {
              y_t += A[p*II+k]*x_local[j+k];
          }
          y_tmp += y_t;
          if (j+II >= m-1) {
              y_local[i] = y_tmp;
              y_tmp = 0;
          }
    }
      L4: for (int i = 0; i < n; i++) {
          y[i] = y_local[i];
      }
  }
  }
  ```
  ![mxv_05](./mxv_05.png)
    - Console message: WARNING: [HLS 200-880] The II Violation in module 'mxv_kernel' (loop 'L2'): Unable to enforce a carried dependence constraint (II = 1, distance = 1, offset = 0) between 'store' operation ('y_tmp_write_ln30', mxv_kernel_test/mxv_kernel.cpp:30) of variable 'y_tmp', mxv_kernel_test/mxv_kernel.cpp:29 on local variable 'y_tmp' and 'load' operation ('y_tmp_load', mxv_kernel_test/mxv_kernel.cpp:29) on local variable 'y_tmp'.
  - Testing update 6 kernel code
  ```cpp
  #define M_MAX 10240
  extern "C" {
  void mxv_kernel(
      float *A, float *x, float *y,
      unsigned int n, unsigned m) {
      float x_local[M_MAX];
      float y_local[M_MAX];
      L1: for (int i = 0; i < m; i++) {
          x_local[i] = x[i];
      }
      double y_total = 0;
      double y_break = 0;
      L2: for (int p = 0; p < n*m; p++) {
          int i = p/m;
          int j = p%m;
          y_total += A[i*m+j]*x_local[j];

          if (j == m-1) {
              y_local[i] =  y_total - y_break;
              y_break    =  y_total;
          }
      }
      L3: for (int i = 0; i < n; i++) {
          y[i] = y_local[i];
      }
  }
  }
  ```
  ![mxv_06](./mxv_06.png)
- Kernel code has been investigated and updated. Now use vitis to run host/kernel code
- Host code
```cpp
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sys/time.h>
double getTimestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + tv.tv_sec*1e6;
}
double hardware_start;
double hardware_end;
double hardware_time;
int main(int argc, char* argv[]) {
	unsigned int n =  512;
	unsigned int m =  1024;
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}
    char* xclbinFilename = argv[1];
    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;
    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
	    if (devices.size()){
		    device = devices[0];
		    found_device = true;
		    break;
	    }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device "
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE;
    }
    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl_mxv(program,"mxv_kernel");
    cl::Buffer buffer_A(context,  CL_MEM_READ_ONLY,  n*m*sizeof(float));
    cl::Buffer buffer_x(context,  CL_MEM_READ_ONLY,  m*sizeof(float));
    cl::Buffer buffer_y(context,  CL_MEM_WRITE_ONLY, n*sizeof(float));
    //set the kernel Arguments
    int narg=0;
    krnl_mxv.setArg(narg++,buffer_A);
    krnl_mxv.setArg(narg++,buffer_x);
    krnl_mxv.setArg(narg++,buffer_y);
    krnl_mxv.setArg(narg++,n);
    krnl_mxv.setArg(narg++,m);
    float *ptr_A = (float *) queue.enqueueMapBuffer (buffer_A , CL_TRUE , CL_MAP_WRITE , 0, n*m*sizeof(float));
    float *ptr_x = (float *) queue.enqueueMapBuffer (buffer_x , CL_TRUE , CL_MAP_WRITE , 0, m*sizeof(float));
    float *ptr_y = (float *) queue.enqueueMapBuffer (buffer_y , CL_TRUE , CL_MAP_READ  , 0, n*sizeof(float));
    for (unsigned int i = 0; i < n*m; i++) {
    	ptr_A[i] = rand()/(1.0*RAND_MAX);
    }
    for (unsigned int i = 0; i < m; i++) {
    	ptr_x[i] = rand()/(1.0*RAND_MAX);
    }
    hardware_start = getTimestamp();
    queue.enqueueMigrateMemObjects({buffer_A, buffer_x},0/* 0 means from host*/);
    queue.enqueueTask(krnl_mxv);
    queue.enqueueMigrateMemObjects({buffer_y},CL_MIGRATE_MEM_OBJECT_HOST);
    queue.finish();
    hardware_end = getTimestamp();
    hardware_time = (hardware_end-hardware_start)/1000;
    std::cout << "Exeution time running mxv on FPGA: "
              << hardware_time << " msec " << std::endl;
    //--------------------------
    hardware_start = getTimestamp();
    queue.enqueueMigrateMemObjects({buffer_A, buffer_x},0/* 0 means from host*/);
    queue.finish();
    hardware_end = getTimestamp();
    hardware_time = (hardware_end-hardware_start)/1000;
    std::cout << "Exeution time running enqueueMigrateMemObjects to FPGA: "
              << hardware_time << " msec " << std::endl;
    //---------------------------------------
    hardware_start = getTimestamp();
    queue.enqueueTask(krnl_mxv);
    queue.finish();
    hardware_end = getTimestamp();
    hardware_time = (hardware_end-hardware_start)/1000;
    std::cout << "Exeution time running kernel on FPGA: "
              << hardware_time << " msec " << std::endl;
    //-----------------------------------------
    hardware_start = getTimestamp();
    queue.enqueueMigrateMemObjects({buffer_y},CL_MIGRATE_MEM_OBJECT_HOST);
    queue.finish();
    hardware_end = getTimestamp();
    hardware_time = (hardware_end-hardware_start)/1000;
    std::cout << "Exeution time running enqueueMigrateMemObjects from FPGA: "
              << hardware_time << " msec " << std::endl;
    //Verify the result
    float *y_sw = (float *)malloc(n*sizeof(float));
	for (unsigned int i = 0; i < n; i++) {
		y_sw[i] = 0;
		for (unsigned int j = 0; j < m; j++) {
			y_sw[i] += ptr_A[i*m+j]*ptr_x[j];
		}
	}
    int match = 0;
    for (unsigned int i = 0; i < n; i++) {
		float diff = fabs(y_sw[i]-ptr_y[i]);
		if(diff > 0.01 || diff != diff){
			std::cout << "error occurs at " << i
					  << " with value y_hw = " << ptr_y[i]
					  << ", should be y_sw = " << y_sw[i]
  					  << std::endl;
            match = 1;
            break;
        }
    }
    queue.enqueueUnmapMemObject(buffer_A , ptr_A);
    queue.enqueueUnmapMemObject(buffer_x , ptr_x);
    queue.enqueueUnmapMemObject(buffer_y , ptr_y);
    queue.finish();
    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
}
```
- kernel code
```cpp
#define M_MAX 10240
//------------------------------------------------------
//         06
//------------------------------------------------------
extern "C" {
void mxv_kernel(
    float *A, float *x, float *y,
    unsigned int n, unsigned m) {
    float x_local[M_MAX];
    float y_local[M_MAX];
    L1: for (int i = 0; i < m; i++) {
        x_local[i] = x[i];
    }
    double y_total = 0;
    double y_break = 0;
    L2: for (int p = 0; p < n*m; p++) {
        int i = p/m;
        int j = p%m;
        y_total += A[i*m+j]*x_local[j];

        if (j == m-1) {
            y_local[i] =  y_total - y_break;
            y_break    =  y_total;
        }
    }
    L3: for (int i = 0; i < n; i++) {
        y[i] = y_local[i];
    }
}
}
```
- We can use the Vitis-HLS toolset to study a given kernel performance
- The enqueueMigrateMemObjects commands produce overhead and this must be optimized

73. Exercises

## Section 12: Hardware/Software Partitioning

74. Introduction
- Explain the concept of hw/sw partitioning
- Use profiling tools to perform hw/sw partitioning

75. Definition
- Is the target compute intensive or memory intensive?
  - Roofline analysis
- FPGA is good at compute intensive, not memory intensive
  - But high speed memory may overcome this

76. Gperftools
- CPU profiler from Google
  - Linking: -lprofiler
  - Running: export CPUPROFILE to the result file name
  - Analyzing: google-pprof to analyze
- Call graph results
  - Class name/method name, local elapsed time (percentage), and cumulative elapsed time (percentage)

77. Gperftools lab
- sudo apt install google-perftools libgoogle-perftools-dev graphviz kcachegrind
  - 
- Download https://www.csie.ntu.edu.tw/~cjlin/libsvm/libsvm-3.3.tar.gz
- Add -lprofiler in the Makefile
- make
- Download https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
```bash
$ LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so CPUPROFILE=./svm.prof ./svm-train ./svm-data/a1a ./svm-data/a1a.model
*
optimization finished, #iter = 537
nu = 0.460270
obj = -673.031415, rho = 0.628337
nSV = 754, nBSV = 722
Total nSV = 754
$ google-pprof --pdf ./svm-train ./svm.prof > svm.callgrind.pdf
Using local file ./svm-train.
Using local file ./svm.prof.
$ google-pprof --callgrind ./svm-train ./svm.prof > svm.callgrind2
Using local file ./svm-train.
Using local file ./svm.prof.
$ kcachegrind ./svm.callgrind2
```
- svm-train is the executable
 ![kcachegrind](./kcachegrind.png)

78. Exercises

## Section 13: Project 1: Sparse matrix-vector multiplication (SpMV)

79. Introduction
- Objectives
  - Implement the SpMV operator on FPGA efficiently
  - Compare the performance of several SpMV FPGA implementations

80. Definition

81. Baseline code lab
- Kernel code
```cpp
extern "C" {
void spmv_kernel(
		float *value,
		int   *col_index,
		int   *row_index,
		float *x,
		float *y,
		int n,
		int m) {
	int rowStart = 0, rowEnd = n;
	for (int i = rowStart; i < rowEnd; ++i) {
		float y0 = 0.0;
		for (int j=row_index[i]; j<row_index[i+1]; j++) {
			int k = col_index[j];
			y0 += value[j] * x[k];
		}
		y[i] = y0;
	}
}
}
```
- spmv.h
```cpp
#pragma once
void spm_get_size(
		char         *inputFile,
		unsigned int &row_size,
		unsigned int &col_size,
		unsigned int &data_size);
unsigned int  read_mtx_spm(
		char  *inFilename,
		float *values,
		unsigned int    *colIndices,
		unsigned int    *rowPtr);
//OCL_CHECK doesn't work if call has templatized function call
#define OCL_CHECK(error,call)                                       \
    call;                                                           \
    if (error != CL_SUCCESS) {                                      \
      printf("%s:%d Error calling " #call ", error code is: %d\n",  \
              __FILE__,__LINE__, error);                            \
      exit(EXIT_FAILURE);                                           \
    }
```
- spmv_host.cpp
```cpp
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "spmv.h"
#include <math.h>
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>
#include <sys/time.h>
#include <time.h>
double getTimestamp()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + tv.tv_sec*1e6;
}
double hardware_start;
double hardware_end;
double hardware_time;
double software_start;
double software_end;
double software_time;
int main(int argc, char* argv[]) {
	char          inFilename[1000];
	unsigned int  n;
	unsigned int  m;
	unsigned int  nnz;
	strcpy(inFilename, "data/TSC_OPF_1047.mtx.csr");
	spm_get_size(inFilename, n, m, nnz);
	if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}
	char* xclbinFilename = argv[1];
    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;
    cl_int err;
    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    OCL_CHECK(err, err = cl::Platform::get(&platforms));
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        OCL_CHECK(err, std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
        if ( platformName == "Xilinx"){
            devices.clear();
            OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
	    if (devices.size()){
		    device = devices[0];
		    found_device = true;
		    break;
	    }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device "
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE;
    }
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, device, cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder, &err));
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    cl::Kernel krnl_spmv;
    OCL_CHECK(err, krnl_spmv = cl::Kernel(program,"spmv_kernel", &err));
    cl::Buffer buffer_values;
    cl::Buffer buffer_col_indices;
    cl::Buffer buffer_row_indices;
    cl::Buffer buffer_x;
    cl::Buffer buffer_y;
    OCL_CHECK(err, buffer_values = cl::Buffer(context, CL_MEM_READ_ONLY, nnz*sizeof(float), nullptr, &err));
    OCL_CHECK(err, buffer_col_indices = cl::Buffer(context, CL_MEM_READ_ONLY, nnz*sizeof(unsigned int), nullptr, &err));
    OCL_CHECK(err, buffer_row_indices = cl::Buffer(context, CL_MEM_READ_ONLY, (n+1)*sizeof(unsigned int), nullptr, &err));
    OCL_CHECK(err, buffer_x = cl::Buffer(context, CL_MEM_READ_ONLY, m*sizeof(float), nullptr, &err));
    OCL_CHECK(err, buffer_y = cl::Buffer(context, CL_MEM_WRITE_ONLY, n*sizeof(float), nullptr, &err));
    int narg=0;
    OCL_CHECK(err, err = krnl_spmv.setArg(narg++,buffer_values));
    OCL_CHECK(err, err = krnl_spmv.setArg(narg++,buffer_col_indices));
    OCL_CHECK(err, err = krnl_spmv.setArg(narg++,buffer_row_indices));
    OCL_CHECK(err, err = krnl_spmv.setArg(narg++,buffer_x));
    OCL_CHECK(err, err = krnl_spmv.setArg(narg++,buffer_y));
    OCL_CHECK(err, err = krnl_spmv.setArg(narg++,n));
    OCL_CHECK(err, err = krnl_spmv.setArg(narg++,m));
    float *ptr_values = (float *) queue.enqueueMapBuffer (buffer_values , CL_TRUE , CL_MAP_WRITE , 0, nnz*sizeof(float));
    unsigned int *ptr_col_indices = (unsigned int *) queue.enqueueMapBuffer (buffer_col_indices , CL_TRUE , CL_MAP_WRITE , 0, nnz*sizeof(unsigned int));
    unsigned int *ptr_row_indices = (unsigned int *) queue.enqueueMapBuffer (buffer_row_indices , CL_TRUE , CL_MAP_WRITE , 0, (n+1)*sizeof(unsigned int));
    float *ptr_x = (float *) queue.enqueueMapBuffer (buffer_x , CL_TRUE , CL_MAP_WRITE , 0, m*sizeof(float));
    float *ptr_y = (float *) queue.enqueueMapBuffer (buffer_y , CL_TRUE , CL_MAP_READ , 0, n*sizeof(float));
    //setting input data
    read_mtx_spm( inFilename, ptr_values, ptr_col_indices, ptr_row_indices);
    for (unsigned int i = 0; i < m; i++) {
    	ptr_x[i] = rand()/RAND_MAX;
    }
    hardware_start = getTimestamp();
    OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_values,buffer_col_indices, buffer_row_indices, buffer_x},0/* 0 means from host*/));
    OCL_CHECK(err, err = queue.enqueueTask(krnl_spmv));
    OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_y},CL_MIGRATE_MEM_OBJECT_HOST));
    queue.finish();
    hardware_end = getTimestamp();
    hardware_time = (hardware_end-hardware_start)/1000;
	std::cout << "Exeution time running kernel in hardware: "
        		    << hardware_time << " msec " << std::endl;
    // GOLDEN model
	software_start = getTimestamp();
    unsigned int i=0, rowStart=0, rowEnd=n;
    float *y_sw = (float *)malloc(n*sizeof(float));
    float y0=0.0;
    for (i = rowStart; i < rowEnd; ++i) {
    	y0 = 0.0;
    	for (unsigned int j=ptr_row_indices[i]; j<ptr_row_indices[i+1]; j++) {
        	y0 += ptr_values[j] * ptr_x[ptr_col_indices[j]];
        }
        y_sw[i] = y0;
    }
    software_end = getTimestamp();
    software_time = (software_end-software_start)/1000;
	std::cout << "Exeution time running kernel in software 2: "
        		    << software_time << " msec " << std::endl;
    //Verify the result
    int match = 0;
    for (unsigned int i = 0; i < n; i++) {
		float diff = fabs(y_sw[i]-ptr_y[i]);
		if(diff > 0.0001 || diff != diff){
			std::cout << "error occurs at " << i
					  << " with value y_hw = " << ptr_y[i]
					  << ", should be y_sw = " << y_sw[i]
  					  << std::endl;
            match = 1;
            break;
        }
    }
    queue.enqueueUnmapMemObject(buffer_values , ptr_values);
    queue.enqueueUnmapMemObject(buffer_col_indices , ptr_col_indices);
    queue.enqueueUnmapMemObject(buffer_row_indices , ptr_row_indices);
    queue.enqueueUnmapMemObject(buffer_y , ptr_y);
    queue.finish();
    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
}
```
- read_spm.cpp
```cpp
#include <iostream>
#include <fstream>
#include <sstream>
int spm_get_size(
		char         *inFilename,
		unsigned int &row_size,
		unsigned int &col_size,
		unsigned int &data_size) {
	std::ifstream spm_file(inFilename);
	if(spm_file.fail()){
		std::cout << "Sparse matrix file " << inFilename << " doesn't exist!" << std::endl;
		return -1;
	}
	std::string line;
	std::stringstream tokenizer;
	while (getline (spm_file, line)) {
		if (line[0] != '%') {
			unsigned int r, c, d;
			tokenizer << line;
			tokenizer >> r >>  c >> d;
			row_size = r;
			col_size = c;
			data_size = d;
			std::cout << "\r row_size = " <<  row_size << std::endl;
			std::cout << "\r col_size = " <<  col_size << std::endl;
			std::cout << "\r data_size = " << data_size << std::endl;
			break;
		}
	}
	spm_file.close();
	return 0;
}
unsigned int  read_mtx_spm(
		char  *inFilename,
		float *values,
		unsigned int    *colIndices,
		unsigned int    *rowPtr) {
	std::ifstream spm_file(inFilename);
	if(spm_file.fail()){
		std::cout << "Sparse matrix file " << inFilename << " doesn't exist!" << std::endl;
		return -1;
	}
	std::string line;
	unsigned int n;
	unsigned int m;
	unsigned int nnz;
	while (getline (spm_file, line)) {
		if (line[0] != '%') {
			std::stringstream tokenizer;
			tokenizer << line;
			tokenizer >> n >>  m >> nnz;
			unsigned int line_number = 0;
			while (getline (spm_file, line)) {// read a line from a file
				if ( line_number < nnz) {
					float v;
					unsigned int c;
					std::stringstream tokenizer;
					tokenizer << line;
					tokenizer >> c >> v;
					colIndices[line_number] = c;
					values[line_number] = v;
				} else {
					unsigned int r;
					std::stringstream tokenizer;
					tokenizer << line;
					tokenizer >> r;
					rowPtr[line_number-nnz] = r;
				}
				line_number++;
			}
		}
	}
	spm_file.close();
	return 0;
}
```
- Create a new project from vitis
  - Add all of kernel/host code, adding HW function
  - Build SW emulation
  - When successful, select kernel.prj in the explorer. In the menu of Hardware Functions, selec the name of kernel function then click the icon of Launch Vitis HLS
    - Now we can see the report of kernel synthesis
      - No test bench code required
    - Get spmv_kernel.cpp from the Explorer in Vitis HLS GUI
    - Run C Simulation/C Synthesis
    ![spmv_kernel_00](./spmv_kernel_00.png)
    - Console log:
    ```
    WARNING: [HLS 200-881] Unable to enforce a carried constraint (II = 1) between 'fadd' operation ('y0', ../../../../src/spmv_kernel.cpp:15) and 'fadd' operation ('y0', ../../../../src/spmv_kernel.cpp:15).
    Resolution: For help on HLS 200-881 see www.xilinx.com/cgi-bin/docs/rdoc?v=2020.2;t=hls+guidance;d=200-881.html
    WARNING: [HLS 200-881] Unable to enforce a carried constraint (II = 2) between 'fadd' operation ('y0', ../../../../src/spmv_kernel.cpp:15) and 'fadd' operation ('y0', ../../../../src/spmv_kernel.cpp:15).
    Resolution: For help on HLS 200-881 see www.xilinx.com/cgi-bin/docs/rdoc?v=2020.2;t=hls+guidance;d=200-881.html
    WARNING: [HLS 200-881] Unable to enforce a carried constraint (II = 3) between 'fadd' operation ('y0', ../../../../src/spmv_kernel.cpp:15) and 'fadd' operation ('y0', ../../../../src/spmv_kernel.cpp:15).
    Resolution: For help on HLS 200-881 see www.xilinx.com/cgi-bin/docs/rdoc?v=2020.2;t=hls+guidance;d=200-881.html
    WARNING: [HLS 200-881] Unable to enforce a carried constraint (II = 4) between 'fadd' operation ('y0', ../../../../src/spmv_kernel.cpp:15) and 'fadd' operation ('y0', ../../../../src/spmv_kernel.cpp:15).
    Resolution: For help on HLS 200-881 see www.xilinx.com/cgi-bin/docs/rdoc?v=2020.2;t=hls+guidance;d=200-881.html
    ```

82. HSL code - 01
- Use burst protocal for read/write memory
- Read row_ptr then calculate the differences b/w two adjacent array
- Then array of value/col_index can have same index
- Using the number of elements in each row, we can usea single loop to describe spmv calculation
![spmv_hls_01](./spmv_hls_01.png)

83. HLS code - 01 lab
```cpp
#define DIM 60098
extern "C" {
void spmv_kernel(
		float          *values,
		unsigned int   *col_indices,
		unsigned int   *row_indices,
		float          *x,
		float          *y,
		unsigned int    n,
		unsigned int    m) {
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=values
#pragma HLS INTERFACE m_axi bundle=gmem_1 port=col_indices
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=x
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=y
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=row_indices
	float x_local[DIM];
	float y_local[DIM];
	float row_indices_diff_local[DIM];
	unsigned int nnz = 0;
	for (unsigned int i =0; i < m; i++) {
		x_local[i] = x[i];
	}
	unsigned int previous_row_index;
	for (unsigned int i =0; i < n+1; i++) {
		unsigned int row_index = row_indices[i];
		if (i > 0) {
			row_indices_diff_local[i-1] = row_index-previous_row_index;
			nnz += row_index-previous_row_index;;
		}
		previous_row_index = row_index;
	}
	double y_tmp = 0.0;
	unsigned int j = 0;
	unsigned int remained_row_index = row_indices_diff_local[j++];
	for (int i = 0; i < nnz; ++i) {
		int k = col_indices[i];
		float y_t = values[i] * x_local[k];
		y_tmp += y_t;

		remained_row_index--;
		if (remained_row_index == 0) {
			y_local[j-1] = y_tmp;
			y_tmp = 0;
			remained_row_index = row_indices_diff_local[j++];
		}
	}
	for (unsigned int i =0; i < n; i++) {
		y[i] = y_local[i];
	}
}
}
```
![spmv_hls_01_lab](./spmv_hls_01_lab.png)
- All memory use burst protocol
- Console log: WARNING: [HLS 200-880] The II Violation in module 'spmv_kernel' (loop 'VITIS_LOOP_56_3'): Unable to enforce a carried dependence constraint (II = 1, distance = 1, offset = 0) between 'store' operation ('y_tmp_write_ln62', ../../../../src/spmv_kernel.cpp:62) of variable 'y_tmp', ../../../../src/spmv_kernel.cpp:59 on local variable 'y_tmp' and 'load' operation ('y_tmp_load', ../../../../src/spmv_kernel.cpp:59) on local variable 'y_tmp'.
  - Issues of using y_tmp
  ![y_tmp_schedule](./y_tmp_schedule.png)
- Loop-carried dependency causes high-initiation interval in SpMV code

84. HLS code - 02
![II_violation_1](II_violation_1.png)
![II_violation_2](II_violation_2.png)
- Introducing two local variables
  - There is a loop-carried depency in y_previous_break but this is very short - 1 cycle only
  ![II_solution](II_solution.png)
- The distance b/w read and write operations on a variable that causes loop-carried dependency defines the initiation interval
- There are two potential solutions for solving a loop-carried dependency in a pipelined loop
  - Utilizing parallel thread to compensate for the negative impact of the high-initiation interval on performance
  - Resolving the dependency by introducing new variables and modifying the code

85. HLS code - 02 lab
- Updated kernel code:
```cpp
#define DIM 60098
#include <ap_int.h>
extern "C" {
void spmv_kernel(
		float          *values,
		unsigned int   *col_indices,
		unsigned int   *row_indices,
		float          *x,
		float          *y,
		unsigned int    n,
		unsigned int    m) {
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=values
#pragma HLS INTERFACE m_axi bundle=gmem_1 port=col_indices
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=x
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=y
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=row_indices
	unsigned int rowStart = 0, rowEnd = n;
	float x_local[DIM];
	float y_local[DIM];
	float row_indices_diff_local[DIM];
	unsigned int nnz = 0;
	for (unsigned int i =0; i < m; i++) {
		x_local[i] = x[i];
	}
	unsigned int previous_row_index;
	for (unsigned int i =0; i < n+1; i++) {
		unsigned int row_index = row_indices[i];
		if (i > 0) {
			row_indices_diff_local[i-1] = row_index-previous_row_index;
			nnz += row_index-previous_row_index;;
		}
		previous_row_index = row_index;
	}
	double y_previous_break = 0.0;
	double y_all_row = 0.0;
	unsigned int j = 0;
	unsigned int remained_row_index = row_indices_diff_local[j++];
	for (int i = 0; i < nnz; ++i) {
		int k = col_indices[i];
		float y_t = values[i] * x_local[k];
		y_all_row += y_t;
		remained_row_index--;
		if (remained_row_index == 0) {
			y_local[j-1] = y_all_row - y_previous_break;
			y_previous_break = y_all_row;
			remained_row_index = row_indices_diff_local[j++];
		}
	}
	for (unsigned int i =0; i < n; i++) {
		y[i] = y_local[i];
	}
}
}
```
![spmv_hls_02_lab](./spmv_hls_02_lab.png)

86. Vitis lab

## Section 14: Project 2: Support Vector Mahcine (svm)

87. Definition

88. Software code

89. Code structure
![svc_q](./svc_q.png)

90. HLS code
- Kernel code: 
```cpp
#define DIM 10098
extern "C" {
	void spmv_kernel(
		double* values,
		int* col_indices,
		int* row_indices,
		double* x,
		double* y,
		int start,
		int end,
		int n,
		int m) {
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=values
#pragma HLS INTERFACE m_axi bundle=gmem_1 port=col_indices
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=x
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=y
#pragma HLS INTERFACE m_axi bundle=gmem_0 port=row_indices
		double x_local[DIM];
		double y_local[DIM];
		int row_indices_diff_local[DIM];
		int nnz = 0;
		for (int i = 0; i < m; i++) {
			x_local[i] = x[i];
		}
		int previous_row_index;
		for (int i = 0; i < n + 1; i++) {
			 int row_index = row_indices[i];
			if (i > 0) {
				row_indices_diff_local[i - 1] = row_index - previous_row_index;
				nnz += row_index - previous_row_index;;
			}
			previous_row_index = row_index;
		}
		double y_previous_break = 0.0;
		double y_all_row = 0.0;
		int j = 0;
		int remained_row_index = row_indices_diff_local[j++];
		for (int i = 0; i < nnz; ++i) {
			int k = col_indices[i];
			double y_t = values[i] * x_local[k];
			y_all_row += y_t;
			remained_row_index--;
			if (remained_row_index == 0) {
				y_local[j - 1] = y_all_row - y_previous_break;
				y_previous_break = y_all_row;
				remained_row_index = row_indices_diff_local[j++];
			}
		}
		for (int i = 0; i < n; i++) {
			y[i] = y_local[i];
		}
	}
}
```

91. Lab

## Section 15: Appendix

92. How to create zybo-z7-20 Hardware Vitis 2020.2 Platform
- Vitis platform creation steps
  - Hardware (Vivado): .XSA file
  - Software (PetaLinux)
  - Platform (Vitis)
- Required SW
  - Vitis unified software platform
  - PetaLinux Tools: UG1144 document

93. Vitis 2021.1 Embedded Platform for Zybo-Z7-20

## Xilinx product
- Virtex: for DSP
- Alveo: for HPC

## Xilinx math library
- Vitis HLS Math library
