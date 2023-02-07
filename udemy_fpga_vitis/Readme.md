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
- From GUI, Explorer->vector_addition_system_kernels->vector_addition_kernels.prj, add Hardware Functions
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
- Build project
- GUI->Browser->Project name->Right mousebutton->Run as -> Launch HW emulation
  - Emulation Console is qemu prompt
  - `shutdown -r now`
  - Rerun with the option of Launch Emulator in GUI mode to display waveforms
  - vivado window will open to show waveform 

26. Actual FPGA Hardware

27. Exercises


## Xilinx product
- Virtex: for DSP
- Alveo: for HPC

## For Alveo U280 
- Ref: https://xilinx.github.io/Vitis-Tutorials/2020-2/docs/build/html/docs/Getting_Started/Vitis/Part2.html
- Xilinx provides base platforms for the Alveo U200, U250, U50 and U280 data-center acceleration cards. Before installing a platform, you need to download the following packages:
  - Xilinx Runtime (XRT)
  - Deployment Target Platform
  - Development Target Platform
