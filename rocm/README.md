## Updating nvidia at Ubuntu 22.04
- sudo ubuntu-drivers list --gpgpu
  - Check all of installed drivers and remove old drivers using sudo apt purge command
  - dpkg -l |grep nvidia
- sudo ubuntu-drivers install nvidia:560
- sudo install cuda-toolkit
- Reboot
- Check nvidia-smi runs OK
- do not use /usr/bin/nvcc. Find /usr/local/cuda/bin/nvcc
- OS upgrade from 20.04 -> 22.04
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu

## Install rocm/hip
- Ref: https://github.com/ROCm/HIP/issues/3310
- Ref: https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html
- Ref: https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/native-install/ubuntu.html
  - Even though we use nvidia gpu (no amd gpu), we need to install amdgpu-dkms
- sudo apt install amdgpu-dkms
- sudo reboot
- sudo apt install rocm # 31GB
- sudo apt-get install hip-runtime-nvidia hip-dev


## Running hip code on nvidia card
```bash
export HIP_PLATFORM=nvidia
export PATH+=:/opt/rocm/bin
hipify-perl ../cuda_code/helper_cuda.h > helper_hip.h
hipify-perl ../cuda_code/helper_string.h > helper_string.h
hipify-perl ../cuda_code/vectorAdd.cu > vectorAdd.cpp
/opt/rocm/bin/hipcc -g vectorAdd.cpp -I.
$ ./a.out
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
```

## Build cuda using cmake
- split vectorAdd.cu into main.cpp and vectorAdd.cu
- Brute-force build
```
g++ -c main.cpp -L/usr/local/cuda/lib64 -lcuda -lcudart  -I/usr/local/cuda/include -I.
nvcc -c vectorAdd.cu -I.
 nvcc main.o vectorAdd.o -o a.exe # here, nvcc is the linker
```
- Using CMakeLists.txt
```
cmake_cuda
├── CMakeLists.txt
└── src
    ├── CMakeLists.txt
    ├── helper_cuda.h
    ├── helper_string.h
    ├── main.cpp
    └── vectorAdd.cu
```

## Build hip code using cmake
- hipify cude code into hip code
```
hipify-perl ../../cmake_cuda/src/helper_cuda.h > helper_hip.h
hipify-perl ../../cmake_cuda/src/helper_string.h  > helper_string.h
hipify-perl ../../cmake_cuda/src/main.cpp > main.cpp
hipify-perl ../../cmake_cuda/src/vectorAdd.cu > vectorAdd.hip
```
- Brute-force build
```
/opt/rocm/bin/hipcc -x cu -c vectorAdd.hip -I/opt/rocm/include -I.
/opt/rocm/bin/hipcc -c main.cpp -I/opt/rocm/include -I. 
/opt/rocm/bin/hipcc main.o vectorAdd.o -o a.exe
$ ./a.exe 
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
```
  - If AMD card is available, may use amdclang++ -x hip -c vectorAdd.hip -I.
- Using cmake build
```
cmake_hip/
├── CMakeLists.txt
└── src
    ├── CMakeLists.txt
    ├── helper_hip.h
    ├── helper_string.h
    ├── main.cpp
    └── vectorAdd.hip
```
- Not working/not compiled.
- Feeding -DCMAKE_HIP_COMPILER=hipcc yields following error message. Use amdclang++
```
$ cmake -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc ..
CMake Error at /usr/share/cmake-3.22/Modules/CMakeDetermineHIPCompiler.cmake:50 (message):
  CMAKE_HIP_COMPILER is set to the hipcc wrapper:
   /opt/rocm/bin/hipcc
  This is not supported.  Use Clang directly, or let CMake pick a default.
Call Stack (most recent call first):
  CMakeLists.txt:1 (project)
```
- Using cmake 3.30
  - export HIP_ROOT=/opt/rocm
  - cmake ..
- Ref: https://github.com/neoblizz/HIP_template/blob/main/library/CMakeLists.txt
  - Download the entire src
  - Add following into CMakeLists.txt
```
set(ROCM_PATH /opt/rocm)
set(HIP_PATH /opt/rocm)
```

## Tentative conclusion
- cmake + rocm may not work on nvidia GPUs.
  - Coupling hipcc + cmake yields an error
  - cmake + rocm ignores nvidia relatd variables, yielding -x hip option when built
- Regarding cmake_hip, add ROCM_PATH or HIP_PATH in the CMakeLists.txt or declare in the CLI
  - Still not working with HIP over nvidia
  - To compile hip code for nvidia architects:   "--generate-code=arch=compute_50,code=sm_50"
    - Requires `export HIP_PLATFORM=nvidia` prior to running cmake
    - TBD
    
## AMD only with cmake
- ~~Not working on nvidia card but can be compiled~~
  - Now runs with GT1030
- cmake_hip_only
- Works at cmake 3.30 but not at 3.22
```
.
├── CMakeLists.txt
└── src
    ├── CMakeLists.txt
    ├── helper_hip.h
    ├── helper_string.h
    ├── main.cpp
    ├── test.hiph
    └── vectorAdd.hip
```
