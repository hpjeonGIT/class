## Shared Libraries in C Programming by John OSullivan
- Ref: https://medium.com/@johnos3747/shared-libraries-in-c-programming-ab149e80be22

### Building shared libraries
- hello.cxx:
```cxx
#include <mpi.h>
#include <cstdio>
int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);
    MPI_Finalize();
}
```
- mpicxx -fPIC hello.cxx

### Analyzing the binary with readelf
- readelf shows the share libraries which the exe depends on
```bash
$ readelf -d a.out |grep NEED
 0x0000000000000001 (NEEDED)             Shared library: [libmpi.so.40]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000006ffffffe (VERNEED)            0x680
 0x000000006fffffff (VERNEEDNUM)         1
$ readelf -d a.out |grep runpath
 0x000000000000001d (RUNPATH)            Library runpath: [/home/hpjeon/sw_local/openmpi/5.0.8/lib]
```
- To find the location, check runpath

### Using pmp to inspect memory
- pmap shows the physical address of any copies of the linked shared library in memory
- While using pmap, the program must be running - add some sleep(100)
```bash
$ ps -ef |grep a.out
hpjeon    421360  343448  2 09:26 pts/0    00:00:00 prterun -n 2 ./a.out
hpjeon    421367  421360  0 09:26 pts/0    00:00:00 ./a.out
hpjeon    421368  421360  0 09:26 pts/0    00:00:00 ./a.out
$ pmap 421367
421367:   ./a.out
0000156718000000    208K rw-s- pmix-gds-shmem2.hakune3-prterun-hakune3-421360@1.session.56e748f4.421360
00002ac928000000    208K rw-s- pmix-gds-shmem2.hakune3-prterun-hakune3-421360@1.jobdata.421360
0000601431042000      4K r---- a.out
0000601431043000      4K r-x-- a.out
0000601431044000      4K r---- a.out
0000601431045000      4K r---- a.out
0000601431046000      4K rw--- a.out
...
00007df8bbe00000    692K r-x-- libcudart.so.12.8.90
00007df8bbead000   2048K ----- libcudart.so.12.8.90
00007df8bc0ad000     16K r---- libcudart.so.12.8.90
00007df8bc0b1000      4K rw--- libcudart.so.12.8.90
...
00007df8bca00000    256K r---- libpmix.so.2.13.8
00007df8bca40000   1576K r-x-- libpmix.so.2.13.8
00007df8bcbca000    328K r---- libpmix.so.2.13.8
00007df8bcc1c000     36K r---- libpmix.so.2.13.8
00007df8bcc25000     76K rw--- libpmix.so.2.13.8
...
```

### Multiple instances of a shared library
- Fromp pmap, we see many instances of the same shared library, loading multiple copies of the library becaus:
  - Different versions
  - Copy-on-write: When a process forks, the child process may copy the memory of the parent process
  - Position independent code (PIC)
- Having multiple instances of a shared library doesn't mean the consumption of more memory

### Using objdump to view virtual addresses
```bash
$ objdump -T ./a.out 
./a.out:     file format elf64-x86-64
DYNAMIC SYMBOL TABLE:
0000000000000000      DF *UND*	0000000000000000 (GLIBC_2.2.5) printf
0000000000000000  w   D  *UND*	0000000000000000  Base        __gmon_start__
0000000000000000      DF *UND*	0000000000000000  Base        MPI_Init
0000000000000000  w   D  *UND*	0000000000000000  Base        _ITM_deregisterTMCloneTable
0000000000000000  w   D  *UND*	0000000000000000  Base        _ITM_registerTMCloneTable
0000000000000000      DF *UND*	0000000000000000  Base        MPI_Get_processor_name
0000000000000000      DF *UND*	0000000000000000  Base        MPI_Comm_size
0000000000000000      DF *UND*	0000000000000000 (GLIBC_2.2.5) sleep
0000000000000000      DF *UND*	0000000000000000  Base        MPI_Comm_rank
0000000000000000      DF *UND*	0000000000000000 (GLIBC_2.4)  __stack_chk_fail
0000000000000000      DF *UND*	0000000000000000  Base        MPI_Finalize
0000000000000000      DO *UND*	0000000000000000  Base        ompi_mpi_comm_world
0000000000000000      DF *UND*	0000000000000000 (GLIBC_2.34) __libc_start_main
0000000000000000  w   DF *UND*	0000000000000000 (GLIBC_2.2.5) __cxa_finalize
```
- Notice that in the main program the virtual address associated with the shared library function is 0000000000000000, implying that the symbols have not been fully resolved at the time of linking.
- Symbols are resolved at runtime

### Using ldd to identify share libraries
- ldd (List Dynamic Dependencies), which is a shell script, displays the shared libraries required by a given executable or shared object, along with their paths and the corresponding memory addresses where they are loaded at runtime
- It reads the dynamic section of the exe, which contains the metadata about the linked libraries
```bash
$ ldd ./a.out 
	linux-vdso.so.1 (0x0000797b3a42e000)
	libmpi.so.40 => /home/hpjeon/sw_local/openmpi/5.0.8/lib/libmpi.so.40 (0x0000797b3a000000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x0000797b39c00000)
	libopen-pal.so.80 => /home/hpjeon/sw_local/openmpi/5.0.8/lib/libopen-pal.so.80 (0x0000797b39eea000)
	libpmix.so.2 => /home/hpjeon/sw_local/openmpi/5.0.8/lib/libpmix.so.2 (0x0000797b39800000)
	libevent_core-2.1.so.7 => /home/hpjeon/sw_local/openmpi/5.0.8/lib/libevent_core-2.1.so.7 (0x0000797b3a3d7000)
	libevent_pthreads-2.1.so.7 => /home/hpjeon/sw_local/openmpi/5.0.8/lib/libevent_pthreads-2.1.so.7 (0x0000797b3a3d0000)
	libhwloc.so.15 => /home/hpjeon/sw_local/openmpi/5.0.8/lib/libhwloc.so.15 (0x0000797b3a372000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x0000797b39b17000)
	/lib64/ld-linux-x86-64.so.2 (0x0000797b3a430000)
```

### Using gdb to find address
```bash
$ gdb ./a.out
(gdb) info address MPI_Init
Symbol "MPI_Init" is at 0x10d0 in a file compiled without debugging.
(gdb) info address MPI_Finalize
Symbol "MPI_Finalize" is at 0x1130 in a file compiled without debugging.
```
