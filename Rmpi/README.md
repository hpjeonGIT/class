# Distributed computing of R with Rmpi
- In this note, I leave the introductory/educational contents for distributed computing of R, using Rmpi - pbdMPI can be used instead
- For parallel computing of R, a single node method of doSNOW or doMC might be used to make use of multiple core architects. However, such methods cannot extend over multiple nodes, and distributed computing is required for very large scale
- Also single node parallel computing methods like doSNOW/doMC have large overhead and the computing efficiency may be less than 80%

# Preparation
- To use Rmpi, R-base and a MPI library are required. In this note, we use openmpi as a sampe MPI library
- R/MPI might be installed using system library installer like yum or apt but here we describe the methods of installing from source code package

## Installing Rmpi
- Pre-requisites
  - R packages
  - MPI libraries like openmpi or mvapich
- Install steps
```bash
# R base install
$ sudo apt install gfortran
$ sudo apt install libreadline6-dev
$ sudo apt install libx11-dev libxt-dev libbz2-dev liblzma-dev libpcre2-dev libcurl4-gnutls-dev
$ tar xvf R-4.6.0.tar.xz 
$ cd R-4.6.0/
$ ./configure --prefix=/home/hpjeon/sw_local/R/4.6.0
$ make -j 10
$ make install
# OpenMPI install
$ tar zxf openmpi-5.0.10.tar.gz 
$ cd openmpi-5.0.10/
$ ./configure --prefix=/home/hpjeon/sw_local/openmpi/5.0.10
$ make -j 10
$ make install
# Rmpi install
# load the module of R/OpenMPI or configure PATH/LD_LIBRARY_PATH to find their bin/lib directories
$ R CMD INSTALL Rmpi_0.7-3.4.tar.gz --configure-args="--with-Rmpi-include=/home/hpjeon/sw_local/openmpi/5.0.10/include/ --with-Rmpi-libpath=/home/hpjeon/sw_local/openmpi/5.0.10/lib --with-Rmpi-type=OPENMPI"
```

## Sample environmental module files
- R_4.6.0
```tcl
#%Module1.0
proc ModulesHelp { } {
    puts stderr "\tConfiguring R 4.6.0"
}
module-version 1.2
# Set the installation directory
set TOP /home/hpjeon/sw_local/R/4.6.0
prepend-path    PATH            $TOP/bin
prepend-path    LD_LIBRARY_PATH $TOP/lib
```
- openmpi_5.0.10 
```tcl
#%Module1.0
proc ModulesHelp { } {
    puts stderr "\tConfiguring openmpi 5.0.10"
}
module-version 1.2
# Set the installation directory
set TOP /home/hpjeon/sw_local/openmpi/5.0.10
prepend-path    PATH            $TOP/bin
prepend-path    LD_LIBRARY_PATH $TOP/lib
```

## Running Rmpi
- Only batch mode
- No interactive session is allowed with MPI libraries
- Sample command: `mpirun -n 10 Rscript my_mpi_script.R`

# Introductory topics

## Serial version of Hello world with Rscript
- hello.R:
```r
cat("Hello world\n")
```
- Demo:
```bash
$ Rscript hello.R 
Hello world
```

## Sample Hello world over multiple ranks
- hello_mpi.R:
```r
library(Rmpi) # loading Rmpi package
my_rank <- mpi.comm.rank(comm=0) # set up rank id
n_ranks <- mpi.comm.size(comm=0) # get the size of all ranks
cat("hello world from ", my_rank, "and ", 
    mpi.get.processor.name(), " out of ", n_ranks, "\n")
ierr = mpi.barrier(comm=0) # sync here. returns 1 when successful but 0 when fails
mpi.quit() # equivalent to MPI_Finalize()
quit("yes")
```
- Demo:
```bash
$ module load R_4.6 openmpi_5.0.10
$ mpirun -n 3 Rscript hello_mpi.R 
hello world from  1 and  hakuneMini  out of  3 
hello world from  2 and  hakuneMini  out of  3 
hello world from  0 and  hakuneMini  out of  3 
```

## Sample collective calls
```r
library(Rmpi)
my_rank <- mpi.comm.rank(comm=0)
n_ranks <- mpi.comm.size(comm=0)
## Reduce example:
res <- mpi.reduce(my_rank,type=1,op="sum", dest=0,comm=0)
if (my_rank==0) { cat("At rank0, the sum of all ranks = ", res, "\n") }
## Allreduce example:
res <- mpi.allreduce(my_rank,type=1,op="sum", comm=0)
cat("At rank ", my_rank, " the all reduced value = ", res,'\n');
## Broadcast example
if (my_rank == 0) { 
  my_value <- 100;
} else {
  my_value <- 10000;
}
my_value <- mpi.bcast(my_value, 1, rank=0, comm=0) # unlike regular MPI_Bcast, my_value is not updated in other ranks. The return value must be assigned
Sys.sleep(my_rank)
cat("At rank ", my_rank, " the broadcasted value = ", my_value,'\n');
# 
ierr <- mpi.barrier(comm=0)
mpi.quit()
quit("yes")
```
- Demo
```bash
$ mpirun -n 3 Rscript mpi_collective.R 
At rank0, the sum of all ranks =  3 
At rank  0  the all reduced value =  3 
At rank  2  the all reduced value =  3 
At rank  1  the all reduced value =  3 
At rank  0  the broadcasted value =  100 
At rank  1  the broadcasted value =  100 
At rank  2  the broadcasted value =  100 
```

## Sample send/receive code
```r
library(Rmpi)
my_rank <- mpi.comm.rank(comm=0)
n_ranks <- mpi.comm.size(comm=0)
my_list <- c()
if (my_rank == 0) {
  my_list <- c()
  for (i in 1:(n_ranks-1)) {
    x <- integer(2) # irecv/recv argument is a vector as isend/send can send a vector
    ierr = mpi.recv(x,type=1,source=i, tag=i, comm=0)
    my_list <- c(my_list,x)
  }
  cat("At rank 0, all received data are ", my_list, "\n")
} else {
    x <- c(my_rank + 100, my_rank + 200)
    cat("At rank ", my_rank, " am sending ", x, "\n")
    ierr = mpi.send(x,type=1,dest=0,tag =my_rank, comm=0)
}
ierr <- mpi.barrier(comm=0)
mpi.quit()
quit("yes")
```
- Demo:
```bash
$ mpirun -n 4 Rscript mpi_sendrecv.R 
At rank  2  am sending  102 202 
At rank  3  am sending  103 203 
At rank  1  am sending  101 201 
At rank 0, all received data are  101 201 102 202 103 203 
```

## Async send
```py
library(Rmpi)
my_rank <- mpi.comm.rank(comm=0)
n_ranks <- mpi.comm.size(comm=0)
my_list <- c()
if (my_rank == 0) {
  ncount <- 0
  for (i in 1:(n_ranks-1)) {
    x <- integer(2) # irecv/recv argument is a vector as isend/send can send a vector
    ierr = mpi.recv(x,type=1,source=i, tag=i, comm=0)
    my_list <- c(my_list,x)
  }
  cat("At rank 0, all received data are ", my_list, "\n")
} else {
    x <- c(my_rank + 100, my_rank + 200)
    cat("At rank ", my_rank, " am sending ", x, "\n")
    ierr = mpi.isend(x,type=1,dest=0,tag=my_rank, comm=0, request=0)
    mpi.wait(request=0,status=0)
}
ierr <- mpi.barrier(comm=0)
mpi.quit()
quit("yes")
```
- Demo:
```bash
$ mpirun -n 4 Rscript mpi_waitall.R 
At rank  3  am sending  103 203 
At rank  1  am sending  101 201 
At rank  2  am sending  102 202 
At rank 0, all received data are  101 201 102 202 103 203 
```
- In order to use async irecv, we need a long vector then must be able to access the vectory by index inside of irecv(). However, Rmpi may not allow such vector index approach

# Advanced topics

## Setting different random number seeds per rank
- The main idea on random number seed on MPI is that every rank must have different seeds
- The common approach is to broadcast some value (like 123) then each rank uses that value + its own rank number to get the unique number
- If multiple ranks have the same random number seed, they will do the same thing and it may hurt the statistics of simulations/computations

## Parallelizing existing packages

### Exhaustive search

### Greedy search
