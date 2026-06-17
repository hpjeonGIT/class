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

## Installing pbdMPI
```bash
$ module load R_4.6 openmpi_5.0.10
$ R CMD INSTALL ./float_0.3-3.tar.gz 
$ R CMD INSTALL pbdMPI_0.5-5.tar.gz --configure-args="--with-Rmpi-include=/home/hpjeon/sw_local/openmpi/5.0.10/include/ --with-Rmpi-libpath=/home/hpjeon/sw_local/openmpi/5.0.10/lib --with-Rmpi-type=OPENMPI"
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
- For pbdMPI:
```r
library(pbdMPI, quietly=TRUE)
my_rank <- comm.rank()
n_ranks <- comm.size()
cat("hello world from ", my_rank, " out of ", n_ranks, "\n")
ierr <- barrier();
finalize();
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
- For pbdMPI:
```r
library(pbdMPI, quietly=TRUE)
my_rank <- comm.rank()
n_ranks <- comm.size()
## Reduce example:
res <- reduce(my_rank,NULL,op="sum",rank.dest=0,comm=0)
if (my_rank==0) { cat("At rank0, the sum of all ranks = ", res, "\n") }
## Allreduce example:
res <- allreduce(my_rank,NULL,op="sum", comm=0)
cat("At rank ", my_rank, " the all reduced value = ", res,'\n');
## Broadcast example
if (my_rank == 0) { 
  my_value <- 100;
} else {
  my_value <- 10000;
}
my_value <- bcast(my_value, rank.source=0, comm=0) # unlike regular MPI_Bcast, my_value is not updated in other ranks. The return value must be assigned
Sys.sleep(my_rank)
cat("At rank ", my_rank, " the broadcasted value = ", my_value,'\n');
# 
ierr <- barrier(comm=0)
finalize()
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
- For pbdMPI:
```r
library(pbdMPI, quietly=TRUE)
my_rank <- comm.rank()
n_ranks <- comm.size()
my_list <- c()
if (my_rank == 0) {
  my_list <- c()
  for (i in 1:(n_ranks-1)) {
    x <- integer(2) # irecv/recv argument is a vector as isend/send can send a vector
    x = recv(NULL, rank.source=i, tag=i, comm=0)
    my_list <- c(my_list,x)
  }
  cat("At rank 0, all received data are ", my_list, "\n")
} else {
    x <- c(my_rank + 100, my_rank + 200)
    cat("At rank ", my_rank, " am sending ", x, "\n")
    ierr = send(x,rank.dest=0,tag=my_rank, comm=0)
}
ierr <- barrier(comm=0)
finalize()
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

#### Source of exhaustive search from FSelector package
- Source: https://cran.r-project.org/web/packages/FSelector/index.html
  - Download the source package and find search.exhaustive.R
```r
exhaustive.search <- function(attributes, eval.fun) {
	len = length(attributes)
	if(len == 0)
		stop("Attributes not specified")
	
	eval.fun = match.fun(eval.fun)
	best = list(
		result = -Inf,
		attrs = rep(0, len)
	)
	
	# main loop
	# for each subset size
	for(size in 1:len) {
		child_comb = combn(1:len, size)
		# for each child
		for(i in 1:dim(child_comb)[2]) {
			subset = rep(0, len)
			subset[child_comb[, i]] = 1
			result = eval.fun(attributes[as.logical(subset)])
			if(result > best$result) {
				best$result = result
				best$attrs = subset
			}
		}
	}
	return(attributes[as.logical(best$attrs)])
}
```
- As shown above, the exhaustive search is composed of 2 for-loops. Looping over the given length, number of combinations is calcuated then 2nd loop runs through those combinations
  - Outer loop: 1-len
  - Inner loop: 1-child_comb per each outer loop index
- Parallelization strategy:
  - Total number of calculation = outer_loop \* inner_loop
  - We divide the total number of calculation with the number of ranks available
  - Then we find the initial loop index and final index per rank

#### Divider of workload
- A sample case
  - When len=10, inner loops are 10, 45, 120, 210, 252, 210, 120, 45, 10, 1 and total inner loops are 1023
  - When 4 ranks are used, we may divide them with 256 each while the last rank has 255
    - 256 + 256 + 256 + 255 = 1023
  - Rank 0 will have 1-10, 1-45, 1-120, and 1-81 (partial loop over 210)
  - Rank 1 will have 82-210 and 1-127 over 252
  - Rank 2 will have 128-252 and 1-131 over 210
  - Rank 3 will have 132-210, 1-120, 1-45, 1-10, 1-1
- A following script shows such division over 4 ranks
```r  
divider <- function(inner, indx_start, indx_end) {
  outer <- c()
  inner_s <- c()
  inner_e <- c()
  tsum <- 0
  for(size in 1:length(inner)){
     dt <- inner[size]
     tsum <- tsum +  dt
     if (tsum >= indx_start) {
       outer <- c(outer, size)
       istart = dt-(tsum-indx_start)
       if (istart <0) istart=1
       inner_s <- c(inner_s, istart)
       if (tsum <= indx_end) {
         inner_e <- c(inner_e, dt)
       } else {
         inner_e <- c(inner_e, dt- (tsum-indx_end))
         break
       }
     }
  }
  return(list(outer=outer, inner_s=inner_s, inner_e=inner_e))
}
##
len <- 10
tsum <-0
inner <- c()
for(size in 1:len) {
  child_comb = combn(1:len, size)
  x = dim(child_comb)[2]
  inner <- c(inner, x)
  tsum <- tsum + x
#cat (size, " ", x, "\n")
}
cat("total size =", tsum, "\n")
cat("inner loop = ", inner, "\n")
ncpus = 4
div = round(tsum/ncpus)
test_sum <- 0
for (my_rank in 0:(ncpus-1)) {
  indx_start = div*my_rank + 1
  indx_end = indx_start + div - 1
  if (my_rank == (ncpus-1)) indx_end = tsum
  ###
  res <- divider(inner, indx_start, indx_end)
  outer <- res$outer
  inner_s <- res$inner_s
  inner_e <- res$inner_e
  cat("my rank = ", my_rank, " outloop = ", outer, " inner loop starting index = ", inner_s, " inner loop end index= ", inner_e, "\n")
  for(i in 1:length(outer)) {
    for (j in inner_s[i]:inner_e[i]) {
      test_sum <- test_sum + 1
    }
  }
}
cat("after division, total sum = ", test_sum, "\n")
## 210 = (rank 0: 1 : 2)
## 210 = (rank 0: 1)
## 210 = (rank 0)
```
#### Implementation of Rmpi over exhaustive search
```r
divider <- function(inner, indx_start, indx_end) {
  outer <- c()
  inner_s <- c()
  inner_e <- c()
  tsum <- 0
  for(size in 1:length(inner)){
     dt <- inner[size]
     tsum <- tsum +  dt
     if (tsum >= indx_start) {
       outer <- c(outer, size)
       istart = dt-(tsum-indx_start)
       if (istart <0) istart=1
       inner_s <- c(inner_s, istart)
       if (tsum <= indx_end) {
         inner_e <- c(inner_e, dt)
       } else {
         inner_e <- c(inner_e, dt- (tsum-indx_end))
         break
       }
     }
  }
  return(list(outer=outer, inner_s=inner_s, inner_e=inner_e))
}
exhaustive.search.Rmpi <- function(attributes, eval.fun) {
	len = length(attributes)
	if(len == 0)
		stop("Attributes not specified")
	
	eval.fun = match.fun(eval.fun)
	best = list(
		result = -Inf,
		attrs = rep(0, len)
	)	
	best.out = as.integer(0)
	best.inn = as.integer(0)
	rs.vec = rep(0,mpi.ncpus)
  rs.out = rep(as.integer(0), mpi.ncpus) # outer loop index
	rs.inn = rep(as.integer(0), mpi.ncpus) # inner loop index
  # Estimation of total loops
	tsum <-0
	inner <- c()
	for(size in 1:len) {
		child_comb = combn(1:len, size)
		x = dim(child_comb)[2]
		inner <- c(inner, x)
		tsum <- tsum + x
	}
	div = round(tsum/mpi.ncpus)
	indx_start = div*mpi.my_rank + 1
  indx_end = indx_start + div - 1
  if (mpi.my_rank == (mpi.ncpus-1)) indx_end = tsum
  res <- divider(inner, indx_start, indx_end)
  outer <- res$outer
  inner_s <- res$inner_s
  inner_e <- res$inner_e
	# Now we have outer, inner_s, inner_e
	# main loop
	# for each subset size
	#for(size in 1:len) {
	for (nsize in 1:length(outer)) {	
		size = outer[nsize]
		child_comb = combn(1:len, size)
		# for each child
		#for(i in 1:dim(child_comb)[2]) {
		for(i in inner_s[nsize]:inner_e[nsize]) {
			subset = rep(0, len)
			subset[child_comb[, i]] = 1
			result = eval.fun(attributes[as.logical(subset)])
			if(result > best$result) {
				best$result = result
				best$attrs = subset
				best.out <- as.integer(size)
				best.inn <- as.integer(i)
			}
		}
	}
  mpi.allgather(best$result, type=2, rs.vec, comm=0)
	mpi.allgather(best.out,    type=1, rs.out, comm=0)
	mpi.allgather(best.inn,    type=1, rs.inn, comm=0)
	update <- FALSE
	for (i in 1:mpi.ncpus) {
		if (rs.vec[i] > best$result) {
				best$result <- rs.vec[i]
				best.out  <- rs.out[i]
				best.inn  <- rs.inn[i]
				update <- TRUE
		}
	}
	if (update) {
		  child_comb = combn(1:len,best.out)
			subset = rep(0, len)
			subset[child_comb[, best.inn]] = 1
			best$attrs <- subset
	}
	return(attributes[as.logical(best$attrs)])
}
```
- To find maximum value, we may use reduce() or allreduce() but as we cannot reduce `best$attrs` as this is a vector. Therefore, we allgather `best$result` and `best$attrs` then find max value locally

### Greedy search
