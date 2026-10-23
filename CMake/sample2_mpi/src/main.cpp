#include <iostream>
#include "func01.h"
#include "mpi.h"
int main(int argc, char** argv){
  int myid, num_procs;
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  //
  if (myid == 0) {
    int x = 123;
    int z = ret_2x(x);
    std::cout << " I am rank 0: " << "Input = " << x << " Answer = " << z << std::endl;
  }
  //
  MPI_Finalize();
  return 0;
}
