#include "func01.h"
#include "gtest/gtest.h"
#include "mpi.h"

class MyUnitTestMPI : public ::testing::Test {
protected:
    int myid, num_procs;
    MyUnitTestMPI() {}
    virtual ~MyUnitTestMPI() {}
    virtual void SetUp() {
      MPI_Init(NULL, NULL);
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    }
    virtual void TearDown() {
      MPI_Finalize();
    }
};

TEST_F(MyUnitTestMPI, testingMPI) {
     int all_sum, local_sum=0;
     for (int i=0; i< num_procs; i++) local_sum += i;
     MPI_Allreduce(&myid, &all_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
     EXPECT_EQ(local_sum,all_sum);
}

int main(int argc, char** argv) {
     testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();
}

