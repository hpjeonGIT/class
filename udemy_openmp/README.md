## Title: The Complete Parallelism Course: From Zero to Expert!
- Instructor: Lucas Bazilio

## Section 1: Course Overview

1. Course Introduction

## Section 2: Parallelism Fundamentals

2. Introduction to Parallelism
![parallel_execution](./2_parallel_execution.png)

3. Expressing Tasks

4. Tasks and Dependencies
- Task dependence graph: graphical representation of the task decomposition

5. Task Dependency Graph
- Task dependency graph:
                - directed acyclic graph (no cycle. tree)
                - node=task, its weight represents the amount of work to be done
                - Edge=dependence, ie successor node can only execute after predecessor node has completed
- Parallel machine abstraction
                - P identical processors
                - Each processor executes a node at a time
                - T = \sum_nodes(work_node_i)
- Critical path: path in the task graph with the highest accumulated work
- Parallelism - T_1/T_inf, if sufficient processors are available
- P_min is the minimum number of processors necessary to achieve parallelism
![wrapup](./ch5_wrapup.png)

### Assignment 1:
(a) T1 = 10+10+9+10+10+6+8 = 63, T_inf = 10+9+8 =27, P = 63/27=2.33, Pmin = 3
(b) T1 = 10+10+10+10+6+11+ = 64, T_inf = 10+6+11+7 = 34, P=64/34 = 1.88, Pmin = 2

6. Granularity and Parallelism
```
count=0
for (i=0;i<n;i++) if (X[i].color == "Green") count ++;
```
- coarse grain decomposition: this task
- fine grain decomposition: loop over i

7. Task definition
- Can the computation be diviced in parts?
  - Task decomposition: functions, loop iterations
  - Data decomposition: elements of a vector, rows of a matrix
  - Task Dependency Graph (TDG)
- Metrics to understand how our task/data decomposition can potentially behave
  - T_1
  - T_inf
  - Parallelism = T_1/T_inf
  - Pmin = ceil(Parallelism)
- Factors: granualrity and overheads
  - Task granularity vs number of tasks
- Vector sum:
  - T_1 \prop n
  - T_inf \pro log2(n)
![vectorsum](./7_vectorsum.png)

8. Advanced Granularity
- Fine graind tasks vs coarse grained tasks
- Ex: Matrix-vector product
  - Inherent bound on hwo fine the granularity of a computation can be n^2
- Stencil computation using Jacobi solver
  - 4 neighbor elements of matrix u
  - snaptshot 16:19
  - trade-off b/w task granularity and task creation overhead

9. Speedup and Efficiency
- Speedup S_p = T_1 / T_p
- Efficiency E_p = S_p / P               
- Strong scaling: increase the number of processors P with constant problem: reduce the execution time
- Week scaling: increase the number of processors P with problem sizse Proportional to P: solve larger problems

10. Amdahl's law

11. Overhead sources
- Task creation
- Barrier synchronization
- Task synchronization
- Exclusive acces to shared data and data sharing
- T_p = (1-phi)*T_1 + phi * T_1/p + overhead(p)
- Amdahl's law can be overly pessimistic

12. Common overheads
- Data sharing: can be explicit via messages or implicit via memory hierarchy (caches)
- Idleness: dependencies, load imbalances, poor communication
- Computation: extra work like replication
- Memory: extra memory to obtain a parallel algorithm
- Contention: competition for the access to shared resource like memory/network
- How to model data sharing overhead?
  - Remote data access
    - Startup: time spent in preparing the remote access
    - Transfer; time spent in transferring the message
    - Synchronization b/w processors may be necessary

## Section 3: Solved Problems - Fundamentals of Parallelism

16. Problem 2
- T_1 = 500+2000+12000 + 500+2000+1000+500+20000+1000+500+2000+1000 = 43k
- T_inf = 500+20000+1000+1000 = 22.5k
- TDG for matrix product

23. Problem 5 - Question A
- S=9, seq=8,
- Tser= 8 + 4*x + 4*4
- Tser/8 = 8, x = 12
- Tser = 8+4*12+16 = 72

## Section 4: Task Decomposition | OpenMP

26. Task decomposition strategies
- Linear task decomposition: a task is a code block or a procedure invocation
```c
int main()
{
  init_A();
  init_B();
}
```           
- Iterative task decomposition: tasks found in iterative constructs such as for-loops/while loop
```c
for (int i=0;i<n;i++) C[i] - A[i] + B[i];
```
- Recursive task decomposition: tasks found in divide-and conquer problems and other recursive algorithm
  - OMP datasharing cannot handle recursive tasks

27. Recursive Task Decomposition
```c
#define N 1024
#define MIN_SIZE 64
void vector_add(int *A, int *B, int *C, int n)
{  for (int i=0;i<n;i++) C[i] = A[i] + B[i];
}
void rec_vector_add(int *A, int *B, int *C, int n)
{ if (n > MIN_SIZE) {
    int n2 = n/2;
                rec_vector_add(A,B,C,n2);
                rec_vector_add(A+n2, B+n2, C+n2, n-n2);
   } else vector_add(A,B,C,n);
}
```

28. Introduction to OpenMP
- int omp_get_num_threads(): number of available threads
- int omp_get_thread_num(): thread ID
- #pragma omp parallel: one implicit task is created
- #pragma omp task: one explicit task is created
- #pragma omp taskloop: explicit tasks are created

29. Task Generation Control
- Iterative task decompositions: we can control task granularity by setting the number of iterations executed by each task
- Recursive task decompositions: we can control task granularity by controlling recursion levels where tasks are generated (cutoff control)
  - After certain number of recursive calls (static)
  - When the size of the vector is too small (static)
  - When there are sufficient tasks pending to be executed (dynamic)
- How we make many chunks run on iterative decomposition?
```c
#pragma omp parallel
#pragma omp single
#pragma ompo taskloop grainsize(BS)
```
- firstprivate: specifies that each thread should have its own instance of a variable, and that the variable should be initialized with the value of the variable, because it exists before the parallel construct

30. Leaf Strategy
- Recursive task decomposition: divide and conquer
```c
void rec_dot_product(int *A, int *B, int n, in depth)
{  
  if (n > MIN_SIZE){
    int n2 = n/2;
    if (depth ==CUTOFF)
    #pragma omp task
    {  rec_dot_product(A,B,n2,depth+1);
        rec_dot_product(A+n2, B+n2, n-n2, depth+1);
    }
    else {
      rec_dot_product(A,B,n2,depth+1);
      rec_dot_product(A+n2,B+n2,n-n2,depth+1);
    }
  } else
      if (depth<=CUTOFF)
        #pragma omp task
        dot_product(A,B,n);
      else
        dot_product(A,B,n);
}
```
![leafstrategy](./30_leafstrategy.png)

31. Tree Strategy
```c
int dot_product(int *A, int *B, int n) {
  int tmp = 0;
  for (int i=0;i<n; i++) tmp += A[i]*B[i];
  return (tmp);
}
int rec_dot_product(int *A, int *B, int n) {
  int tmp1, tmp2=0;
  if (n>MIN_SIZE) {
      int n2 = n/2;
      #pragma omp task shared(tmp1)
      tmp1 = rec_dot_product(A,B,n2);
      #pragma omp task shared(tmp2)
      tmp2 = rec_dot_product(A+n2,B+n2, n-n2);
      #pragma omp taskwait
   } else tmp1 = dot_product(A,B,n);
   return (tmp1+tmp2);
}
void main() {
  #pragma omp parallel
  #pragma omp single
  result =rec_dot_product(a,b,N);
}
```
![treestrategy](./31_treestrategy.png)

32. Depth Recursion Control
- Tree strategy with depth recursion control
![depthrecursion](./32_depthrecursion.png)
```c
int rec_dot_product(int *A, int *B, int n, int depth) {
  int tmp1, tmp2=0;
  if (n>MIN_SIZE) {
    int n2 = n/2;
    if (depth < CUTOFF) {
        #pragma omp task shared(tmp1)
        tmp1 = rec_dot_product(A,B,n2);
        #pragma omp task shared(tmp2)
        tmp2 = rec_dot_product(A+n2,B+n2, n-n2);
        #pragma omp taskwait
      } else {
        tmp1 = rec_dot_product(A,B,n2,depth+1);
                    tmp2 = rec_dot_product(A+n2,B+n2,n-n2,depth+1);
      }
    } else tmp1 = dot_product(A,B,n);
   return (tmp1+tmp2);
}
```
- Or using omp_in_final()
```c
int rec_dot_product(int *A, int *B, int n, int depth) {
  int tmp1, tmp2=0;
  if (n>MIN_SIZE) {
    int n2 = n/2;
    if (!omp_in_final()) {
        #pragma omp task shared(tmp1) final(depth >=CUTOFF)
        tmp1 = rec_dot_product(A,B,n2);
        #pragma omp task shared(tmp2) final(depth >=CUTOFF)
        tmp2 = rec_dot_product(A+n2,B+n2, n-n2);
        #pragma omp taskwait
      } else {
        tmp1 = rec_dot_product(A,B,n2,depth+1);
                    tmp2 = rec_dot_product(A+n2,B+n2,n-n2,depth+1);
      }
    } else tmp1 = dot_product(A,B,n);
   return (tmp1+tmp2);
}
```

33. Atomic Directive
- Atomic access: mechanism to guarantee atomicity in load/store instructions
```c
#pragma omp atomic [update |read|write]
  expression
```
                - update: x += 1, x = x-foo(), x[index[i]]++ (read+write)
                - reads: value = *p
                - writes: *p = value
                - atomic without a close is equivalent to atomic update

34. Critical Directive
```c
#pragma omp critical [(name)]
structured block
```
- Provides a region of mutual exclusion where only one thread can be working at any given time
- By default all critical regions are the same
- Multiple mutual exclusion regions by providing them with a name
  - Only those with the same name synchronize
```c
int x=0,y=0;
#pragma omp parallel num_threads(4)
{
...
#pragma omp critical (x)
x++;
#pragma omp critical (y)
y++;
}
```
- x++ and y++ can be done simultaneously, using different threads.

35. Reduction Clause
```c
reduction(operator:list)
```
- valid operators are: +,-,*,|,||, &, &&, ^, min, max

36. Locks in OpenMP
- Special variables that live in memory with two basic operations
  - Acquire: while a thread has the lock, nobody else gets it. this allows the thread to do its work in private, not bothered by other threads
  - Release: allow other threads to acquire the lock and do their work (one at a time) in private
  - omp_init_lock, omp_set_lock, omp_unset_lock, omp_test_lock, omp_destroy_lock

## Section 5: Task Ordering | OpenMP

37. Taskwait and Taskgroup
- taskwait: suspends the execution of the current task, waiting on the completion of its child tasks. The taskwait construct is a stand-alone directive
```c
#pragma omp task {} //T1
#pragma omp task  //T2
{
  #pragma omp task // T3
}
#pragma om task {} // T4
#pragmaomp taskwait // Only T1, T2, T4 are guaranteed to have finished here
}
```
- taskgroup: suspends the execution of the current task at the end of structured block, waiting on the completion of child tasks of the current task and their descendent tasks
```c
#pragma omp task {} // T1
#pragma omp taskgroup{
{
  #prgam omp task // T2
  {
    #pragma omp task{} // T3
  }
  #pragma omp task {} //T4
}
// Here, only T2, T3, T4 are guaranteed to have finished here.
```

38. Task Dependency Clauses
```c
#prgma omp task [depend (in: var_list)]
                [depend (out: var_list)]
                [depend (inout: var_list)]
```

39. Wavefront Example
![wavefront](./39_wavefront.png)
![wavefront2](./39_wavefront2.png)

## Section 6: Solved Problems - OpenMP

40. Problem 1 - Question A
- Parallelize count_key()
```c
#include <omp.h>
#include <stdio.h>
#define N 131072
long count_key(long Nlen, long *a, long key) {
    long count = 0;
    for (int i=0; i<Nlen; i++)
      if (a[i] == key) count ++;
    return count;
}
long count_iter(long Nlen, long *a, long key) {
    long count = 0;
    #pragma omp parallel reduction(+:count)
    {
    for (int i=0; i<Nlen; i++)
      if (a[i] == key) count ++;
    }
    return count;
}
long count_task(long Nlen, long *a, long key) {
    long count = 0;
    long tmp1=0, tmp2=0;
    if (Nlen>1000) {
      #pragma omp task shared(tmp1)
      tmp1 = count_task(Nlen/2, a, key);
      #pragma omp task shared(tmp2)
      tmp2 = count_task(Nlen/2, a+Nlen/2, key);
      #pragma omp taskwait
    } else tmp1 = count_key(Nlen,a,key);
    return tmp1+tmp2;
}
long count_recur(long Nlen, long *a, long key) {
    long count = 0;
    #pragma omp parallel reduction(+:count)
    {
    for (int i=0; i<Nlen; i++)
      if (a[i] == key) count ++;
    }
    return count;
}
int main() {
    long a[N], key = 42, nkey = 0;
    for (long i=0; i<N; i++) a[i] = random()*N;
    a[N%43] = key; a[N%73] = key; a[N%3] = key;
    #pragma omp parallel
    {
        #pragma omp single // prints only once
        printf("Number of threads = %d\n", omp_get_num_threads());
    }
    //
    double c0 = omp_get_wtime();
    nkey = count_key(N, a, key); // count key by sequentially
    printf("serial nkey = %d wall time = %f sec\n", nkey, omp_get_wtime() - c0);
    double c1 = omp_get_wtime();
    #pragma omp parallel
    {
       #pragma omp simgle
       {
         nkey = count_iter(N, a, key); // using iterative decomposition
       }
    }
    printf("omp iter nkey = %d wall time = %f sec\n", nkey, omp_get_wtime() - c1);
    double c2 = omp_get_wtime();
    #pragma omp parallel
    {
       #pragma omp simgle
       {
         nkey = count_task(N, a, key); // using iterative decomposition
      }
    }
    printf("omp task nkey = %d wall time = %f sec\n", nkey, omp_get_wtime() - c2);
    //neky = count_recur(N, a, key); // using divide and conquer
}
```
