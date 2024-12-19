<<<<<<< HEAD
# data_science_basics
=======
# Distributed Matrix Multiplication using MPI

## Project Overview

### Objective
The objective of this project is to implement and evaluate the performance of matrix multiplication across multiple nodes using MPI (Message Passing Interface). The project involves setting up an MPI development environment, implementing both serial and distributed matrix multiplication algorithms, measuring performance metrics, and benchmarking the distributed implementation against the serial implementation.

### Tasks
1. **Environment Setup:** Set up an MPI development environment.
2. **Matrix Multiplication:** Implement a standard matrix multiplication algorithm.
3. **Distributed Implementation:** Modify the algorithm for distributed computation using MPI, focusing on data partitioning and inter-process communication.
4. **Performance Metrics:** Develop a system to measure execution time and scalability.
5. **Scalability Testing:** Test the algorithm on different numbers of nodes/processes.
6. **Benchmarking:** Benchmark against a serial implementation to evaluate performance gains.
7. **Deliverables:** MPI-based distributed matrix multiplication code, performance metrics, benchmarking report, and detailed documentation.

## Environment Setup

### Installing MPI (OpenMPI) on macOS
1. **Install OpenMPI using Homebrew:**
    ```
    brew install open-mpi 
    ```
2. **Set up SSH keys for passwordless SSH:**
    ``` 
    ssh-keygen -t rsa
    ssh-copy-id user@remote_host
    ```
3. **Verify MPI Installation:**
```
// hello_mpi.c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("Hello world from rank %d out of %d processors\n", world_rank, world_size);
    MPI_Finalize();
    return 0;
}
```
>>>>>>> 652c7a0 (distributed matrix multiplication)
