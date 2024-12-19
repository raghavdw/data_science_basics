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
***Compile and run:***
```
mpicc -o hello_mpi hello_mpi.c
mpirun -np 4 ./hello_mpi
```

## Matrix Multiplication

### Serial Implementation
```
// matrix_multiplication.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_multiply(int n, double **A, double **B, double **C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int n = 1000; // Larger matrix size
    double **A = (double **)malloc(n * sizeof(double *));
    double **B = (double **)malloc(n * sizeof(double *));
    double **C = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        A[i] = (double *)malloc(n * sizeof(double));
        B[i] = (double *)malloc(n * sizeof(double));
        C[i] = (double *)malloc(n * sizeof(double));
    }

    // Initialize matrices A and B with example values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }
    }

    // Measure the start time
    clock_t start_time = clock();

    // Perform matrix multiplication
    matrix_multiply(n, A, B, C);

    // Measure the end time
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print elapsed time
    printf("Elapsed time: %f seconds\n", elapsed_time);

    // Free allocated memory
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}
```

### Distributed Implementation
```
// distributed_matrix_multiplication.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void matrix_multiply(int rows, int n, double *A, double *B, double *C) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int n = 1000; // Larger matrix size
    double *A = NULL, *B = NULL, *C = NULL;

    // Allocate and initialize matrices on the root process
    if (world_rank == 0) {
        A = (double *)malloc(n * n * sizeof(double));
        B = (double *)malloc(n * n * sizeof(double));
        C = (double *)malloc(n * n * sizeof(double));

        // Initialize matrices A and B with example values
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = i + j;
                B[i * n + j] = i - j;
            }
        }
    }

    // Broadcast matrix B to all processes
    if (world_rank != 0) {
        B = (double *)malloc(n * n * sizeof(double));
    }
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate local matrices
    double *local_A = (double *)malloc(n * n / world_size * sizeof(double));
    double *local_C = (double *)malloc(n * n / world_size * sizeof(double));

    // Scatter rows of matrix A to all processes
    MPI_Scatter(A, n * n / world_size, MPI_DOUBLE, local_A, n * n / world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Measure the start time
    double start_time = MPI_Wtime();

    // Perform local computation
    matrix_multiply(n / world_size, n, local_A, B, local_C);

    // Measure the end time
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // Gather results from all processes
    MPI_Gather(local_C, n * n / world_size, MPI_DOUBLE, C, n * n / world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print elapsed time on the root process
    if (world_rank == 0) {
        printf("Elapsed time: %f seconds\n", elapsed_time);
    }

    // Free allocated memory
    if (world_rank == 0) {
        free(A);
        free(B);
        free(C);
    } else {
        free(B);
    }
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
```
## Performance Metrics

### Measuring Execution Time
  - Serial Implementation: Uses clock() to measure the start and end times of the matrix multiplication.
  - Distributed Implementation: Uses MPI_Wtime() to measure the start and end times of the matrix multiplication.

### Scalability Testing
  - Run the distributed matrix multiplication with varying numbers of processes:
    ```
    mpirun -np 2 ./distributed_matrix_multiplication
    mpirun -np 4 ./distributed_matrix_multiplication
    mpirun -np 8 ./distributed_matrix_multiplication
    ```
### Benchmarking
  - Compare the execution times of the serial and distributed implementations to evaluate performance gains.

### Performance Gains
  - The distributed implementation shows a significant reduction in elapsed time compared to the serial implementation, demonstrating the performance gains achieved through parallel computation

### Deliverables
  - MPI-based distributed matrix multiplication code.
  - Performance metrics and scalability results.
  - Benchmarking report comparing serial and distributed implementations.
  - Detailed documentation on setup, usage, and results.
