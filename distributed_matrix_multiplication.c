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

    int n = 1000; // Large matrix size
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

    // Print result matrix C and elapsed time on the root process
    if (world_rank == 0) {
        printf("Result Matrix C:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", C[i * n + j]);
            }
            printf("\n");
        }
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