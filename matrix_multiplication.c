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
    int n = 1009; // Larger matrix size
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

    // Print result matrix C
    printf("Result Matrix C:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", C[i][j]);
        }
        printf("\n");
    }

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