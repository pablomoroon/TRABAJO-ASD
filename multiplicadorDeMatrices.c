#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 600

void init_matrix(double *M, int n) {
    for (int i = 0; i < n * n; i++) {
        M[i] = (double)(rand() % 100) / 10.0;
    }
}

void zero_matrix(double *M, int n) {
    for (int i = 0; i < n * n; i++) {
        M[i] = 0.0;
    }
}

void matmul_seq(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    srand(42);

    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Error reservando memoria\n");
        return 1;
    }

    init_matrix(A, N);
    init_matrix(B, N);
    zero_matrix(C, N);

    clock_t start = clock();
    matmul_seq(A, B, C, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Tiempo secuencial: %.6f segundos\n", elapsed);
    printf("C[0][0] = %f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}