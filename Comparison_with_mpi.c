#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 99

void mat_vec_mult(int n, double A[n][n], double *x, double *result, int start, int end) {
    for (int i = start; i < end; i++) {
        result[i] = 0;
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * x[j];
        }
    }
}

double dot_product(int n, double *v1, double *v2, int start, int end) {
    double dot = 0.0;
    for (int i = start; i < end; i++) {
        dot += v1[i] * v2[i];
    }
    return dot;
}

void vec_add(int n, double *v1, double *v2, double *result, double alpha, int start, int end) {
    for (int i = start; i < end; i++) {
        result[i] = v1[i] + alpha * v2[i];
    }
}

void conjugate_gradient(int n, double A[n][n], double *b, double *x, int max_iter, double tol, int *iters, int rank, int size) {
    double r[n], p[n], Ap[n], rsold, rsnew;
    int chunk = n / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? n : (rank + 1) * chunk;

    mat_vec_mult(n, A, x, Ap, start, end);
    for (int i = start; i < end; i++) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, r, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    rsold = dot_product(n, r, r, start, end);
    MPI_Allreduce(MPI_IN_PLACE, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < max_iter; i++) {
        mat_vec_mult(n, A, p, Ap, start, end);
        double alpha = rsold / dot_product(n, p, Ap, start, end);
        MPI_Allreduce(MPI_IN_PLACE, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        vec_add(n, x, p, x, alpha, start, end);
        vec_add(n, r, Ap, r, -alpha, start, end);
        rsnew = dot_product(n, r, r, start, end);
        MPI_Allreduce(MPI_IN_PLACE, &rsnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (sqrt(rsnew) < tol) {
            *iters = i + 1;
            return;
        }
        for (int j = start; j < end; j++) {
            p[j] = r[j] + (rsnew / rsold) * p[j];
        }
        rsold = rsnew;
    }
    *iters = max_iter;
}

void gauss_seidel(int n, double A[n][n], double *b, double *x, int max_iter, double tol, int *iters, int rank, int size) {
    double x_old[n];
    int chunk = n / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? n : (rank + 1) * chunk;

    for (int k = 0; k < max_iter; k++) {
        MPI_Allgather(x + start, chunk, MPI_DOUBLE, x_old, chunk, MPI_DOUBLE, MPI_COMM_WORLD);
        for (int i = start; i < end; i++) {
            double sum1 = 0.0;
            double sum2 = 0.0;
            for (int j = 0; j < i; j++) {
                sum1 += A[i][j] * x[j];
            }
            for (int j = i + 1; j < n; j++) {
                sum2 += A[i][j] * x_old[j];
            }
            x[i] = (b[i] - sum1 - sum2) / A[i][i];
        }
        double norm = 0.0;
        for (int i = start; i < end; i++) {
            norm += fabs(x[i] - x_old[i]);
        }
        MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (norm < tol) {
            *iters = k + 1;
            return;
        }
    }
    *iters = max_iter;
}

void print_vector(int n, double *v) {
    for (int i = 0; i < n; i++) {
        printf("%f ", v[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = N;
    double A[N][N];
    double b[N];
    double x_cg[N] = {0};
    double x_gs[N] = {0};
    int max_iter = 1000;
    double tol = 1e-10;
    int iters_cg, iters_gs;
    clock_t start, end;
    double cpu_time_used;

    // Initialize A as a symmetric positive definite matrix and b
    srand(time(0) + rank);
    for (int i = 0; i < n; i++) {
        b[i] = rand() % 10;
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 10;
        }
    }

    // Make A symmetric positive definite
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            A[i][j] = A[j][i];
        }
        A[i][i] += n;
    }

    // Conjugate Gradient Method
    start = clock();
    conjugate_gradient(n, A, b, x_cg, max_iter, tol, &iters_cg, rank, size);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    if (rank == 0) {
        printf("Conjugate Gradient Solution:\n");
        //print_vector(n, x_cg);
        printf("Iterations: %d\n", iters_cg);
        printf("Time: %f seconds\n", cpu_time_used);
    }

    // Gauss-Seidel Method
    start = clock();
    gauss_seidel(n, A, b, x_gs, max_iter, tol, &iters_gs, rank, size);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    if (rank == 0) {
        printf("Gauss-Seidel Solution:\n");
        //print_vector(n, x_gs);
        printf("Iterations: %d\n", iters_gs);
        printf("Time: %f seconds\n", cpu_time_used);
    }

    MPI_Finalize();
    return 0;
}
