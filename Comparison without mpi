#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 1000

// Function to multiply matrix A and vector x
void mat_vec_mult(int n, double A[n][n], double *x, double *result) {
    for (int i = 0; i < n; i++) {
        result[i] = 0;
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * x[j];
        }
    }
}

// Function to calculate dot product of two vectors
double dot_product(int n, double *v1, double *v2) {
    double dot = 0.0;
    for (int i = 0; i < n; i++) {
        dot += v1[i] * v2[i];
    }
    return dot;
}

// Function to perform vector addition
void vec_add(int n, double *v1, double *v2, double *result, double alpha) {
    for (int i = 0; i < n; i++) {
        result[i] = v1[i] + alpha * v2[i];
    }
}

// Conjugate Gradient method
void conjugate_gradient(int n, double A[n][n], double *b, double *x, int max_iter, double tol, int *iters) {
    double r[n], p[n], Ap[n], rsold, rsnew;

    // r = b - A * x
    mat_vec_mult(n, A, x, Ap);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }
    rsold = dot_product(n, r, r);

    for (int i = 0; i < max_iter; i++) {
        mat_vec_mult(n, A, p, Ap);
        double alpha = rsold / dot_product(n, p, Ap);
        vec_add(n, x, p, x, alpha);
        vec_add(n, r, Ap, r, -alpha);
        rsnew = dot_product(n, r, r);
        if (sqrt(rsnew) < tol) {
            *iters = i + 1;
            return;
        }
        for (int j = 0; j < n; j++) {
            p[j] = r[j] + (rsnew / rsold) * p[j];
        }
        rsold = rsnew;
    }
    *iters = max_iter;
}

// Gauss-Seidel method
void gauss_seidel(int n, double A[n][n], double *b, double *x, int max_iter, double tol, int *iters) {
    double x_old[n];

    for (int k = 0; k < max_iter; k++) {
        for (int i = 0; i < n; i++) {
            x_old[i] = x[i];
        }

        for (int i = 0; i < n; i++) {
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
        for (int i = 0; i < n; i++) {
            norm += fabs(x[i] - x_old[i]);
        }
        if (norm < tol) {
            *iters = k + 1;
            return;
        }
    }
    *iters = max_iter;
}

// Function to print a vector
void print_vector(int n, double *v) {
    for (int i = 0; i < n; i++) {
        printf("%f ", v[i]);
    }
    printf("\n");
}

int main() {
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
    srand(time(0));
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
    conjugate_gradient(n, A, b, x_cg, max_iter, tol, &iters_cg);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Conjugate Gradient Solution:\n");
    //print_vector(n, x_cg);
    printf("Iterations: %d\n", iters_cg);
    printf("Time: %f seconds\n", cpu_time_used);

    // Gauss-Seidel Method
    start = clock();
    gauss_seidel(n, A, b, x_gs, max_iter, tol, &iters_gs);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Gauss-Seidel Solution:\n");
    //print_vector(n, x_gs);
    printf("Iterations: %d\n", iters_gs);
    printf("Time: %f seconds\n", cpu_time_used);

    return 0;
}
