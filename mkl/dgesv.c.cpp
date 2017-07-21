//
// Created by Myungjun Son on 2017. 7. 21..
//

/*******************************************************************************
*  Copyright (C) 2009-2015 Intel Corporation. All Rights Reserved.
*  The information and material ("Material") provided below is owned by Intel
*  Corporation or its suppliers or licensors, and title to such Material remains
*  with Intel Corporation or its suppliers or licensors. The Material contains
*  proprietary information of Intel or its suppliers and licensors. The Material
*  is protected by worldwide copyright laws and treaty provisions. No part of
*  the Material may be copied, reproduced, published, uploaded, posted,
*  transmitted, or distributed in any way without Intel's prior express written
*  permission. No license under any patent, copyright or other intellectual
*  property rights in the Material is granted to or conferred upon you, either
*  expressly, by implication, inducement, estoppel or otherwise. Any license
*  under such intellectual property rights must be express and approved by Intel
*  in writing.
*
********************************************************************************
*/
/*
   LAPACKE_dgesv Example.
   ======================

   The program computes the solution to the system of linear
   equations with a square matrix A and multiple
   right-hand sides B, where A is the coefficient matrix:

     6.80  -6.05  -0.45   8.32  -9.67
    -2.11  -3.30   2.58   2.71  -5.14
     5.66   5.36  -2.70   4.35  -7.26
     5.97  -4.44   0.27  -7.17   6.08
     8.23   1.08   9.04   2.14  -6.87

   and B is the right-hand side matrix:

     4.02  -1.56   9.81
     6.19   4.00  -4.09
    -8.22  -8.67  -4.57
    -7.57   1.75  -8.61
    -3.03   2.86   8.99

   Description.
   ============

   The routine solves for X the system of linear equations A*X = B,
   where A is an n-by-n matrix, the columns of matrix B are individual
   right-hand sides, and the columns of X are the corresponding
   solutions.

   The LU decomposition with partial pivoting and row interchanges is
   used to factor A as A = P*L*U, where P is a permutation matrix, L
   is unit lower triangular, and U is upper triangular. The factored
   form of A is then used to solve the system of equations A*X = B.

   Example Program Results.
   ========================

 LAPACKE_dgesv (column-major, high-level) Example Program Results

 Solution
  -0.80  -0.39   0.96
  -0.70  -0.55   0.22
   0.59   0.84   1.90
   1.32  -0.10   5.36
   0.57   0.11   4.04

 Details of LU factorization
   8.23   1.08   9.04   2.14  -6.87
   0.83  -6.94  -7.92   6.55  -3.99
   0.69  -0.67 -14.18   7.24  -5.19
   0.73   0.75   0.02 -13.82  14.19
  -0.26   0.44  -0.59  -0.34  -3.43

 Pivot indices
      5      5      3      4      5
*/
#include <stdlib.h>
#include <stdio.h>
#include "mkl_lapacke.h"
#include <time.h>

/* Auxiliary routines prototypes */
extern void print_matrix(char *desc, MKL_INT m, MKL_INT n, double *a, MKL_INT lda);

extern void print_int_vector(char *desc, MKL_INT n, MKL_INT *a);

//void print_arr(MKL_INT N, char *name, double *array);

void init_arr(MKL_INT N, double *a);

/* Parameters */
//#define N 5
#define NRHS N
#define LDA N
#define LDB N

/* Main program */
int main(int argc, char *argv[]) {
    clock_t start, stop;
//    INT N;
//	MKL_INT N,lda = LDA, ldb= LDB, n = N, nrhs=NRHS, info;
    MKL_INT N;
    srand((unsigned) time(NULL));
//	MKL_INT ipiv[N];
    double *a;
    double *b;
    if (argc < 2) {
        printf("Enter matrix size N=");
        //please enter small number first to ensure that the
        //multiplication is correct! and then you may enter
        //a "reasonably" large number say like 500 or even 1000
        scanf("%lld", &N);
    } else {
        N = atoi(argv[1]);
    }
    MKL_INT lda = LDA, ldb = LDB, n = N, nrhs = NRHS, info;

    MKL_INT ipiv[N];

//    printf("%lld\n", N);
//	printf("%lld\n", LDA);
//	printf("%lld\n", NRHS);
//	printf("%lld\n", LDB);
//	printf("n is %lld\n", n);
//	printf("lda is %lld\n", lda);

    a = (double *) malloc(sizeof(double) * N * LDA);
    b = (double *) malloc(sizeof(double) * NRHS * LDB);
    init_arr(N, a);
    init_arr(N, b);


    //reallocate to cash to be flushed
    a = (double *) malloc(sizeof(double) * N * LDA);
    b = (double *) malloc(sizeof(double) * NRHS * LDB);
    init_arr(N, a);
    init_arr(N, b);


    if (N < 10) {
        print_matrix("A matrix", n, n, a, lda);
        print_matrix("B matrix", n, nrhs, b, ldb);
    }
    start = clock();

    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);

    stop = clock();

    printf("\ndgesv(). Elapsed time = %g seconds\n\n",
           ((double) (stop - start)) / CLOCKS_PER_SEC);
    /* Check for the exact singularity */
    if (info > 0) {
        printf("The diagonal element of the triangular factor of A,\n");
        printf("U(%lli,%lli) is zero, so that A is singular;\n", info, info);
        printf("the solution could not be computed.\n");
        exit(1);
    }
    /* Print solution */
    if (N < 10) {
        print_matrix("Solution", n, nrhs, b, ldb);
    }
    free(a);
    free(b);
    return 0;
}

void init_arr(MKL_INT N, double *a) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[i * N + j] = rand() % 100; //keep all entries less than 10. pleasing to the eye!
        }
    }
}

//void print_arr(MKL_INT N, char *name, double *array) {
//    int i, j;
//    printf("\n%s\n", name);
//    for (i = 0; i < N; i++) {
//        for (j = 0; j < N; j++) {
//            printf("%g\t", array[N * i + j]);
//        }
//        printf("\n");
//    }
//}

void print_matrix(char *desc, MKL_INT m, MKL_INT n, double *a, MKL_INT lda) {
    MKL_INT i, j;
    printf("\n %s\n", desc);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) printf(" %6.2f", a[i + j * lda]);
        printf("\n");
    }
}