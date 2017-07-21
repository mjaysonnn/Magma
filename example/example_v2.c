// This is a simple standalone example. See README.txt

#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"     //CUblas는 어디서 왔더라 Cuda버전 BLAS를 뜻한다
#include "magma_v2.h"      //
#include "magma_lapack.h"  // if you need BLAS & LAPACK


// ------------------------------------------------------------
// Replace with your code to initialize the A matrix.
// This simply initializes it to random values.
// Note that A is stored column-wise, not row-wise.                 
//
// m   - number of rows,    m >= 0.
// n   - number of columns, n >= 0.
// A   - m-by-n array of size lda*n.
// lda - leading dimension of A, lda >= m.                          
//
// When lda > m, rows (m, ..., lda-1) below the bottom of the matrix are ignored.       
// This is helpful for working with sub-matrices, and for aligning the top
// of columns to memory boundaries (or avoiding such alignment).
// Significantly better memory performance is achieved by having the outer loop
// over columns (j), and the inner loop over rows (i), than the reverse.
void zfill_matrix(
        magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda) {
#define A(i_, j_) A[ (i_) + (j_)*lda ]  //column-wise이다 보니 이렇게 식이 생긴다

    magma_int_t i, j;
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i) {
            A(i, j) = MAGMA_Z_MAKE(rand() /
                                   ((double) RAND_MAX),    // real part   실수를 뜻한다           //MAGMA_Z_MAKE란 하나의 complex number를 만드는 것이다
                                   rand() / ((double) RAND_MAX));  // imag part , 허수를 뜻한다
        }
    }

#undef A
}


// ------------------------------------------------------------

/*// ____________________________________________________________               임의로 행렬을 만들기 위해 새로운 함수를 하나 만듬
void zfill_matrix1(
        magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda )
{
#define A(i_, j_) A[ (i_) + (j_)*lda ]

    //magma_int_t i, j;
    //for (j=0; j < n; ++j) {
    //    for (i=0; i < m; ++i) {
    //        A(i,j) = MAGMA_Z_MAKE( rand() / ((double) RAND_MAX),0);}    // real part
    //rand() / ((double) RAND_MAX) );  // imag part
    //A(i,j)=MAGMA_Z_MAKE(2,0);
    //A(i,j)=
    A(0,0)=MAGMA_Z_MAKE(3,0);
    A(1,0)=MAGMA_Z_MAKE(1,0);
    A(2,0)=MAGMA_Z_MAKE(2,0);
    A(0,1)=MAGMA_Z_MAKE(2,0);
    A(1,1)=MAGMA_Z_MAKE(0,0);
    A(2,1)=MAGMA_Z_MAKE(0,0);
    A(0,2)=MAGMA_Z_MAKE(1,0);
    A(1,2)=MAGMA_Z_MAKE(0,0);
    A(2,2)=MAGMA_Z_MAKE(2,0);
    //}
    //}
    // A = [ 2 1 0 1; 2 1 2 3; 0 0 1 2; -4 -1 0 -2];
#undef A
}
// ------------------------------------------------------------
 */

/*// _____________________________________   이것도 B행렬을 임의적으로 만들기 위해 함수를 만
void zfill_matrix2(
        magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda )
{
#define A(i_, j_) A[ (i_) + (j_)*lda ]

    //magma_int_t i, j;
    //for (j=0; j < n; ++j) {
    //    for (i=0; i < m; ++i) {
    //        A(i,j) = MAGMA_Z_MAKE( rand() / ((double) RAND_MAX),0);}    // real part
    //rand() / ((double) RAND_MAX) );  // imag part
    //A(i,j)=MAGMA_Z_MAKE(2,0);
    //A(i,j)=
    A(0,0)=MAGMA_Z_MAKE(2,0);
    A(1,0)=MAGMA_Z_MAKE(1,0);
    A(2,0)=MAGMA_Z_MAKE(8,0);
#undef A
}

// ------------------------------------------------------------
*/

// Replace with your code to initialize the X rhs.
void zfill_rhs(
        magma_int_t m, magma_int_t nrhs, magmaDoubleComplex *X, magma_int_t ldx) {
    zfill_matrix(m, nrhs, X, ldx);      //Ax=B 에서 B를 만드는 코드
}


// ------------------------------------------------------------
// Replace with your code to initialize the dA matrix on the GPU device.
// This simply leverages the CPU version above to initialize it to random values,
// and copies the matrix to the GPU.
void zfill_matrix_gpu(
        magma_int_t m, magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldda,
        magma_queue_t queue) {
    magmaDoubleComplex *A;
    magma_int_t lda = ldda;
    magma_zmalloc_cpu(&A, m * lda);                     //GPU로 복사 하기 위해 동적 할당을 해준다
    if (A == NULL) {
        fprintf(stderr, "malloc failed\n");
        return;
    }
    zfill_matrix(m, n, A, lda);
    magma_zsetmatrix(m, n, A, lda, dA, ldda, queue);   //이게 CPU에서 GPU로 복사해주는 코드
    magma_free_cpu(A);
}

// ------------------------------------------------------------
/*
// ------------------------------------------------------------    // 임의적으로 행렬을 만들기 위해 만들어주는 코드
// Replace with your code to initialize the dA matrix on the GPU device.
// This simply leverages the CPU version above to initialize it to random values,
// and copies the matrix to the GPU.
void zfill_matrix_gpu1(
        magma_int_t m, magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldda,
        magma_queue_t queue )
{
    magmaDoubleComplex *A;
    magma_int_t lda = ldda;
    magma_zmalloc_cpu( &A, m*lda );
    if (A == NULL) {
        fprintf( stderr, "malloc failed\n" );
        return;
    }
    //    zfill_matrix1( m, n, A, lda );
//                      zfill_matrix1( m, n, A, lda );
    zfill_matrix2( m, n, A, lda );
    magma_zsetmatrix( m, n, A, lda, dA, ldda, queue );
    magma_free_cpu( A );
}


// ------------------------------------------------------------
*/
// Replace with your code to initialize the dX rhs on the GPU device.
void zfill_rhs_gpu(
        magma_int_t m, magma_int_t nrhs, magmaDoubleComplex *dX, magma_int_t lddx,
        magma_queue_t queue) {
    zfill_matrix_gpu(m, nrhs, dX, lddx, queue);           //GPU에서 Ax=B 중 B를 만드는 코드
}


// ------------------------------------------------------------
// Solve A * X = B, where A and X are stored in CPU host memory.
// Internally, MAGMA transfers data to the GPU device
// and uses a hybrid CPU + GPU algorithm.
void cpu_interface(magma_int_t n, magma_int_t nrhs)       //CPU에서 Linear Solve를 하는 코드이다
{
    magmaDoubleComplex *A = NULL, *X = NULL;
    magma_int_t *ipiv = NULL;
    magma_int_t lda = n;
    magma_int_t ldx = lda;
    magma_int_t info = 0;               //초기화

    // magma_*malloc_cpu routines for CPU memory are type-safe and align to memory boundaries,
    // but you can use malloc or new if you prefer.
    magma_zmalloc_cpu(&A, lda * n);         //메모리 할당
    magma_zmalloc_cpu(&X, ldx * nrhs);
    magma_imalloc_cpu(&ipiv, n);
    if (A == NULL || X == NULL || ipiv == NULL) {
        fprintf(stderr, "malloc failed\n");
        goto cleanup;
    }

    // Replace these with your code to initialize A and X
    zfill_matrix(n, n, A, lda);           //A 행렬 만들어주기
    printf("A matrix on CPU  \n\n");
    magma_zprint(n, n, A, lda);    //A 행렬 출력해주는 코드

    zfill_rhs(n, nrhs, X,
              ldx);           //B를 초기화해준다 그런데 X라고 써있는 이유는 이게 B가 X로 바뀌기 때문이다. 나중에 나온다 GitHub에는 B라고 했는데 사실 X이다 이해 돕기 쉽게 하기 위해서
    printf("B matrix on CPU  \n\n");
    magma_zprint(n, nrhs, X, ldx); //B 행렬 출력해주는 코

    magma_zgesv(n, 1, A, lda, ipiv, X, lda,
                &info);       // 핵심코드 Z- Complex Number GE- General SV- Solve  Ax=B 중 X를 구하는 과정이다

    printf("A matrix on CPU after Linear Solve which is LU  \n\n"); //A와 X행렬이 어떻게 바뀌는지 보여주는 코드
    magma_zprint(n, n, A, lda);
    printf("B matrix on CPU after Linear Solve which is X \n\n");
    magma_zprint(n, nrhs, X, ldx);

    if (info != 0) {
        fprintf(stderr, "magma_zgesv failed with info=%d\n", info);
    }

    // TODO: use result in X

    cleanup:
    magma_free_cpu(A);
    magma_free_cpu(X);
    magma_free_cpu(ipiv);         // 메모리 동적 할당 해제
}


// ------------------------------------------------------------
// Solve dA * dX = dB, where dA and dX are stored in GPU device memory.
// Internally, MAGMA uses a hybrid CPU + GPU algorithm.
void gpu_interface(magma_int_t n, magma_int_t nrhs)           //GPU에서 일어나는 코드이다
{
    magmaDoubleComplex *dA = NULL, *dX = NULL;
    magma_int_t *ipiv = NULL;
    magma_int_t ldda = magma_roundup(n, 32);  // round up to multiple of 32 for best GPU performance
    magma_int_t lddx = ldda;
    magma_int_t info = 0;
    magma_queue_t queue = NULL;


    magmaDoubleComplex alpha = MAGMA_Z_MAKE(1, 0);   //GEMM 할때 필요한 scalar 이다
    magmaDoubleComplex beta = MAGMA_Z_MAKE(1, 0);

    // magma_*malloc routines for GPU memory are type-safe,
    // but you can use cudaMalloc if you prefer.
    magma_zmalloc(&dA, ldda * n);
    magma_zmalloc(&dX, lddx * nrhs);
    magma_imalloc_cpu(&ipiv, n);  // ipiv always on CPU
    if (dA == NULL || dX == NULL || ipiv == NULL) {
        fprintf(stderr, "malloc failed\n");
        goto cleanup;
    }

    magma_int_t dev = 0;
    magma_queue_create(dev, &queue);

    // Replace these with your code to initialize A and X
    zfill_matrix_gpu(n, n, dA, ldda, queue);
    printf("A matrix on GPU\n");
    magma_zprint_gpu(n, n, dA, ldda, queue);

    zfill_rhs_gpu(n, nrhs, dX, lddx, queue);              //CPU랑 비슷하다
    printf("B matrix on GPU\n");
    magma_zprint_gpu(n, nrhs, dX, lddx, queue);

    magma_zgesv_gpu(n, 1, dA, ldda, ipiv, dX, lddx, &info);       //GPU버전 magma_zgesv랑 비슷하다

    printf("A matrix after zgesv on GPU which is LU\n");
    magma_zprint_gpu(n, n, dA, ldda, queue);
    printf("B matrix after zgesv on GPU which is X matrix\n");
    magma_zprint_gpu(n, nrhs, dX, lddx, queue);

    int i;
    printf("Permutation Rank\n");   //치환 행렬을 나타내기 위한 코드이다 , 이를 통해 어떤 행이랑 어떤 행이 교환됬는지 알수있다
    for (i = 0; i < n; i++) {
        printf("%dth row  <-> %dth row \n", i + 1, ipiv[i]);
    }

    if (info != 0) {
        fprintf(stderr, "magma_zgesv_gpu failed with info=%d\n", info);
    }

    // TODO: use result in dX

    cleanup:
    magma_queue_destroy(queue);
    magma_free(dA);
    magma_free(dX);
    magma_free_cpu(ipiv);
}


// ------------------------------------------------------------
int main(int argc, char **argv) {
    magma_init();

    magma_int_t n = 1000;
    magma_int_t nrhs = 1;           //잘 만들었다  1000*1인 X 행렬을 구한다는 것이다 즉 1000*1000 A행렬 1000*1 B행렬을 뜻한다

    printf("using MAGMA CPU interface\n");
    cpu_interface(n, nrhs);        //CPU에서 A와 B 행렬을 만들고 X를 구한다고 보면 된다

    printf("using MAGMA GPU interface\n");
    gpu_interface(n, nrhs);       // GPU에서 A와 B 행렬을 만들고 X를 구한다고 보면 된다

    magma_finalize();               // Magma 그만 쓸게요
    return 0;
}
