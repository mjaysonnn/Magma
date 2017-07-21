#ifdef WIN32
#include <windows.h>
#else

#include <pthread.h>
#include <stdlib.h>

#ifdef ICC // The ICC is defined (by default) for enabling Intel Compiler specific headers and calls
#include <immintrin.h>
#endif
#endif

#include <stdio.h>
#include <time.h>
#include "omp.h"
//#include <malloc.h>

//#include "multiply.h" //multipy.h

#define MAXTHREADS 16
#define NUM 2048
#define MATRIX_BLOCK_SIZE 64

typedef unsigned long long UINT64;  //새로운 자료형 UINT 64
typedef double TYPE;        // 문자
typedef TYPE array[NUM];    // 문자열

double TRIP_COUNT = (double) NUM * (double) NUM * (double) NUM;  //뭘 갔다온다고 하네

int FLOP_PER_ITERATION = 2; // basic matrix multiplication

typedef struct tparam {
    array *a, *b, *c, *t;
    int msize;
    int tidx;
    int numt;
} _tparam;

extern int getCPUCount();

extern double getCPUFreq();

// routine to initialize an array with data
void init_arr(TYPE row, TYPE col, TYPE off, TYPE a[][NUM]) {
    int i, j;

    for (i = 0; i < NUM; i++) {
        for (j = 0; j < NUM; j++) {
            a[i][j] = row * i + col * j + off;
        }
    }
}

// routine to print out contents of small arrays
void print_arr(char *name, TYPE array[][NUM]) {
    int i, j;

    printf("\n%s\n", name);
    for (i = 0; i < NUM; i++) {
        for (j = 0; j < NUM; j++) {
            printf("%g\t", array[i][j]);
        }
        printf("\n");
        fflush(stdout);
    }
}

void GetModelParams(int *p_nthreads, int *p_msize, int print) {    // 0 0 1
    int msize = NUM;          //2048
    int nthr = MAXTHREADS;    // 16
    int ncpu = omp_get_max_threads();  //ncpu를 얻을수있나보구먼  10이라고 해보자  returns a number of threads available
    if (ncpu < nthr)       // 10 < 16
        nthr = ncpu;        //cpu에 있는 thread가 nthr 즉 10개이다
    omp_set_num_threads(nthr);   //thread를 8개만 구하겠다

    if (p_nthreads != 0)     //0 이 아니면 이걸로 해준다
        *p_nthreads = nthr;
    if (p_msize != 0)        // 0 이 아니면 이걸로 해준다
        *p_msize = msize;

    if (print) {
        printf("Threads #: %d %s\n", nthr, "OpenMP threads");
        fflush(stdout);
        printf("Matrix size: %d\n", msize);
        fflush(stdout);
        printf("Using multiply kernel: %s\n", xstr(MULTIPLY));
        fflush(stdout);
    }
}

void ParallelMultiply(int msize, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]) {
    int NTHREADS = MAXTHREADS;
    int MSIZE = NUM;

#ifdef WIN32
    HANDLE ht[MAXTHREADS];
    DWORD tid[MAXTHREADS];
#else
    pthread_t ht[MAXTHREADS];
    int tret[MAXTHREADS];
    int rc;
    void *status;
#endif
    _tparam par[MAXTHREADS];
    int tidx;

    GetModelParams(&NTHREADS, &MSIZE, 0);

    for (tidx = 0; tidx < NTHREADS; tidx++) {
        par[tidx].msize = MSIZE;
        par[tidx].numt = NTHREADS;
        par[tidx].tidx = tidx;
        par[tidx].a = a;
        par[tidx].b = b;
        par[tidx].c = c;
        par[tidx].t = t;
#ifdef WIN32
        ht[tidx] = (HANDLE)CreateThread(NULL, 0, ThreadFunction, &par[tidx], 0, &tid[tidx]);
#else
        tret[tidx] = pthread_create(&ht[tidx], NULL, (void *) ThreadFunction, (void *) &par[tidx]);
#endif
    }
#ifdef WIN32
    WaitForMultipleObjects(NTHREADS, ht, TRUE, INFINITE);
#else // Pthreads
    for (tidx = 0; tidx < NTHREADS; tidx++) {
        //  printf("Enter join\n"); fflush(stdout);
        rc = pthread_join(ht[tidx], (void **) &status);
        //  printf("Exit join\n"); fflush(stdout);
    }
#endif

}

int main() {
#ifdef WIN32  //윈도우로 컴파일 됬다는 뜻이네 이건 Linux이니
    clock_t start=0.0, stop=0.0;
#else // Pthreads
    double start = 0.0, stop = 0.0;
    struct timeval before, after;
#endif
    double secs;    //seconds
    double flops;   //Floating Operations
    double mflops;

    char *buf1, *buf2, *buf3, *buf4;   //문자열
    char *addr1, *addr2, *addr3, *addr4; //주소
    array *a, *b, *c, *t;  //array 는 double
    int Offset_Addr1 = 128, Offset_Addr2 = 192, Offset_Addr3 = 0, Offset_Addr4 = 64;
    //OffSet라는 의미에 맞게 상대 주소를 뜻할수 있다 아니면 변위치이니까

// malloc arrays space
// Define ALIGNED in the preprocessor
// Also add '/Oa' for Windows and '-fno-alias' for Linux
#ifdef ALIGNED

#ifdef WIN32
#ifdef ICC
    buf1 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
    buf2 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
    buf3 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
    buf4 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
#else
    buf1 = _aligned_malloc((sizeof (double))*NUM*NUM, 64);
    buf2 = _aligned_malloc((sizeof (double))*NUM*NUM, 64);
    buf3 = _aligned_malloc((sizeof (double))*NUM*NUM, 64);
    buf4 = _aligned_malloc((sizeof (double))*NUM*NUM, 64);
#endif //ICC
#else // WIN32
    buf1 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
    buf2 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
    buf3 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
    buf4 = _mm_malloc((sizeof (double))*NUM*NUM, 64);
#endif //WIN32
    addr1 = buf1;
    addr2 = buf2;
    addr3 = buf3;
    addr4 = buf4;

#else //!ALIGNED
    buf1 = (char *) malloc(NUM * NUM * (sizeof(double)) + 1024);
    printf("Addr of buf1 = %p\n", buf1);
    fflush(stdout);
    addr1 = buf1 + 256 - ((UINT64) buf1 % 256) + (UINT64) Offset_Addr1;
    printf("Offs of buf1 = %p\n", addr1);
    fflush(stdout);

    buf2 = (char *) malloc(NUM * NUM * (sizeof(double)) + 1024);
    printf("Addr of buf2 = %p\n", buf2);
    fflush(stdout);
    addr2 = buf2 + 256 - ((UINT64) buf2 % 256) + (UINT64) Offset_Addr2;
    printf("Offs of buf2 = %p\n", addr2);
    fflush(stdout);

    buf3 = (char *) malloc(NUM * NUM * (sizeof(double)) + 1024);
    printf("Addr of buf3 = %p\n", buf3);
    fflush(stdout);
    addr3 = buf3 + 256 - ((UINT64) buf3 % 256) + (UINT64) Offset_Addr3;
    printf("Offs of buf3 = %p\n", addr3);
    fflush(stdout);

    buf4 = (char *) malloc(NUM * NUM * (sizeof(double)) + 1024);
    printf("Addr of buf4 = %p\n", buf4);
    fflush(stdout);
    addr4 = buf4 + 256 - ((UINT64) buf4 % 256) + (UINT64) Offset_Addr4;
    printf("Offs of buf4 = %p\n", addr4);
    fflush(stdout);

#endif //ALIGNED

    a = (array *)
            addr1;
    b = (array *)
            addr2;
    c = (array *)
            addr3;
    t = (array *)
            addr4;

// initialize the arrays with data
    init_arr(3, -2, 1, a);
    init_arr(-2, 1, 3, b);

    // Printing model parameters
    GetModelParams(0, 0, 1);

// start timing the matrix multiply code
#ifdef WIN32
    start = clock();
#else


#ifdef ICC
    start = (double)_rdtsc();
#else
    gettimeofday(&before, NULL);
#endif
#endif

    ParallelMultiply(NUM, a, b, c, t);


#ifdef WIN32
    stop = clock();
    secs = ((double)(stop - start)) / CLOCKS_PER_SEC;
#else
#ifdef ICC
    stop = (double)_rdtsc();
    secs = ((double)(stop - start)) / (double) getCPUFreq();
#else
    gettimeofday(&after, NULL);
    secs = (after.tv_sec - before.tv_sec) + (after.tv_usec - before.tv_usec) / 1000000.0;
#endif
#endif


    flops = TRIP_COUNT * FLOP_PER_ITERATION;
    mflops = flops / 1000000.0f / secs;
    printf("Execution time = %2.3lf seconds\n", secs);
    fflush(stdout);
    //printf("MFLOPS: %2.3f mflops\n", mflops);

// print simple test case of data to be sure multiplication is correct
    if (NUM < 5) {
        print_arr("a", a);
        fflush(stdout);
        print_arr("b", b);
        fflush(stdout);
        print_arr("c", c);
        fflush(stdout);
    }

    //free memory
#ifdef ALIGNED
#ifdef WIN32
#ifdef ICC
    _mm_free(buf1);
    _mm_free(buf2);
    _mm_free(buf3);
    _mm_free(buf4);
#else
    _aligned_free(buf1);
    _aligned_free(buf2);
    _aligned_free(buf3);
    _aligned_free(buf4);
#endif //ICC
#else // ICC or GCC Linux
    _mm_free(buf1);
    _mm_free(buf2);
    _mm_free(buf3);
    _mm_free(buf4);
#endif //WIN32
#else //ALIGNED
    free(buf1);
    free(buf2);
    free(buf3);
    free(buf4);
#endif //ALIGNED
