#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define T 10000
#define N 2048
#define blk_cnt 32
#define thrd_per_blk N / 32
__global__ void AcopytoB(double A[], double B[])
{
    int my_element = blockDim.x * blockIdx.x + threadIdx.x;
    if (my_element > 0 && my_element < N)
        B[my_element] = A[my_element];
}
__global__ void avg(double A[], double B[])
{
    int my_element = blockDim.x * blockIdx.x + threadIdx.x;
    if (my_element > 0 && my_element < N - 1)
        A[my_element] = (B[my_element - 1] + B[my_element + 1]) / 2.0;
}
int main()
{
    double *A, *B;

    double sigma;
    int i;
    double cpu_time;
    clock_t begin, end;
        cudaMallocManaged(&A, N * sizeof(double));
    cudaMallocManaged(&B, N * sizeof(double));
    A[0] = 0;
    A[N - 1] = 0;
    for (i = 1; i < N - 1; i++)
        A[i] = 100.0;
    begin = clock();
    int t = 0;

    while (t++ < T)
    {
        AcopytoB<<<blk_cnt, thrd_per_blk>>>(A, B);
        cudaDeviceSynchronize();
        avg<<<blk_cnt, thrd_per_blk>>>(A, B);
        cudaDeviceSynchronize();
    }
    end = clock();
    cpu_time = (double)(end - begin) / CLOCKS_PER_SEC;
    for (i = 0; i < N; i++)
        sigma = sigma + A[i];
    printf("sigma = %.4f\n", sigma);
    printf("cpu time = %.4f\n", cpu_time);
    return 0;
}