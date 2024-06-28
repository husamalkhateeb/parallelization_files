#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 1024
#define M 1024

__global__ void parallel_init(int **A)
{
  A[blockIdx.x][threadIdx.x] = (int)(0.5 * blockIdx.x + 0.5 * threadIdx.x + 1);
 
}

__global__ void parallel_sum(int **A, int *sum)
{
  int offset = blockDim.x;
  for (int d = offset; d >= 1; d = d / 2)
  {
    if (threadIdx.x < d)
      A[blockIdx.x][threadIdx.x] += A[blockIdx.x][threadIdx.x + d];
    __syncthreads();
  }

  if (threadIdx.x == 0)
    atomicAdd(sum,A[blockIdx.x][0]);
}

int main()
{
  int **A;
  int i;
  int * sum = 0;

 cudaMallocManaged(&A,N * sizeof(int *));
  cudaMallocManaged(&sum, 1*sizeof(int));
  for (i = 0; i < N; i++)
 cudaMallocManaged(&A[i],M * sizeof(int));

  parallel_init<<<N, M>>>(A);
  cudaDeviceSynchronize();

  parallel_sum<<<N, M / 2>>>(A, sum);
  cudaDeviceSynchronize();
  printf("sum = %d\n", *sum);
  for(i=0; i<N; i++){
    cudaFree(A[i]);  }
  
  cudaFree(A);
  cudaFree(sum);
  return 0;
}