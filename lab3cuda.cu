#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#define T 10000
#define N 512
#define M 512

__global__ void matrix_add(double **a, double **b)
{
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= 0 && row < N && col >= 0 && col < M)
    b[row][col] = a[row][col] ;
}

__global__ void matrix_avg(double **a, double **b)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= 1 && i < N-1 && j >= 1 && j < M-1)
    a[i][j] = (b[i - 1][j] + b[i + 1][j] + b[i][j - 1] + b[i][j + 1]) / 4.0;
}

int main()
{
  double **A, **B;
  double sigma;
  int i, j;
  double cpu_time;
  clock_t begin, end;
  cudaMallocManaged(&A, N * sizeof(double *));
  cudaMallocManaged(&B, N * sizeof(double *));
  for (i = 0; i < N; i++)
  {
    cudaMallocManaged(&A[i], M * sizeof(double));
    cudaMallocManaged(&B[i], M * sizeof(double));
  }

  dim3 grid_dims, block_dims;
  grid_dims.x = 16;  // number of blocks in the x-dimension
  grid_dims.y = 16;  // number of blocks in the x-dimension
  block_dims.x = 32; // number of threads in the x-dimension
  block_dims.y = 32; // number of threads in the y-dimension

  
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
    {
      if (i == 0 || j == 0 || i == N - 1 || j == M - 1)
        A[i][j] = 0.0;
      else
        A[i][j] = 100.0;
    }

  begin = clock();
  int t = 0;
  while (t++ < T)
  {
  matrix_add <<<grid_dims, block_dims>>> (A, B);
  cudaDeviceSynchronize();

  matrix_avg <<<grid_dims, block_dims>>> (A, B);
  cudaDeviceSynchronize();


  }
  end = clock();
  cpu_time = (double)(end - begin) / CLOCKS_PER_SEC;

  sigma = 0.0;
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      sigma = sigma + A[i][j];

  printf("sigma = %.4f\n", sigma);
  printf("cpu time = %.4f\n", cpu_time);
   for(i=0; i<N; i++){
    cudaFree(A[i]); cudaFree(B[i]); ;
  }
  cudaFree(A); cudaFree(B); 
  
  return 0;
}