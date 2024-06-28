#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define T 10000
#define N 128
#define M 64


int main(){
  double A[N][M], B[N][M];
  double sigma;
  int i, j;
  
  double cpu_time;
  double begin, end;

  for(i=0; i<N; i++)
    for(j=0; j<M; j++){
        if(i==0 || j==0 || i==N-1 || j==M-1) 
           A[i][j] = 0.0;
        else   
           A[i][j] = 100.0;
    }

   begin = omp_get_wtime();
  #pragma omp parallel num_threads(8)
  {
  int t = 0;
  int i =0;
  int j=0;
  while(t++ < T){

    #pragma omp for
    for(i=0; i<N; i++)
      for(j=0; j<M; j++)
         B[i][j] = A[i][j];  
  #pragma omp for
    for(i=1; i<N-1; i++)
      for(j=1; j<M-1; j++)
         A[i][j] = (B[i-1][j] + B[i+1][j] + B[i][j-1] + B[i][j+1]) / 4.0;

  }
  }
  end = omp_get_wtime();
  cpu_time = end-begin;
  sigma = 0.0;
  for(int i=0; i<N; i++)
     for(j=0; j<M; j++)
         sigma = sigma + A[i][j];

  printf("sigma = %.4f\n",sigma);
  printf("cpu time = %.4f\n", cpu_time);
  
  return 0;
}

