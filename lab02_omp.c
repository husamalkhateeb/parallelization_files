#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define T 10000
#define N 2048

int main(){
  double A[N], B[N];
  double sigma;
  int i;
  double cpu_time;
  double begin, end;

  A[0] = 0;
  A[N-1] = 0;
  for(i=1; i<N-1; i++)
    A[i] = 100.0;

  begin = omp_get_wtime();
  omp_set_num_threads(8);
#pragma omp parallel 
{
        int t = 0;
  while(t++ < T){
  
    #pragma omp for
    for(i=1; i<N-1; i++)
      B[i] = A[i];
    #pragma omp for
    for(i=1; i<N-1; i++)
      A[i] = (B[i - 1] + B[i + 1])/2.0;
    }
  }
  end = omp_get_wtime();
  cpu_time = end-begin;

  for(i=0; i<N; i++)
    sigma = sigma + A[i];

  printf("sigma = %.4f\n",sigma);
  printf("cpu time = %f\n", cpu_time);
  return 0;
}

