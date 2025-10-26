#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void fill_n(int *m, int n , int val) {
    for (int i = 0; i < n; i++)
        m[i] = val;
}   

void Parfill_n(int *m, int n , int val) {
    #pragma omp for nowait
        for (int i = 0; i < n; i++)
            m[i] = val;
}

int pos(int *m, int i, int j, int cols) {
    // Row major
    return m[i * cols + j];
}

int areMatricesEqual(int *a, int *b, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (pos(a, i, j, m) != pos(b, i, j, m))
    return 0;
    }          
  }
  return 1;
}

int *allocate(int rows, int cols) {
    return (int *) malloc(rows * cols * sizeof(int));
}

// SEQUENCIAL COM CACHE
void MatMulCache(int* c, int* a, int* b, int n, int m, int p) {

  fill_n(c, n  * p, 0);

  for (int i = 0; i < n; i++){
    for (int k = 0; k <  m; k++){
      for (int j = 0; j < p; j++){
	    c[i*p + j] += pos(a, i, k, m) * pos(b, k, j, p);
      }
    }
  }
}

// PARALELO COM CACHE
void MatMulCacheOpenMP(int* c, int* a, int* b, int n, int m, int p) {
    #pragma omp parallel
    {
        Parfill_n(c, n * p, 0);
        #pragma omp for
        for (int i = 0; i < n; i++) {  
            for (int k = 0; k < m; k++) {
                for (int j = 0; j < p; j++) {
                    c[i*p + j] += pos(a, i, k, m) * pos(b, k, j, p);
                }
            }
        }
    }
}

//PARALELO 1D
void MatMul1D(int* c, int* a, int* b, int n, int m, int p){
  #pragma omp parallel
  {
    Parfill_n(c, n * p, 0); 
    #pragma omp for
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
        for (int k = 0; k < m; k++) {
          c[i*p + j] += pos(a, i, k, m) * pos(b, k, j, p);
        }
      }
    }
  }
}

//PARALELO 2D
#define BLOCK 64
void MatMul2D(int* c, int* a, int* b, int N){
 #pragma omp parallel
  {
    Parfill_n(c, N * N, 0); 
    #pragma omp for collapse(2) schedule(dynamic)
    for (int i2 = 0; i2 < N; i2 += BLOCK)
      for (int j2 = 0; j2 < N; j2 += BLOCK)
        for (int k2 = 0; k2 < N; k2 += BLOCK)
          for (int i = i2; i < i2 + BLOCK && i < N; i++)
            for (int j = j2; j < j2 + BLOCK && j < N; j++)
              for (int k = k2; k < k2 + BLOCK && k < N; k++)
                c[i*N + j] += a[i*N + k] * b[k*N + j];
    }
  }

int main(void){
  int N;
  int *a, *b, *c, *r;
  double ts;

  printf("Digite o tamanho da matriz: ");
  scanf("%d", &N);
  
  printf("Tamanho da matriz: %d  Memoria usada: %.2f MB\n",
         N, N * N * sizeof(int) / 1e6);
  
  a = allocate(N, N);
  b = allocate(N, N);
  c = allocate(N, N);
  r = allocate(N, N);

  // Inicializa matrizes
  for (int i = 0; i < N*N; i++) {
    a[i] = rand() % 100; 
    b[i] = rand() % 100;
  }
  // Sequencial com cache
  ts = omp_get_wtime();
  MatMulCache(r, a, b, N, N, N);
  printf("Tempo Sequencial com Cache: %f s\n", omp_get_wtime() - ts);

  // Paralelo com cache
  ts = omp_get_wtime();
  MatMulCacheOpenMP(c, a, b, N, N, N);
  if (areMatricesEqual(c, r, N, N))
    printf("Tempo Paralelo com Cache: %f s\n", omp_get_wtime() - ts);

  //1D
  ts = omp_get_wtime();
  MatMul1D(c, a, b, N, N, N);
  if (areMatricesEqual(c, r, N, N))
    printf("Tempo Paralelo 1D: %f s\n", omp_get_wtime() - ts);

  //2D
  ts = omp_get_wtime();
  MatMul2D(c, a, b, N);
  if (areMatricesEqual(c, r, N, N))
    printf("Tempo Paralelo 2D: %f s\n", omp_get_wtime() - ts);

  free(a);
  free(b);
  free(c);
  free(r);

  return 0;
}
