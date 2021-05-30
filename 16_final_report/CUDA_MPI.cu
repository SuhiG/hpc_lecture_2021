#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;
#define THREADS_X 8
#define THREADS_Y 1

__global__ void init_rand(int seed, curandState_t* states){
      int row = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, row, 0, &states[row]);

	}

__global__ void randoms(curandState_t *states, float *numbers){
        int row = blockIdx.x * blockDim.x + threadIdx.x;
	numbers[row] = curand_normal(&states[row]);

}


__global__ void subA_calc(float *subA, float *A, int N, int offset, int size){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    //float sum = 0.0;

    if(row < N/size){
        for(int j = 0; j < N; j++){
            subA[N * row + j] = A[N * (offset + row) + j];
        }
    }


}

__global__ void subB_calc(float *subB, float *B, int N, int offset, int size){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    //float sum = 0.0;

    if(row < N){
        for(int j = 0; j < N/size; j++){
            subB[N/size * row + j] = B[N * row + offset + j];
        }
    }


}__global__ void subC_calc(float *subA, float *subB, float *subC, int N, int offset, int size){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if(row < N/size){
        for(int j = 0; j < N/size; j++){
            sum = subC[N * row + j + offset];
            for(int k = 0; k < N; k++){
                sum += subA[N * row + k] * subB[N / size * k + j];
            }
            subC[N * row + j + offset] = sum;

        }

    }

}


int main(int argc, char** argv) {
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 256;
  /*vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  vector<float> subA(N*N/size);
  vector<float> subB(N*N/size);
  vector<float> subC(N*N/size, 0);
 */
// cudaGetDeviceCount(&gpusize);
 //edited
// #pragma omp parallel for

float *A, *B, *C, *subA, *subB, *subC;
//float *h_Arand, *h_Brand;
float *d_A, *d_B, *d_C, *d_subA, *d_subB, *d_subC;
//float *d_Arand, *d_Brand;

int K = 1;

int grid_x = ceil((N + THREADS_X - 1) / THREADS_X);
int grid_y = ceil((K + THREADS_Y - 1) / THREADS_Y);

dim3 dimGrid(grid_x, grid_y);
dim3 dimBlock(THREADS_X, THREADS_Y);


cudaMallocHost((void**)&A, sizeof(float) * N * N);
cudaMallocHost((void**)&B, sizeof(float) * N * N);
cudaMallocHost((void**)&C, sizeof(float) * N * N);
cudaMallocHost((void**)&subA, sizeof(float) * N * N/size);
cudaMallocHost((void**)&subB, sizeof(float) * N * N/size);
cudaMallocHost((void**)&subC, sizeof(float) * N * N/size);

/*cudaMallocHost((void**)&h_Arand, sizeof(float) * N * N);
cudaMalloc((void**)&d_Arand, sizeof(float) * N * N);
cudaMallocHost((void**)&h_Brand, sizeof(float) * N * N);
cudaMalloc((void**)&d_Brand, sizeof(float) * N * N);


curandState_t *states;
cudaMalloc((void**)&states, sizeof(curandState_t) * N * N);

init_rand<<<dimGrid, dimBlock>>>(1, states);

randoms<<<dimGrid,dimBlock>>>(states, d_Arand);
cudaMemcpy(h_Arand, d_Arand, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();


randoms<<<dimGrid,dimBlock>>>(states, d_Brand);
cudaMemcpy(h_Brand, d_Brand, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
*/
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      //A[N*i+j] = h_Arand[N*i+j];
	//printf("%f\n", A[N*i+j]);
      B[N*i+j] = drand48();
      //B[N*i+j] = h_Brand[N*i+j];
    }
  }
  int offset = N/size*rank;

cudaMalloc((void**)&d_A, sizeof(float) * N * N);
cudaMalloc((void**)&d_B, sizeof(float) * N * N);
cudaMalloc((void**)&d_C, sizeof(float) * N * N);
cudaMalloc((void**)&d_subA, sizeof(float) * N * N/size);
cudaMalloc((void**)&d_subB, sizeof(float) * N * N/size);
cudaMalloc((void**)&d_subC, sizeof(float) * N * N/size);

cudaMemcpy(d_A, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);
cudaMemcpy(d_C, C, sizeof(float) * N * N, cudaMemcpyHostToDevice);
cudaMemcpy(d_subA, subA, sizeof(float) * N * N/size, cudaMemcpyHostToDevice);
cudaMemcpy(d_subB, subB, sizeof(float) * N * N/size, cudaMemcpyHostToDevice);
cudaMemcpy(d_subC, subC, sizeof(float) * N * N/size, cudaMemcpyHostToDevice);
 
/*int K = 1;

int grid_x = ceil((N + THREADS_X - 1) / THREADS_X);
int grid_y = ceil((K + THREADS_Y - 1) / THREADS_Y);

dim3 dimGrid(grid_x, grid_y);
dim3 dimBlock(THREADS_X, THREADS_Y);
*/
 /* for (int i=0; i<N/size; i++){
    for (int j=0; j<N; j++){
      subA[N*i+j] = A[N*(i+offset)+j];
   }
}*/

subA_calc<<<dimGrid, dimBlock>>>(d_subA, d_A, N, offset,size);
cudaMemcpy(subA, d_subA, sizeof(float) * N * N/size, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();


/*  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];
*/

subB_calc<<<dimGrid, dimBlock>>>(d_subB, d_B, N, offset, size);
cudaMemcpy(subB, d_subB, sizeof(float) * N * N/size, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();


  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;

  double comp_time = 0, comm_time = 0;

  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();
    offset = N/size*((rank+irank) % size);

   /* for (int i=0; i<N/size; i++)
      for (int j=0; j<N/size; j++)
        for (int k=0; k<N; k++)
          subC[N*i+j+offset] += subA[N*i+k] * subB[N/size*k+j];
   */

	subC_calc<<<dimGrid, dimBlock>>>(d_subA, d_subB, d_subC, N, offset, size);
	cudaMemcpy(subC, d_subC, sizeof(float) * N * N/size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();

    MPI_Request request[2];
    MPI_Isend(&subB[0], N*N/size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&subB[0], N*N/size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Waitall(2, request, MPI_STATUS_IGNORE);

    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();

  }

  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];

  double err = 0;

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)

      err += fabs(C[N*i+j]);

  if(rank==0) {

    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  MPI_Finalize();

 cudaFreeHost(A);
 cudaFreeHost(B);
 cudaFreeHost(C);
 cudaFreeHost(subA);
 cudaFreeHost(subB);
 cudaFreeHost(subC);

 cudaFree(d_A);
 cudaFree(d_B);
 cudaFree(d_C);
 cudaFree(d_subA);
 cudaFree(d_subB);
 cudaFree(d_subC);

}
