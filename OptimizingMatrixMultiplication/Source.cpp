#include <iostream>
#include <chrono>
#include <thread>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::thread;

// writing mutliple matrix mutliplication algorithms to compare their performance
// 1. naive cpu matrix multiplication
/*void main(){
  define A, B, C
  for i = 0 to M do
	for j = 0 to N do
	  for k = 0 to K do
		C(i,j) <= C(i,j) + A(i,k) * B(k,j)
	  end
	end
  end
}*/

void naiveMatrixMultiplication(int M, int N, int K, float* A, float* B, float* C) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < K; k++) {
				C[i * N + j] += A[i * K + k] * B[k * N + j];
			}
		}
	}
}

// 2. naive multitheaded matrix multiplication
/*void main(){

	define A_cpu, B_cpu, C_cpu in the CPU memory
	define A_gpu, B_gpu, C_gpu in the GPU memory

	memcopy A_cpu to A_gpu
	memcopy B_cpu to B_gpu

	dim3 dimBlock(16, 16)
	dim3 dimGrid(N/dimBlock.x, M/dimBlock.y)

	matrixMul<<<dimGrid, dimBlock>>>(A_gpu,B_gpu,C_gpu,K)

	memcopy C_gpu to C_cpu

}
__global__ void matrixMul(A_gpu,B_gpu,C_gpu,K){

	temp <= 0

	i <= blockIdx.y * blockDim.y + threadIdx.y    // Row i of matrix C
	j <= blockIdx.x * blockDim.x + threadIdx.x    // Column j of matrix C

	for k = 0 to K-1 do
		accu <= accu + A_gpu(i,k) * B_gpu(k,j)
	end

	C_gpu(i,j) <= accu

}*/

void naiveMultithreadedMatrixMultiplication(int M, int N, int K, float* A, float* B, float* C, int i, int j) {
	for (int k = 0; k < K; k++) {
		C[i * N + j] += A[i * K + k] * B[k * N + j];
	}
}

void naiveMultithreadedMatrixMultiplicationSetup(int M, int N, int K, float* A, float* B, float* C) {
	const int threads = 8;
	thread processes[threads];
	int processesPerThread = (M * N) / threads;
	int processesPerThreadRemainder = M * N - threads * processesPerThread;

	int currentNode = 0;
	for (int i = 0; i < processesPerThreadRemainder; i++) {
		processes[i] = thread(naiveMultithreadedMatrixMultiplication, M, N, K, A, B, C, currentNode / N, currentNode % N);
		currentNode += processesPerThread + 1;
	}
}

int main() {
	const int batchSize = 100;
	int iter;

	int M = 100;
	int N = 100;
	int K = 100;
	float* A = new float[M * K];
	float* B = new float[K * N];
	float* C = new float[M * N];

	memset(A, 0.34, M * K * sizeof(float));
	memset(B, 1.23, K * N * sizeof(float));
	memset(C, 0, M * N * sizeof(float));

	iter = 600;
	while (iter--) {
		naiveMatrixMultiplication(M, N, K, A, B, C);
	}

	iter = batchSize;
	auto start = high_resolution_clock::now();
	while (iter--) {
		naiveMatrixMultiplication(M, N, K, A, B, C);
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by naiveMatrixMultiplication: " << duration.count() * 1e-6 / batchSize << " seconds" << endl;

	iter = batchSize;
	start = high_resolution_clock::now();
	while (iter--) {
		naiveMultithreadedMatrixMultiplicationSetup(M, N, K, A, B, C);
	}
	end = high_resolution_clock::now();
	duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by naiveMultithreadedMatrixMultiplication: " << duration.count() * 1e-6 / batchSize << " seconds" << endl;

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}