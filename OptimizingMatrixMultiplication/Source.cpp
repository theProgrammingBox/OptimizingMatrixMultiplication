#include <iostream>
#include <chrono>
#include <thread>
#include "Randoms.h"

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

void naiveMultithreadedMatrixMultiplication(int M, int N, int K, float* A, float* B, float* C, int row, int column, int nodes) {
	for (int i = 0; i < nodes; i++) {
		for (int k = 0; k < K; k++) {
			C[row * N + column] += A[row * K + k] * B[k * N + column];
		}
		row++;
		column += row >= M;
		row *= row < M;
	}
}

void naiveMultithreadedMatrixMultiplicationSetup(int M, int N, int K, float* A, float* B, float* C) {
	const int threads = 12;
	thread processes[threads];
	int processesPerThread = (M * N) / threads;
	int processesPerThreadRemainder = (M * N) - threads * processesPerThread;

	int row = 0;
	int column = 0;
	int columnsPerThread1 = (processesPerThread + 1) / M;
	int rowsPerThread1 = (processesPerThread + 1) - columnsPerThread1 * M;
	int columnsPerThread2 = processesPerThread / M;
	int rowsPerThread2 = processesPerThread - columnsPerThread2 * M;
	
	for (int i = 0; i < processesPerThreadRemainder; i++) {
		processes[i] = thread(naiveMultithreadedMatrixMultiplication, M, N, K, A, B, C, row, column, processesPerThread + 1);
		row += rowsPerThread1;
		column += columnsPerThread1 + (row >= M);
		row -= M * (row >= M);
	}
	for (int i = processesPerThreadRemainder; i < threads; i++) {
		processes[i] = thread(naiveMultithreadedMatrixMultiplication, M, N, K, A, B, C, row, column, processesPerThread);
		row += rowsPerThread2;
		column += columnsPerThread2 + (row >= M);
		row -= M * (row >= M);
	}
	for (int i = 0; i < threads; i++) {
		processes[i].join();
	}
}

void naiveMultithreadedMatrixMultiplication2(int M, int N, int K, float* A, float* B, float* C, int startNode, int nodes) {
	for (int i = startNode; i < startNode + nodes; i++) {
		int row = i / N;
		int column = i % N;
		for (int k = 0; k < K; k++) {
			C[i] += A[row * K + k] * B[k * N + column];
		}
	}
}

void naiveMultithreadedMatrixMultiplicationSetup2(int M, int N, int K, float* A, float* B, float* C) {
	const int threads = 8;
	thread processes[threads];
	int processesPerThread = (M * N) / threads;
	int processesPerThreadRemainder = M * N - threads * processesPerThread;

	int currentNode = 0;
	for (int i = 0; i < processesPerThreadRemainder; i++) {
		processes[i] = thread(naiveMultithreadedMatrixMultiplication2, M, N, K, A, B, C, currentNode, processesPerThread + 1);
		currentNode += processesPerThread + 1;
	}
	for (int i = processesPerThreadRemainder; i < threads; i++) {
		processes[i] = thread(naiveMultithreadedMatrixMultiplication2, M, N, K, A, B, C, currentNode, processesPerThread);
		currentNode += processesPerThread;
	}
	for (int i = 0; i < threads; i++) {
		processes[i].join();
	}
}

int main() {
	const int batchSize = 1;
	int iter;
	Random random;

	int M = 1000;
	int N = 1000;
	int K = 1000;
	float* A = new float[M * K];
	float* B = new float[K * N];
	float* C = new float[M * N];
	float* D = new float[M * N];

	for (int i = 0; i < M * K; i++) {
		A[i] = random.normalRand();
		B[i] = random.normalRand();
	}
	memset(C, 0, M * N * sizeof(float));

	iter = batchSize;
	auto start = high_resolution_clock::now();
	while (iter--) {
		naiveMatrixMultiplication(M, N, K, A, B, C);
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by naiveMatrixMultiplication: " << duration.count() * 1e-6 / batchSize << " seconds" << endl;
	
	/*for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout << C[i * N + j] << " ";
		}
		cout << endl;
	}
	cout << endl;*/

	memset(C, 0, M * N * sizeof(float));
	iter = batchSize;
	start = high_resolution_clock::now();
	while (iter--) {
		naiveMultithreadedMatrixMultiplicationSetup2(M, N, K, A, B, C);
	}
	end = high_resolution_clock::now();
	duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by naiveMultithreadedMatrixMultiplication2: " << duration.count() * 1e-6 / batchSize << " seconds" << endl;

	memcpy(D, C, M * N * sizeof(float));
	memset(C, 0, M * N * sizeof(float));
	iter = batchSize;
	start = high_resolution_clock::now();
	while (iter--) {
		naiveMultithreadedMatrixMultiplicationSetup(M, N, K, A, B, C);
	}
	end = high_resolution_clock::now();
	duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by naiveMultithreadedMatrixMultiplication: " << duration.count() * 1e-6 / batchSize << " seconds" << endl;
	
	/*for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout << C[i * N + j] << " ";
		}
		cout << endl;
	}
	cout << endl;*/
	
	/*for (int i = 0; i < M * N; i++) {
		if (C[i] != D[i]) {
			cout << "Error" << endl;
			break;
		}
	}*/
	
	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}