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

void MatrixMultiplicationBaseline(const float* A, const float* B, float* C, uint32_t N) {
	for (uint32_t row = 0; row < N; row++)
		for (uint32_t col = 0; col < N; col++) {
			for (uint32_t idx = 0; idx < N; idx++)
				C[row * N + col] += A[row * N + idx] * B[idx * N + col];
		}
}

void MatrixMultiplicationParallel(const float* A, const float* B, float* C, uint32_t N, uint32_t start_row, uint32_t end_row) {
	for (uint32_t row = start_row; row < end_row; row++)
		for (uint32_t col = 0; col < N; col++) {
			for (uint32_t idx = 0; idx < N; idx++)
				C[row * N + col] += A[row * N + idx] * B[idx * N + col];
		}
}

void MatrixMultiplicationParallelSetup(const float* A, const float* B, float* C, uint32_t N, uint32_t num_threads) {
	const uint32_t threads = 12;
	const uint32_t rows_per_thread = N / threads;
	thread t[threads];

	for (uint32_t i = 0; i < threads; i++) {
		t[i] = thread(MatrixMultiplicationParallel, A, B, C, N, i * rows_per_thread, (i + 1) * rows_per_thread);
	}

	MatrixMultiplicationParallel(A, B, C, N, threads * rows_per_thread, N);

	for (uint32_t i = 0; i < threads; i++) {
		t[i].join();
	}
}

void MatrixMultiplicationBlocked(const float* A, const float* B, float* C, uint32_t N) {
	for (uint32_t row = 0; row < N; row++) {
		uint32_t block;
		for (block = 0; block < N; block += 16) {
			uint32_t chunk;
			for (chunk = 0; chunk < N; chunk += 16) {
				for (uint32_t sub_chunk = 0; sub_chunk < 16; sub_chunk++) {
					for (uint32_t idx = 0; idx < 16; idx++) {
						C[row * N + block + idx] += A[row * N + chunk + sub_chunk] * B[chunk * N + sub_chunk * N + block + idx];
					}
				}
			}
			for (uint32_t sub_chunk = 0; sub_chunk < chunk - N; sub_chunk++) {
				for (uint32_t idx = 0; idx < 16; idx++) {
					C[row * N + block + idx] += A[row * N + sub_chunk] * B[sub_chunk * N + block + idx];
				}
			}
		}
		uint32_t chunk;
		for (chunk = 0; chunk < N; chunk += 16) {
			for (uint32_t sub_chunk = 0; sub_chunk < 16; sub_chunk++) {
				for (uint32_t idx = 0; idx < block - N; idx++) {
					C[row * N + idx] += A[row * N + chunk + sub_chunk] * B[chunk * N + sub_chunk * N + idx];
				}
			}
		}
		for (uint32_t sub_chunk = 0; sub_chunk < chunk - N; sub_chunk++) {
			for (uint32_t idx = 0; idx < block - N; idx++) {
				C[row * N + idx] += A[row * N + sub_chunk] * B[sub_chunk * N + idx];
			}
		}
	}
}

void MatrixMultiplicationBlockedParallel(const float* A, const float* B, float* C, uint32_t N, uint32_t start_row, uint32_t end_row) {
	for (uint32_t row = start_row; row < end_row; row++) {
		uint32_t block;
		for (block = 0; block < N; block += 16) {
			uint32_t chunk;
			for (chunk = 0; chunk < N; chunk += 16) {
				for (uint32_t sub_chunk = 0; sub_chunk < 16; sub_chunk++) {
					for (uint32_t idx = 0; idx < 16; idx++) {
						C[row * N + block + idx] += A[row * N + chunk + sub_chunk] * B[chunk * N + sub_chunk * N + block + idx];
					}
				}
			}
			for (uint32_t sub_chunk = 0; sub_chunk < chunk - N; sub_chunk++) {
				for (uint32_t idx = 0; idx < 16; idx++) {
					C[row * N + block + idx] += A[row * N + sub_chunk] * B[sub_chunk * N + block + idx];
				}
			}
		}
		uint32_t chunk;
		for (chunk = 0; chunk < N; chunk += 16) {
			for (uint32_t sub_chunk = 0; sub_chunk < 16; sub_chunk++) {
				for (uint32_t idx = 0; idx < block - N; idx++) {
					C[row * N + idx] += A[row * N + chunk + sub_chunk] * B[chunk * N + sub_chunk * N + idx];
				}
			}
		}
		for (uint32_t sub_chunk = 0; sub_chunk < chunk - N; sub_chunk++) {
			for (uint32_t idx = 0; idx < block - N; idx++) {
				C[row * N + idx] += A[row * N + sub_chunk] * B[sub_chunk * N + idx];
			}
		}
	}
}

void MatrixMultiplicationBlockedParallelSetup(const float* A, const float* B, float* C, uint32_t N, uint32_t num_threads) {
	const uint32_t threads = 12;
	const uint32_t rows_per_thread = N / threads;
	thread t[threads];
	
	for (uint32_t i = 0; i < threads; i++) {
		t[i] = thread(MatrixMultiplicationBlockedParallel, A, B, C, N, i * rows_per_thread, (i + 1) * rows_per_thread);
	}
	
	MatrixMultiplicationBlockedParallel(A, B, C, N, threads * rows_per_thread, N);
	
	for (uint32_t i = 0; i < threads; i++) {
		t[i].join();
	}
}

void MatrixMultiplicationBlockedColumn(const float* A, const float* B, float* C, uint32_t N) {
	uint32_t col_chunk;
	for (col_chunk = 0; col_chunk < N; col_chunk += 16) {
		for (uint32_t row = 0; row < N; row++) {
			uint32_t tile;
			for (tile = 0; tile < N; tile += 16) {
				for (uint32_t tile_row = 0; tile_row < 16; tile_row++) {
					for (uint32_t idx = 0; idx < 16; idx++) {
						C[row * N + col_chunk + idx] += A[row * N + tile + tile_row] * B[tile * N + tile_row * N + col_chunk + idx];
					}
				}
			}
			for (uint32_t tile_row = 0; tile_row < tile - N; tile_row++) {
				for (uint32_t idx = 0; idx < 16; idx++) {
					C[row * N + col_chunk + idx] += A[row * N + tile_row] * B[tile_row * N + col_chunk + idx];
				}
			}
		}
	}
	uint32_t tile;
	for (tile = 0; tile < N; tile += 16) {
		for (uint32_t tile_row = 0; tile_row < 16; tile_row++) {
			for (uint32_t idx = 0; idx < col_chunk - N; idx++) {
				C[idx] += A[tile + tile_row] * B[tile * N + tile_row * N + idx];
			}
		}
	}
	for (uint32_t tile_row = 0; tile_row < tile - N; tile_row++) {
		for (uint32_t idx = 0; idx < col_chunk - N; idx++) {
			C[idx] += A[tile_row] * B[tile_row * N + idx];
		}
	}
}

void MatrixMultiplicationBlockedColumnParallel(const float* A, const float* B, float* C, uint32_t N, uint32_t start_row, uint32_t end_row) {
	uint32_t col_chunk;
	for (col_chunk = 0; col_chunk < N; col_chunk += 16) {
		for (uint32_t row = start_row; row < end_row; row++) {
			uint32_t tile;
			for (tile = 0; tile < N; tile += 16) {
				for (uint32_t tile_row = 0; tile_row < 16; tile_row++) {
					for (uint32_t idx = 0; idx < 16; idx++) {
						C[row * N + col_chunk + idx] += A[row * N + tile + tile_row] * B[tile * N + tile_row * N + col_chunk + idx];
					}
				}
			}
			for (uint32_t tile_row = 0; tile_row < tile - N; tile_row++) {
				for (uint32_t idx = 0; idx < 16; idx++) {
					C[row * N + col_chunk + idx] += A[row * N + tile_row] * B[tile_row * N + col_chunk + idx];
				}
			}
		}
	}
	uint32_t tile;
	for (tile = 0; tile < N; tile += 16) {
		for (uint32_t tile_row = 0; tile_row < 16; tile_row++) {
			for (uint32_t idx = 0; idx < col_chunk - N; idx++) {
				C[idx] += A[tile + tile_row] * B[tile * N + tile_row * N + idx];
			}
		}
	}
	for (uint32_t tile_row = 0; tile_row < tile - N; tile_row++) {
		for (uint32_t idx = 0; idx < col_chunk - N; idx++) {
			C[idx] += A[tile_row] * B[tile_row * N + idx];
		}
	}
}

void MatrixMultiplicationBlockedColumnParallelSetup(const float* A, const float* B, float* C, uint32_t N, uint32_t num_threads) {
	const uint32_t threads = 12;
	const uint32_t rows_per_thread = N / threads;
	thread t[threads];

	for (uint32_t i = 0; i < threads; i++) {
		t[i] = thread(MatrixMultiplicationBlockedColumnParallel, A, B, C, N, i * rows_per_thread, (i + 1) * rows_per_thread);
	}
	
	MatrixMultiplicationBlockedColumnParallel(A, B, C, N, threads * rows_per_thread, N);

	for (uint32_t i = 0; i < threads; i++) {
		t[i].join();
	}
}

int main() {
	const int batchSize = 1;
	int iter;
	Random random;

	int M = 1024;
	int N = 1024;
	int K = 1024;
	float* A = new float[M * K];
	float* B = new float[K * N];
	float* C = new float[M * N];
	float* D = new float[M * N];

	for (int i = 0; i < M * K; i++) {
		A[i] = random.normalRand();
	}
	for (int i = 0; i < K * N; i++) {
		B[i] = random.normalRand();
	}
	memset(C, 0, M * N * sizeof(float));
	iter = batchSize;
	auto start = high_resolution_clock::now();
	while (iter--) {
		MatrixMultiplicationBaseline(A, B, C, M);
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by naiveMatrixMultiplication: " 
		<< duration.count() * 1e-6 / batchSize << " seconds" << endl;
	memcpy(D, C, M * N * sizeof(float));
	
	
	memset(C, 0, M* N * sizeof(float));
	iter = batchSize;
	start = high_resolution_clock::now();
	while (iter--) {
		MatrixMultiplicationParallelSetup(A, B, C, M, 16);
	}
	end = high_resolution_clock::now();
	duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by naiveMultithreadedMatrixMultiplication: "
		<< duration.count() * 1e-6 / batchSize << " seconds" << endl;

	
	memset(C, 0, M * N * sizeof(float));
	iter = batchSize;
	start = high_resolution_clock::now();
	while (iter--) {
		MatrixMultiplicationBlocked(A, B, C, M);
	}
	end = high_resolution_clock::now();
	duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by blockedMatrixMultiplication: "
		<< duration.count() * 1e-6 / batchSize << " seconds" << endl;

	
	memset(C, 0, M * N * sizeof(float));
	iter = batchSize;
	start = high_resolution_clock::now();
	while (iter--) {
		MatrixMultiplicationBlockedParallelSetup(A, B, C, M, 16);
	}
	end = high_resolution_clock::now();
	duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by blockedMultithreadedMatrixMultiplication: "
		<< duration.count() * 1e-6 / batchSize << " seconds" << endl;
	
	memset(C, 0, M * N * sizeof(float));
	iter = batchSize;
	start = high_resolution_clock::now();
	while (iter--) {
		MatrixMultiplicationBlockedColumn(A, B, C, M);
	}
	end = high_resolution_clock::now();
	duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by blockedColumnMatrixMultiplication: "
		<< duration.count() * 1e-6 / batchSize << " seconds" << endl;

	memset(C, 0, M * N * sizeof(float));
	iter = batchSize;
	start = high_resolution_clock::now();
	while (iter--) {
		MatrixMultiplicationBlockedColumnParallelSetup(A, B, C, M, 16);
	}
	end = high_resolution_clock::now();
	duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by blockedColumnMultithreadedMatrixMultiplication: "
		<< duration.count() * 1e-6 / batchSize << " seconds" << endl;
	
	for (int i = 0; i < M * N; i++) {
		if (C[i] != D[i]) {
			cout << "Error" << endl;
			break;
		}
	}
	
	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}