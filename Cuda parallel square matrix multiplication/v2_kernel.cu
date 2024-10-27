#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <time.h>
#include <vector>
#include <fstream>
#include <algorithm>

#define BLOCK_SIZE 64
#define STRASSEN_THRESHOLD 64  // Threshold for switching to normal multiplication

// Existing kernel code for ordinary matrix multiplication
__global__ static void matrixMultiplyKernel(float* A, float* B, float* C, int ARows, int ACols, int BCols) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int m = 0; m < (ACols - 1) / BLOCK_SIZE + 1; ++m) {
        if (row < ARows && m * BLOCK_SIZE + tx < ACols)
            As[ty][tx] = A[row * ACols + m * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < BCols && m * BLOCK_SIZE + ty < ACols)
            Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * BCols + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < ARows && col < BCols)
        C[row * BCols + col] = sum;
}

// Auxiliary functions for the Strassen algorithm
__global__ void matrixAdd(float* A, float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        C[index] = A[index] + B[index];
    }
}

__global__ void matrixSubtract(float* A, float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        C[index] = A[index] - B[index];
    }
}

class GPUInfo {
public:
    void printGPUInfo() {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        printf("GPU Information:\n");
        printf("Name: %s\n", deviceProp.name);
        printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("Global memory: %zu MB\n", deviceProp.totalGlobalMem / (1024 * 1024));
        printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max block dimensions: %d x %d x %d\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Max grid dimensions: %d x %d x %d\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("\n");
    }
};

class MatrixOperations {
public:
    void generateRandomMatrix(float* matrix, int rows, int cols) {
        for (int i = 0; i < rows * cols; ++i) {
            matrix[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    void matrixMultiplyCPU(float* A, float* B, float* C, int ARows, int ACols, int BCols) {
        for (int i = 0; i < ARows; ++i) {
            for (int j = 0; j < BCols; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < ACols; ++k) {
                    sum += A[i * ACols + k] * B[k * BCols + j];
                }
                C[i * BCols + j] = sum;
            }
        }
    }

    // CPU version of Strassen's algorithm
    void strassenMultiplyCPU(float* A, float* B, float* C, int n) {
        if (n <= STRASSEN_THRESHOLD) {
            matrixMultiplyCPU(A, B, C, n, n, n);
            return;
        }

        int newSize = n / 2;
        int size = newSize * newSize;

        float* A11 = new float[size];
        float* A12 = new float[size];
        float* A21 = new float[size];
        float* A22 = new float[size];
        float* B11 = new float[size];
        float* B12 = new float[size];
        float* B21 = new float[size];
        float* B22 = new float[size];

        // Separation of matrices by submatrix
        for (int i = 0; i < newSize; i++) {
            for (int j = 0; j < newSize; j++) {
                A11[i * newSize + j] = A[i * n + j];
                A12[i * newSize + j] = A[i * n + j + newSize];
                A21[i * newSize + j] = A[(i + newSize) * n + j];
                A22[i * newSize + j] = A[(i + newSize) * n + j + newSize];

                B11[i * newSize + j] = B[i * n + j];
                B12[i * newSize + j] = B[i * n + j + newSize];
                B21[i * newSize + j] = B[(i + newSize) * n + j];
                B22[i * newSize + j] = B[(i + newSize) * n + j + newSize];
            }
        }

        float* P1 = new float[size];
        float* P2 = new float[size];
        float* P3 = new float[size];
        float* P4 = new float[size];
        float* P5 = new float[size];
        float* P6 = new float[size];
        float* P7 = new float[size];

        float* temp1 = new float[size];
        float* temp2 = new float[size];

        // P1 = A11 * (B12 - B22)
        for (int i = 0; i < size; i++) temp1[i] = B12[i] - B22[i];
        strassenMultiplyCPU(A11, temp1, P1, newSize);

        // P2 = (A11 + A12) * B22
        for (int i = 0; i < size; i++) temp1[i] = A11[i] + A12[i];
        strassenMultiplyCPU(temp1, B22, P2, newSize);

        // P3 = (A21 + A22) * B11
        for (int i = 0; i < size; i++) temp1[i] = A21[i] + A22[i];
        strassenMultiplyCPU(temp1, B11, P3, newSize);

        // P4 = A22 * (B21 - B11)
        for (int i = 0; i < size; i++) temp1[i] = B21[i] - B11[i];
        strassenMultiplyCPU(A22, temp1, P4, newSize);

        // P5 = (A11 + A22) * (B11 + B22)
        for (int i = 0; i < size; i++) {
            temp1[i] = A11[i] + A22[i];
            temp2[i] = B11[i] + B22[i];
        }
        strassenMultiplyCPU(temp1, temp2, P5, newSize);

        // P6 = (A12 - A22) * (B21 + B22)
        for (int i = 0; i < size; i++) {
            temp1[i] = A12[i] - A22[i];
            temp2[i] = B21[i] + B22[i];
        }
        strassenMultiplyCPU(temp1, temp2, P6, newSize);

        // P7 = (A11 - A21) * (B11 + B12)
        for (int i = 0; i < size; i++) {
            temp1[i] = A11[i] - A21[i];
            temp2[i] = B11[i] + B12[i];
        }
        strassenMultiplyCPU(temp1, temp2, P7, newSize);

        // Calculation of the resulting submatrices
        float* C11 = new float[size];
        float* C12 = new float[size];
        float* C21 = new float[size];
        float* C22 = new float[size];

        // C11 = P5 + P4 - P2 + P6
        for (int i = 0; i < size; i++) {
            C11[i] = P5[i] + P4[i] - P2[i] + P6[i];
        }

        // C12 = P1 + P2
        for (int i = 0; i < size; i++) {
            C12[i] = P1[i] + P2[i];
        }

        // C21 = P3 + P4
        for (int i = 0; i < size; i++) {
            C21[i] = P3[i] + P4[i];
        }

        // C22 = P5 + P1 - P3 - P7
        for (int i = 0; i < size; i++) {
            C22[i] = P5[i] + P1[i] - P3[i] - P7[i];
        }

        // Combining the results into one matrix
        for (int i = 0; i < newSize; i++) {
            for (int j = 0; j < newSize; j++) {
                C[i * n + j] = C11[i * newSize + j];
                C[i * n + j + newSize] = C12[i * newSize + j];
                C[(i + newSize) * n + j] = C21[i * newSize + j];
                C[(i + newSize) * n + j + newSize] = C22[i * newSize + j];
            }
        }

        // Cleaning memory
        delete[] C11;
        delete[] C12;
        delete[] C21;
        delete[] C22;

        for (int i = 0; i < newSize; i++) {
            for (int j = 0; j < newSize; j++) {
                C[i * n + j] = P5[i * newSize + j] + P4[i * newSize + j] -
                    P2[i * newSize + j] + P6[i * newSize + j];

                C[i * n + j + newSize] = P1[i * newSize + j] + P2[i * newSize + j];

                C[(i + newSize) * n + j] = P3[i * newSize + j] + P4[i * newSize + j];

                C[(i + newSize) * n + j + newSize] = P5[i * newSize + j] + P1[i * newSize + j] -
                    P3[i * newSize + j] - P7[i * newSize + j];
            }
        }

        // Cleaning memory
        delete[] A11; delete[] A12; delete[] A21; delete[] A22;
        delete[] B11; delete[] B12; delete[] B21; delete[] B22;
        delete[] P1; delete[] P2; delete[] P3; delete[] P4; delete[] P5; delete[] P6; delete[] P7;
        delete[] temp1; delete[] temp2;
    }
};

class MatrixMultiplicationTest {
public:
    void testLargeMatrixMultiplication() {
        MatrixOperations matrixOps;
        // The dimensions of the matrices must be a power of 2 for Strassen's algorithm
        std::vector<int> sizes = { 64, 128, 256, 512, 1024 };
        std::ofstream outFile("matrix_multiplication_results.csv");
        outFile << "Size,Standard GPU Time (ms),Standard CPU Time (ms),Strassen CPU Time (ms),GPU/CPU Speedup,Strassen/Standard CPU Speedup\n";

        for (int size : sizes) {
            printf("\nTesting matrix multiplication of size %dx%d\n", size, size);

            size_t matrixSize = size * size * sizeof(float);
            float* h_A = (float*)malloc(matrixSize);
            float* h_B = (float*)malloc(matrixSize);
            float* h_C = (float*)malloc(matrixSize);
            float* h_C_cpu = (float*)malloc(matrixSize);
            float* h_C_strassen = (float*)malloc(matrixSize);

            matrixOps.generateRandomMatrix(h_A, size, size);
            matrixOps.generateRandomMatrix(h_B, size, size);

            // GPU standard multiplication
            float* d_A, * d_B, * d_C;
            cudaMalloc(&d_A, matrixSize);
            cudaMalloc(&d_B, matrixSize);
            cudaMalloc(&d_C, matrixSize);

            cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

            dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            matrixMultiplyKernel << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, size, size, size);
            cudaEventRecord(stop);

            cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

            cudaEventSynchronize(stop);
            float gpu_milliseconds = 0;
            cudaEventElapsedTime(&gpu_milliseconds, start, stop);

            // CPU standard multiplication
            clock_t cpu_start = clock();
            matrixOps.matrixMultiplyCPU(h_A, h_B, h_C_cpu, size, size, size);
            clock_t cpu_end = clock();
            double cpu_milliseconds = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

            // CPU multiplication by the Strassen method
            clock_t strassen_start = clock();
            matrixOps.strassenMultiplyCPU(h_A, h_B, h_C_strassen, size);
            clock_t strassen_end = clock();
            double strassen_milliseconds = 1000.0 * (strassen_end - strassen_start) / CLOCKS_PER_SEC;

            printf("Standard GPU time: %.3f ms\n", gpu_milliseconds);
            printf("Standard CPU time: %.3f ms\n", cpu_milliseconds);
            printf("Strassen CPU time: %.3f ms\n", strassen_milliseconds);
            printf("GPU/CPU Speedup: %.3f\n", cpu_milliseconds / gpu_milliseconds);
            printf("Strassen/Standard CPU Speedup: %.3f\n", cpu_milliseconds / strassen_milliseconds);

            outFile << size << ","
                << gpu_milliseconds << ","
                << cpu_milliseconds << ","
                << strassen_milliseconds << ","
                << cpu_milliseconds / gpu_milliseconds << ","
                << cpu_milliseconds / strassen_milliseconds << "\n";

            free(h_A);
            free(h_B);
            free(h_C);
            free(h_C_cpu);
            free(h_C_strassen);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }

        outFile.close();
    }
};

int main() {
    srand((unsigned int)time(NULL));

    GPUInfo gpuInfo;
    gpuInfo.printGPUInfo();

    MatrixMultiplicationTest test;
    test.testLargeMatrixMultiplication();

    return 0;
}