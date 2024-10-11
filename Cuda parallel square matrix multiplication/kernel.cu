#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <fstream>

#define BLOCK_SIZE 64 // Define block size for CUDA

// Function to print GPU information
void printGPUInfo() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Get properties of the first GPU (index 0)
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

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int ARows, int ACols, int BCols)
{
    // Allocate shared memory for blocks of matrices A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate global indices
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Calculate row and column indices for the current thread
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Iterate over all blocks of matrices A and B
    for (int m = 0; m < (ACols - 1) / BLOCK_SIZE + 1; ++m) {
        // Load data into shared memory
        if (row < ARows && m * BLOCK_SIZE + tx < ACols)
            As[ty][tx] = A[row * ACols + m * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < BCols && m * BLOCK_SIZE + ty < ACols)
            Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * BCols + col];
        else
            Bs[ty][tx] = 0.0f;

        // Synchronize threads
        __syncthreads();

        // Perform multiplication for the current block
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        // Synchronize threads before the next iteration
        __syncthreads();
    }

    // Store the result
    if (row < ARows && col < BCols)
        C[row * BCols + col] = sum;
}

// Function to print block multiplication information
void printBlockMultiplicationInfo(int ARows, int ACols, int BCols) {
    int blockRowsA = (ARows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blockColsA = (ACols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blockColsB = (BCols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\nBlock Multiplication Information:\n");
    printf("Block size: %dx%d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("Matrix A is split into %dx%d blocks\n", blockRowsA, blockColsA);
    printf("Matrix B is split into %dx%d blocks\n", blockColsA, blockColsB);
    printf("Resulting matrix C will have %dx%d blocks\n", blockRowsA, blockColsB);

    printf("\nMultiplication process:\n");
    for (int i = 0; i < blockRowsA; ++i) {
        for (int j = 0; j < blockColsB; ++j) {
            printf("Calculating block C[%d][%d]:\n", i, j);
            for (int k = 0; k < blockColsA; ++k) {
                printf("  A[%d][%d] * B[%d][%d]\n", i, k, k, j);
            }
        }
    }
}

// Function to print a matrix
void printMatrix(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Function to input a matrix from the console
void inputMatrix(float* matrix, int rows, int cols)
{
    printf("Enter the elements of the matrix %dx%d:\n", rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            scanf_s("%f", &matrix[i * cols + j]);
        }
    }
}

// New function to generate a random matrix
void generateRandomMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function for matrix multiplication on CPU (for comparison)
void matrixMultiplyCPU(float* A, float* B, float* C, int ARows, int ACols, int BCols)
{
    for (int i = 0; i < ARows; ++i)
    {
        for (int j = 0; j < BCols; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < ACols; ++k)
            {
                sum += A[i * ACols + k] * B[k * BCols + j];
            }
            C[i * BCols + j] = sum;
        }
    }
}

// Function to test multiplication of large matrices
void testLargeMatrixMultiplication() {
    std::vector<int> sizes = { 50, 100, 250, 500, 1000, 2000 };
    std::ofstream outFile("matrix_multiplication_results.csv");
    outFile << "Size,GPU Time (ms),CPU Time (ms)\n";

    for (int size : sizes) {
        printf("\nTesting matrix multiplication of size %dx%d\n", size, size);

        // Allocate memory for matrices
        size_t matrixSize = size * size * sizeof(float);
        float* h_A = (float*)malloc(matrixSize);
        float* h_B = (float*)malloc(matrixSize);
        float* h_C = (float*)malloc(matrixSize);
        float* h_C_cpu = (float*)malloc(matrixSize);

        // Generate random matrices
        generateRandomMatrix(h_A, size, size);
        generateRandomMatrix(h_B, size, size);

        // Allocate memory on GPU
        float* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, matrixSize);
        cudaMalloc(&d_B, matrixSize);
        cudaMalloc(&d_C, matrixSize);

        // Copy data to GPU
        cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

        // Set block and grid sizes for CUDA
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Launch kernel and measure time
        cudaEventRecord(start);
        matrixMultiplyKernel << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, size, size, size);
        cudaEventRecord(stop);

        // Copy result back to CPU
        cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

        // Calculate execution time on GPU
        cudaEventSynchronize(stop);
        float gpu_milliseconds = 0;
        cudaEventElapsedTime(&gpu_milliseconds, start, stop);

        // Perform multiplication on CPU and measure time
        clock_t cpu_start = clock();
        matrixMultiplyCPU(h_A, h_B, h_C_cpu, size, size, size);
        clock_t cpu_end = clock();
        double cpu_milliseconds = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

        // Output results
        printf("Execution time on GPU: %.3f ms\n", gpu_milliseconds);
        printf("Execution time on CPU: %.3f ms\n", cpu_milliseconds);
        printf("Speedup: %.3f\n", cpu_milliseconds / gpu_milliseconds);

        // Save results to CSV file
        outFile << size << "," << gpu_milliseconds << "," << cpu_milliseconds << "\n";

        // Free memory
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_cpu);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    outFile.close();
}

// Main function
int main()
{
    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Print GPU info
    printGPUInfo();

    // Test large matrix multiplication
    testLargeMatrixMultiplication();

    return 0;
}
