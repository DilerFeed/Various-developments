#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <memory>

#define BLOCK_SIZE 32
#define STRASSEN_THRESHOLD 64

// CUDA kernel must be global function
__global__ void matrixMultiplyKernel(float* A, float* B, float* C,
    int ARows, int ACols, int BCols) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Глобальні індекси
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Calculate the number of blocks
    int numBlocks = (ACols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int m = 0; m < numBlocks; m++) {
        // Load matrix block A
        if (row < ARows && (m * BLOCK_SIZE + tx) < ACols) {
            As[ty][tx] = A[row * ACols + (m * BLOCK_SIZE + tx)];
        }
        else {
            As[ty][tx] = 0.0f;
        }

        // Load matrix block B
        if ((m * BLOCK_SIZE + ty) < ACols && col < BCols) {
            Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * BCols + col];
        }
        else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply the blocks
        if (row < ARows && col < BCols) {
            for (int k = 0; k < BLOCK_SIZE && (m * BLOCK_SIZE + k) < ACols; k++) {
                sum += As[ty][k] * Bs[k][tx];
            }
        }

        __syncthreads();
    }

    // Record the result
    if (row < ARows && col < BCols) {
        C[row * BCols + col] = sum;
    }
}

__global__ void matrixAddKernel(float* A, float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void matrixSubtractKernel(float* A, float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] - B[idx];
    }
}


// Base Matrix class
class Matrix {
protected:
    float* data;
    int rows;
    int cols;
    bool isOnDevice;

public:
    Matrix(int r, int c) : rows(r), cols(c), isOnDevice(false) {
        data = new float[rows * cols];
    }

    virtual ~Matrix() {
        if (!isOnDevice && data != nullptr) {
            delete[] data;
        }
    }

    void generateRandom() {
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    void inputFromConsole() {
        printf("Enter the elements of the %dx%d matrix:\n", rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                printf("Element[%d][%d]: ", i, j);
                scanf_s("%f", &data[i * cols + j]);
            }
        }
    }

    void print() const {
        printf("\nMatrix %dx%d:\n", rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                printf("%.2f ", data[i * cols + j]);
            }
            printf("\n");
        }
    }

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    float* getData() const { return data; }

    std::unique_ptr<Matrix> getSubMatrix(int startRow, int startCol, int subRows, int subCols) const {
        auto subMatrix = std::make_unique<Matrix>(subRows, subCols);
        for (int i = 0; i < subRows; ++i) {
            for (int j = 0; j < subCols; ++j) {
                subMatrix->data[i * subCols + j] = data[(startRow + i) * cols + (startCol + j)];
            }
        }
        return subMatrix;
    }

    static std::unique_ptr<Matrix> add(const Matrix& A, const Matrix& B) {
        if (A.rows != B.rows || A.cols != B.cols) return nullptr;

        auto result = std::make_unique<Matrix>(A.rows, A.cols);
        for (int i = 0; i < A.rows * A.cols; ++i) {
            result->data[i] = A.data[i] + B.data[i];
        }
        return result;
    }

    static std::unique_ptr<Matrix> subtract(const Matrix& A, const Matrix& B) {
        if (A.rows != B.rows || A.cols != B.cols) return nullptr;

        auto result = std::make_unique<Matrix>(A.rows, A.cols);
        for (int i = 0; i < A.rows * A.cols; ++i) {
            result->data[i] = A.data[i] - B.data[i];
        }
        return result;
    }
};

class GPUMatrix : public Matrix {
public:
    GPUMatrix(int r, int c) : Matrix(r, c) {
        cudaMalloc(&data, rows * cols * sizeof(float));
        isOnDevice = true;
    }

    ~GPUMatrix() {
        if (isOnDevice && data != nullptr) {
            cudaFree(data);
        }
    }

    void copyToDevice(const float* hostData) {
        cudaMemcpy(data, hostData, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    }

    void copyToHost(float* hostData) const {
        cudaMemcpy(hostData, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    }

    static std::unique_ptr<GPUMatrix> add(const GPUMatrix& A, const GPUMatrix& B) {
        if (A.rows != B.rows || A.cols != B.cols) return nullptr;

        auto result = std::make_unique<GPUMatrix>(A.rows, A.cols);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (A.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matrixAddKernel << <gridSize, blockSize >> > (
            A.data, B.data, result->data, A.rows, A.cols);

        return result;
    }

    static std::unique_ptr<GPUMatrix> subtract(const GPUMatrix& A, const GPUMatrix& B) {
        if (A.rows != B.rows || A.cols != B.cols) return nullptr;

        auto result = std::make_unique<GPUMatrix>(A.rows, A.cols);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (A.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matrixSubtractKernel << <gridSize, blockSize >> > (
            A.data, B.data, result->data, A.rows, A.cols);

        return result;
    }

    std::unique_ptr<GPUMatrix> getSubMatrix(int startRow, int startCol, int subRows, int subCols) const {
        auto result = std::make_unique<GPUMatrix>(subRows, subCols);

        // Allocate temporary memory on the CPU
        float* temp = new float[subRows * subCols];
        float* fullMatrix = new float[rows * cols];

        // Copy the entire matrix to the CPU
        copyToHost(fullMatrix);

        // Select a submatrix
        for (int i = 0; i < subRows; ++i) {
            for (int j = 0; j < subCols; ++j) {
                temp[i * subCols + j] = fullMatrix[(startRow + i) * cols + (startCol + j)];
            }
        }

        // Copy the submatrix to the GPU
        result->copyToDevice(temp);

        delete[] temp;
        delete[] fullMatrix;

        return result;
    }
};

class MatrixMultiplier {
public:
    virtual std::unique_ptr<Matrix> multiply(const Matrix& A, const Matrix& B) = 0;
    virtual ~MatrixMultiplier() = default;
};

class StandardMultiplier : public MatrixMultiplier {
public:
    std::unique_ptr<Matrix> multiply(const Matrix& A, const Matrix& B) override {
        if (A.getCols() != B.getRows()) return nullptr;

        int ARows = A.getRows(), ACols = A.getCols(), BCols = B.getCols();
        auto result = std::make_unique<Matrix>(ARows, BCols);

        // Allocate memory on the GPU
        GPUMatrix d_A(ARows, ACols);
        GPUMatrix d_B(B.getRows(), BCols);
        GPUMatrix d_C(ARows, BCols);

        // Copy the data to the GPU
        d_A.copyToDevice(A.getData());
        d_B.copyToDevice(B.getData());

        // Adjust the block and grid sizes
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks(
            (BCols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (ARows + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

        // Start the kernel
        matrixMultiplyKernel << <numBlocks, threadsPerBlock >> > (
            d_A.getData(), d_B.getData(), d_C.getData(), ARows, ACols, BCols);

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return nullptr;
        }

        // Synchronize and wait for completion
        cudaDeviceSynchronize();

        // Copy the result back
        d_C.copyToHost(result->getData());

        return result;
    }
};

class StrassenMultiplier : public MatrixMultiplier {
private:
    std::unique_ptr<StandardMultiplier> standardMultiplier;

    bool isPowerOfTwo(int n) {
        return (n & (n - 1)) == 0;
    }

public:
    StrassenMultiplier() : standardMultiplier(std::make_unique<StandardMultiplier>()) {}

    std::unique_ptr<Matrix> multiply(const Matrix& A, const Matrix& B) override {
        if (A.getCols() != B.getRows()) return nullptr;

        if (!isPowerOfTwo(A.getRows()) || !isPowerOfTwo(A.getCols()) ||
            !isPowerOfTwo(B.getCols()) ||
            A.getRows() != A.getCols() || B.getRows() != B.getCols()) {
            printf("Error: Strassen's method requires square matrices with dimensions that are powers of 2\n");
            return nullptr;
        }

        // Create GPU matrices
        GPUMatrix d_A(A.getRows(), A.getCols());
        GPUMatrix d_B(B.getRows(), B.getCols());
        d_A.copyToDevice(A.getData());
        d_B.copyToDevice(B.getData());

        // Call the recursive function for GPU matrices
        auto d_result = multiplyGPU(d_A, d_B);

        // Copy the result back to the CPU
        auto result = std::make_unique<Matrix>(A.getRows(), B.getCols());
        d_result->copyToHost(result->getData());

        return result;
    }

private:
    std::unique_ptr<GPUMatrix> multiplyGPU(const GPUMatrix& A, const GPUMatrix& B) {
        int n = A.getRows();

        if (n <= STRASSEN_THRESHOLD) {
            auto result = std::make_unique<GPUMatrix>(n, n);

            dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            matrixMultiplyKernel << <numBlocks, threadsPerBlock >> > (
                A.getData(), B.getData(), result->getData(), n, n, n);

            return result;
        }

        int m = n / 2;

        // Get submatrices (all operations are performed on the GPU)
        auto A11 = A.getSubMatrix(0, 0, m, m);
        auto A12 = A.getSubMatrix(0, m, m, m);
        auto A21 = A.getSubMatrix(m, 0, m, m);
        auto A22 = A.getSubMatrix(m, m, m, m);

        auto B11 = B.getSubMatrix(0, 0, m, m);
        auto B12 = B.getSubMatrix(0, m, m, m);
        auto B21 = B.getSubMatrix(m, 0, m, m);
        auto B22 = B.getSubMatrix(m, m, m, m);

        // Calculate M1-M7 (all operations on the GPU)
        auto M1 = multiplyGPU(*GPUMatrix::add(*A11, *A22), *GPUMatrix::add(*B11, *B22));
        auto M2 = multiplyGPU(*GPUMatrix::add(*A21, *A22), *B11);
        auto M3 = multiplyGPU(*A11, *GPUMatrix::subtract(*B12, *B22));
        auto M4 = multiplyGPU(*A22, *GPUMatrix::subtract(*B21, *B11));
        auto M5 = multiplyGPU(*GPUMatrix::add(*A11, *A12), *B22);
        auto M6 = multiplyGPU(*GPUMatrix::subtract(*A21, *A11), *GPUMatrix::add(*B11, *B12));
        auto M7 = multiplyGPU(*GPUMatrix::subtract(*A12, *A22), *GPUMatrix::add(*B21, *B22));

        // Calculate parts of the resulting matrix (all operations on the GPU)
        auto C11 = GPUMatrix::add(*GPUMatrix::subtract(*GPUMatrix::add(*M1, *M4), *M5), *M7);
        auto C12 = GPUMatrix::add(*M3, *M5);
        auto C21 = GPUMatrix::add(*M2, *M4);
        auto C22 = GPUMatrix::add(*GPUMatrix::subtract(*GPUMatrix::add(*M1, *M3), *M2), *M6);

        // Combine the result
        auto result = std::make_unique<GPUMatrix>(n, n);
        float* temp = new float[n * n];
        float* C11Data = new float[m * m];
        float* C12Data = new float[m * m];
        float* C21Data = new float[m * m];
        float* C22Data = new float[m * m];

        C11->copyToHost(C11Data);
        C12->copyToHost(C12Data);
        C21->copyToHost(C21Data);
        C22->copyToHost(C22Data);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                temp[i * n + j] = C11Data[i * m + j];
                temp[i * n + (j + m)] = C12Data[i * m + j];
                temp[(i + m) * n + j] = C21Data[i * m + j];
                temp[(i + m) * n + (j + m)] = C22Data[i * m + j];
            }
        }

        result->copyToDevice(temp);

        delete[] temp;
        delete[] C11Data;
        delete[] C12Data;
        delete[] C21Data;
        delete[] C22Data;

        return result;
    }
};

class CPUMultiplier : public MatrixMultiplier {
public:
    std::unique_ptr<Matrix> multiply(const Matrix& A, const Matrix& B) override {
        if (A.getCols() != B.getRows()) return nullptr;

        int ARows = A.getRows(), ACols = A.getCols(), BCols = B.getCols();
        auto result = std::make_unique<Matrix>(ARows, BCols);
        float* C = result->getData();
        const float* a = A.getData();
        const float* b = B.getData();

        for (int i = 0; i < ARows; i++) {
            for (int j = 0; j < BCols; j++) {
                float sum = 0;
                for (int k = 0; k < ACols; k++) {
                    sum += a[i * ACols + k] * b[k * BCols + j];
                }
                C[i * BCols + j] = sum;
            }
        }
        return result;
    }
};

class CPUStrassenMultiplier : public MatrixMultiplier {
private:
    std::unique_ptr<CPUMultiplier> standardMultiplier;

public:
    CPUStrassenMultiplier() : standardMultiplier(std::make_unique<CPUMultiplier>()) {}

    std::unique_ptr<Matrix> multiply(const Matrix& A, const Matrix& B) override {
        if (A.getCols() != B.getRows()) return nullptr;

        if (A.getRows() <= STRASSEN_THRESHOLD || A.getCols() <= STRASSEN_THRESHOLD ||
            B.getCols() <= STRASSEN_THRESHOLD) {
            return standardMultiplier->multiply(A, B);
        }

        int n = A.getRows();
        int m = n / 2;

        auto A11 = A.getSubMatrix(0, 0, m, m);
        auto A12 = A.getSubMatrix(0, m, m, m);
        auto A21 = A.getSubMatrix(m, 0, m, m);
        auto A22 = A.getSubMatrix(m, m, m, m);

        auto B11 = B.getSubMatrix(0, 0, m, m);
        auto B12 = B.getSubMatrix(0, m, m, m);
        auto B21 = B.getSubMatrix(m, 0, m, m);
        auto B22 = B.getSubMatrix(m, m, m, m);

        auto M1 = multiply(*Matrix::add(*A11, *A22), *Matrix::add(*B11, *B22));
        auto M2 = multiply(*Matrix::add(*A21, *A22), *B11);
        auto M3 = multiply(*A11, *Matrix::subtract(*B12, *B22));
        auto M4 = multiply(*A22, *Matrix::subtract(*B21, *B11));
        auto M5 = multiply(*Matrix::add(*A11, *A12), *B22);
        auto M6 = multiply(*Matrix::subtract(*A21, *A11), *Matrix::add(*B11, *B12));
        auto M7 = multiply(*Matrix::subtract(*A12, *A22), *Matrix::add(*B21, *B22));

        auto C11 = Matrix::add(*Matrix::subtract(*Matrix::add(*M1, *M4), *M5), *M7);
        auto C12 = Matrix::add(*M3, *M5);
        auto C21 = Matrix::add(*M2, *M4);
        auto C22 = Matrix::add(*Matrix::subtract(*Matrix::add(*M1, *M3), *M2), *M6);

        auto result = std::make_unique<Matrix>(n, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                result->getData()[i * n + j] = C11->getData()[i * m + j];
                result->getData()[i * n + (j + m)] = C12->getData()[i * m + j];
                result->getData()[(i + m) * n + j] = C21->getData()[i * m + j];
                result->getData()[(i + m) * n + (j + m)] = C22->getData()[i * m + j];
            }
        }

        return result;
    }
};

class MatrixMultiplicationTester {
private:
    std::unique_ptr<MatrixMultiplier> multiplier;
    std::ofstream outFile;
    const char* multiplierName;

public:
    MatrixMultiplicationTester(std::unique_ptr<MatrixMultiplier> m, const std::string& filename, const char* name)
        : multiplier(std::move(m)), multiplierName(name) {
        outFile.open(filename);
        outFile << "Size,Time (ms)\n";
    }

    ~MatrixMultiplicationTester() {
        if (outFile.is_open()) {
            outFile.close();
        }
    }

    void testManualInput() {
        int rows_a, cols_a, cols_b;

        printf("\nTesting %s multiplier with manual input\n", multiplierName);
        printf("Enter dimensions for first matrix (rows cols): ");
        scanf_s("%d %d", &rows_a, &cols_a);

        printf("Enter dimensions for second matrix (cols): ");
        scanf_s("%d", &cols_b);

        Matrix A(rows_a, cols_a);
        Matrix B(cols_a, cols_b);

        printf("\nEnter first matrix:\n");
        A.inputFromConsole();

        printf("\nEnter second matrix:\n");
        B.inputFromConsole();

        printf("\nFirst matrix:");
        A.print();
        printf("\nSecond matrix:");
        B.print();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        auto C = multiplier->multiply(A, B);
        cudaEventRecord(stop);

        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("\nResult matrix:");
        C->print();
        printf("\nExecution time: %.3f ms\n", milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void runPerformanceTests(const std::vector<int>& sizes) {
        printf("\nRunning performance tests for %s multiplier:\n", multiplierName);
        for (int size : sizes) {
            printf("\nTesting %dx%d matrices\n", size, size);

            Matrix A(size, size);
            Matrix B(size, size);
            A.generateRandom();
            B.generateRandom();

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            auto C = multiplier->multiply(A, B);
            cudaEventRecord(stop);

            float milliseconds = 0;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            printf("Execution time: %.3f ms\n", milliseconds);
            outFile << size << "," << milliseconds << "\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
};

int main() {
    srand((unsigned int)time(NULL));

    // Print GPU info
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("GPU Information:\n");
    printf("Name: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf("Global memory: %zu MB\n", deviceProp.totalGlobalMem / (1024 * 1024));
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

    int choice;
    do {
        printf("\nMatrix Multiplication Menu:\n");
        printf("1. Manual input test (Standard multiplier)\n");
        printf("2. Manual input test (Strassen multiplier)\n");
        printf("3. Run performance tests\n");
        printf("4. Exit\n");
        printf("Enter your choice: ");
        scanf_s("%d", &choice);

        switch (choice) {
        case 1: {
            MatrixMultiplicationTester tester(
                std::make_unique<StandardMultiplier>(),
                "standard_manual_test.csv",
                "Standard"
            );
            tester.testManualInput();
            break;
        }
        case 2: {
            MatrixMultiplicationTester tester(
                std::make_unique<StrassenMultiplier>(),
                "strassen_manual_test.csv",
                "Strassen"
            );
            tester.testManualInput();
            break;
        }
        case 3: {
            std::vector<int> sizes = { 64, 128, 256, 512, 1024, 2048 };

            // Test Standard GPU multiplier
            {
                MatrixMultiplicationTester tester(
                    std::make_unique<StandardMultiplier>(),
                    "standard_gpu_performance_test.csv",
                    "Standard GPU"
                );
                tester.runPerformanceTests(sizes);
            }

            // Test Strassen GPU multiplier
            {
                MatrixMultiplicationTester tester(
                    std::make_unique<StrassenMultiplier>(),
                    "strassen_gpu_performance_test.csv",
                    "Strassen GPU"
                );
                tester.runPerformanceTests(sizes);
            }

            // Test Standard CPU multiplier
            {
                MatrixMultiplicationTester tester(
                    std::make_unique<CPUMultiplier>(),
                    "standard_cpu_performance_test.csv",
                    "Standard CPU"
                );
                tester.runPerformanceTests(sizes);
            }

            // Test Strassen CPU multiplier
            {
                MatrixMultiplicationTester tester(
                    std::make_unique<CPUStrassenMultiplier>(),
                    "strassen_cpu_performance_test.csv",
                    "Strassen CPU"
                );
                tester.runPerformanceTests(sizes);
            }
            break;
        }
        case 4:
            printf("\nExiting program...\n");
            break;
        default:
            printf("\nInvalid choice! Please try again.\n");
        }
    } while (choice != 4);

    return 0;
}