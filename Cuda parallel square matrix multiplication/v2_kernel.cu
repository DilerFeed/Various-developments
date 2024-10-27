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

#define BLOCK_SIZE 64
#define STRASSEN_THRESHOLD 64

// CUDA kernel must be global function
__global__ void matrixMultiplyKernel(float* A, float* B, float* C,
    int ARows, int ACols, int BCols) {
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

    void copyToHost(float* hostData) {
        cudaMemcpy(hostData, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
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

        GPUMatrix d_A(ARows, ACols);
        GPUMatrix d_B(B.getRows(), BCols);
        GPUMatrix d_C(ARows, BCols);

        d_A.copyToDevice(A.getData());
        d_B.copyToDevice(B.getData());

        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((BCols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (ARows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matrixMultiplyKernel << <numBlocks, threadsPerBlock >> > (
            d_A.getData(), d_B.getData(), d_C.getData(), ARows, ACols, BCols);

        d_C.copyToHost(result->getData());

        return result;
    }
};

class StrassenMultiplier : public MatrixMultiplier {
private:
    std::unique_ptr<StandardMultiplier> standardMultiplier;

public:
    StrassenMultiplier() : standardMultiplier(std::make_unique<StandardMultiplier>()) {}

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

        if (cols_a != cols_b) {
            printf("Error: Matrix dimensions don't match for multiplication\n");
            return;
        }

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
