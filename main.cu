#include <cstdio>
#include <cstdlib>
#include <ctime>

__global__ void editRows(double *data, int matrixOrder, int rowNumber);

__host__ double gaussDeterminant (double* data, double* dev_data, int matrixOrder);

int main() {
    FILE *f1, *f2;

    for (int n = 100; n < 3500; n += 20) {
        double *data = (double *) malloc(sizeof(double) * (n * n));

        srand(time(0));
        for (int i = 0; i < n * n; i++) {
            data[i] = (double) (rand() % 11);
        }

        double determinant = 1;

        f1 = fopen("Matrix Orders.txt", "a");
        printf("matrix order = %d\n", n);
        fprintf(f1, "%d\n", n);
        fclose(f1);

        double *dev_data;

        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

        cudaMalloc((void **) &dev_data, sizeof(double) * (n * n));
        cudaMemcpy(dev_data, data, sizeof(double) * n * n, cudaMemcpyHostToDevice);

        float timeToCalculate = 0.0;

        determinant = gaussDeterminant(data, dev_data, n);

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&timeToCalculate, begin, end);

        f2 = fopen("Time in milli seconds GPU.txt", "a");
        fprintf(f2, "%.2f\n", timeToCalculate);
        fclose(f2);

        printf("time to calculate = %.2f milliseconds\n", timeToCalculate);

        //printf("determinant %f\n", determinant);

        cudaEventDestroy(begin);
        cudaEventDestroy(end);

        cudaFree(dev_data);
        free(data);
    }
    return 0;
}

__global__ void editRows(double *data, int matrixOrder, int rowNumber) {
    __syncthreads();
    unsigned int idx = blockIdx.x;
    if (idx > rowNumber && idx <= matrixOrder - 1) {
        double divider = data[matrixOrder * idx + rowNumber] / data[rowNumber * matrixOrder + rowNumber];
        for (int j = rowNumber; j < matrixOrder; j++)
            data[matrixOrder * idx + j] -= data[rowNumber * matrixOrder + j] * divider;
    }
}

__host__ double gaussDeterminant(double *data, double *dev_data, int matrixOrder) {
    double determinant = 0;
    for (int i = 0; i < matrixOrder - 1; i++) {
        editRows<<<matrixOrder, 1>>>(dev_data, matrixOrder, i);
        cudaDeviceSynchronize();
    }
    cudaThreadSynchronize();
    cudaMemcpy(data, dev_data, matrixOrder * matrixOrder * sizeof(double), cudaMemcpyDeviceToHost);
    determinant = data[0];
    for (int i = 1; i < matrixOrder; i++) {
        determinant *= data[i * matrixOrder + i];
    }
    return determinant;
}
