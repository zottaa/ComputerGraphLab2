#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <ctime>


__global__ void determinantGaussGPU(double* data, int matrixOrder, double* result);
__device__ void matrixToTriangle(double* data, int matrixOrder);

int main() {
    // FILE *f1, *f2;
    // f1 = fopen("Matrix Orders.txt", "w");
    // f2 = fopen("Time in milli seconds.txt", "w");
    int n = 0;

    int orders[1] = {3};

    for(n = 0; n < 1; n++){
        double *data = (double*) malloc(sizeof(double) * (orders[n] * orders[n]));

        srand(time(0));
        for (int i = 0; i < orders[n] * orders[n]; i++){
            data[i] =(double) (rand() % 11);
        }

        for (int i = 0; i != orders[n]; i++){
            for (int j = 0; j != orders[n]; j++)
                printf("%f ", data[i + orders[n] * j]);
            printf("\n");
        }

        double determinant = 1;

        printf("matrix order = %d\n", orders[n]);

        double *dev_det;
        double *dev_data;

        cudaMalloc((void**)&dev_data, sizeof(double) * (orders[n] * orders[n]));
        cudaMemcpy(dev_data, data, sizeof(double) * orders[n] * orders[n], cudaMemcpyHostToDevice);

        cudaMalloc(&dev_det, sizeof(double));
        cudaMemcpy(dev_det, &determinant, sizeof(double), cudaMemcpyHostToDevice);

        float timeToCalculate = 0.0;
        cudaEvent_t begin, end;

        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

        determinantGaussGPU <<<orders[n], orders[n]>>>(dev_data, orders[n], dev_det);

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&timeToCalculate, begin, end);
        printf("time to calculate = %.2f milliseconds\n", timeToCalculate);


        cudaMemcpy(&determinant, dev_det, sizeof(double), cudaMemcpyDeviceToHost);

        printf("determinant %f", determinant);

        cudaEventDestroy(begin);
        cudaEventDestroy(end);

        cudaFree(dev_det);
        cudaFree(dev_data);
        free(data);
    }


    return 0;
}

__global__ void determinantGaussGPU(double* data, int matrixOrder, double* result){
    unsigned int row = blockIdx.x;

    matrixToTriangle(data, matrixOrder);

    *result *= data[row + blockDim.x * row];
}


__device__ void matrixToTriangle(double* data, int matrixOrder){
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;

    double divider = 0.0;
    if (data[row + blockDim.x * row] != 0)
        divider = data[col + blockDim.x * row] / data[row + blockDim.x * row];
    for(int k = 0; k < matrixOrder; k++){
        data[col + blockDim.x * k] -= data[row + blockDim.x] * divider;
    }
}
