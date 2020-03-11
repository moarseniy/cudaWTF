
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>
#include <ctime>

#define N 50000

void add_cpu(int* a, int* b, int* c)
{
	int tid = 0;
	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		tid += 1;
	}
}

__global__ void add_gpu(int* a, int* b, int* c)
{
	int tid = blockIdx.x;
	if (tid<N)
		c[tid] = a[tid] + b[tid];
}
int main()
{
	cudaEvent_t start, stop;
	float gpuTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//1 part
	int a[N];
	int b[N];
	int c[N];
	
	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	cudaEventRecord(start, 0);
	add_cpu(a, b, c);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("GPU time = %.4f \n", gpuTime);
	gpuTime = 0.0;
	for (int i = 0; i < N; i++)
	{
		//printf("%d + %d =%d\n", a[i], b[i], c[i]);
	}
	


	//2 part
	int* dev_a;
	int* dev_b;
	int* dev_c;
	
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);

	add_gpu <<< N, 1 >>> (dev_a, dev_b, dev_c);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("GPU time = %.4f \n", gpuTime);

	cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		//printf("%d + %d =%d\n", a[i], b[i], c[i]);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
