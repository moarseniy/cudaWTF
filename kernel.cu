
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>

using namespace std;


cudaEvent_t start, stop;


__global__ void func(char* strDev, char* referenceDev, int* dlinaDev, float* scoringDev, int* startDev, int ref_len, int reads_len, int n, int m, const int maximum_read)
{

	int stolb = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ char slice_reference[2561];//84//2136

	//int xstr = stolb % reads_len, ystr = stolb / reads_len;
	//int istr = ystr * reads_len + xstr;

	int sum = 0; int start = startDev[stolb];
	int total_slides = ref_len - dlinaDev[stolb] + 1;
	for (int stroka = 0; stroka < total_slides; stroka++) {
		for (int k = 0; k < dlinaDev[stolb]; k++) {
			slice_reference[k] = referenceDev[stroka + k];
		}
		for (int k = 0; k < dlinaDev[stolb]; k++) {
			sum += (int)(slice_reference[k] == strDev[start + k]);
		}
		scoringDev[stroka + stolb * n] = (float)sum / dlinaDev[stolb];
		sum = 0; //start = 0;
	}

}

/*
__global__ void func(char* strDev, char* referenceDev, int* dlinaDev, float* scoringDev, int ref_len)
{
	extern __shared__ char slice_reference[];
	int index = threadIdx.x;

	int xstr = index % dlinaDev[index], ystr = index / dlinaDev[index];
	int istr = ystr * dlinaDev[index] + xstr;

	int sum = 0; int start = 0;
	int total_slides = ref_len - dlinaDev[index] + 1;
	for (int i = 0; i < total_slides; i++) {
		for (int k = 0; k < index; k++) {
			start += dlinaDev[k];
		}
		for (int k = 0; k < dlinaDev[index]; k++) {
			slice_reference[k] = referenceDev[i + k];
		}
		for (int k = 0; k < dlinaDev[index]; k++) {
			sum +=(int)(slice_reference[k] == strDev[start + k]);
		}
		scoringDev[istr] = sum / dlinaDev[index];
		sum = 0;
	}

}*/


int minimum(int* dlina, int reads_count) {
	int min = dlina[0];
	for (int i = 1; i < reads_count; i++) {
		if (dlina[i] < min) {
			min = dlina[i];
		}
	}
	return min;
}

int maximum(int* dlina, int reads_count) {
	int max = dlina[0];
	for (int i = 1; i < reads_count; i++) {
		if (dlina[i] > max) {
			max = dlina[i];
		}
	}
	return max;
}


int main(void)
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float gpuTime = 0.0;

	ifstream dna, reads, lengths;
	ofstream out;
	out.open("output.txt");
	lengths.open("lengths.txt");
	dna.open("dna.txt");
	reads.open("reads.txt");

	int ref_len = 48502;//48502;
	int reads_len = 302094;//2056551;//302094;//2056551;//302094;//2056551;
	int reads_count = 900;//6000;

	int* dlina = (int*)malloc(reads_count * sizeof(int));
	for (int i = 0; i < reads_count; i++) {
		lengths >> dlina[i];
		//cout << dlina[i] << " ";
	}

	/*
	int number_elements = 0;
	for (int i = 0; i < reads_count; i++) {
		number_elements += dlina[i];
	}
	cout << "summa=" << number_elements << endl;
	*/

	char* reference = (char*)malloc(ref_len * sizeof(char));
	for (int i = 0; i < ref_len; i++) {
		dna >> reference[i];
	}
	//for (int i = 0; i < ref_len; i++) {
	//	cout << reference[i];
	//}

	/*
	char** str = (char**)malloc(6000 * sizeof(char*));
	for (int i = 0; i < 6000; i++) {
		str[i] = (char*)malloc(dlina[i] * sizeof(char));
	}

	for (int i = 0; i < 6000; i++) {
		for (int j = 0; j < dlina[i]; j++) {
			reads >> str[i][j];
		}
	}
	*/
	char* str = new char[reads_len];
	for (int i = 0; i < reads_len; i++) {
		reads >> str[i];
	}
	/*
	for (int j = 0; j < reads_len; j++) {
		cout << str[j];
	}*/

	int n_scoring = ref_len - minimum(dlina, reads_count) + 1;//48459;
	int m_scoring = reads_count;

	float* scoring_matrix = new float[n_scoring * m_scoring];
	for (int i = 0; i < n_scoring * m_scoring; i++) {
		scoring_matrix[i] = 0;
	}

	int maximum_read = maximum(dlina, reads_count);
	cout << maximum_read << endl;

	//char* slice_reference = new char[maximum_read];
	//cout << n_scoring * m_scoring;
	//cout << ref_len - minimum_read(dlina, reads_count) + 1 << endl;
	//cout << "min=" << minimum_read(dlina, reads_count) << endl;

	int sum = 0;
	int* start_values = (int*)malloc(reads_count * sizeof(int));
	for (int i = 0; i < reads_count; i++) {
		sum += dlina[i];
		start_values[i] = sum;
	}

	cudaSetDevice(0);

	char* strDev;
	char* referenceDev;
	int* dlinaDev;
	float* scoringDev;
	int* startDev;

	cudaMalloc((void**)&strDev, reads_len * sizeof(char));
	cudaMalloc((void**)&referenceDev, ref_len * sizeof(char));
	cudaMalloc((void**)&dlinaDev, reads_count * sizeof(int));
	cudaMalloc((void**)&scoringDev, n_scoring * m_scoring * sizeof(float));
	cudaMalloc((void**)&startDev, reads_count * sizeof(int));

	cudaMemcpy(strDev, str, reads_len * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(referenceDev, reference, ref_len * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(dlinaDev, dlina, reads_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(startDev, start_values, reads_count * sizeof(int), cudaMemcpyHostToDevice);


	cudaEventRecord(start, 0);

	func << <3, 300 >> > (strDev, referenceDev, dlinaDev, scoringDev, startDev, ref_len, reads_len, n_scoring, m_scoring, maximum_read);





	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("GPU time = %.4f \n", gpuTime);


	cudaMemcpy((void*)scoring_matrix, scoringDev, n_scoring * m_scoring * sizeof(float), cudaMemcpyDeviceToHost);


	
	for (int i = 0; i < n_scoring; i++) {
		for (int j = 0; j < m_scoring; j++) {
			out << scoring_matrix[i + j * n_scoring] << " ";
		}
		out << endl;
	}
	/*
	for (int i = 0; i < n_scoring; i++) {
		for (int j = 0; j < m_scoring; j++) {
			cout << scoring_matrix[i + j * n_scoring] << " ";
		}
		cout << endl;
	}*/

	cout << scoring_matrix[0] << endl;
	cout << scoring_matrix[1] << endl;
	cout << scoring_matrix[2] << endl;

	delete[] str;
	//delete[] reference;
	//delete[] dlina;
	delete[] scoring_matrix;

	cudaFree(strDev);
	cudaFree(referenceDev);
	cudaFree(dlinaDev);
	cudaFree(scoringDev);


	return 0;
}
