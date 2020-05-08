//BETA TEST SOLVE SLAE SPARSE MATRIX
#include <stdio.h>
#include <stdlib.h>
//#include <iostream>
#include <assert.h>
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cusparse_v2.h>


cusparseHandle_t handle;


cusparseMatDescr_t descrA = 0;
cusparseMatDescr_t descr_L = 0;
cusparseMatDescr_t descr_U = 0;

csrilu02Info_t info_A = 0;
csrsv2Info_t info_L = 0;
csrsv2Info_t info_U = 0;

void* pBuffer = 0;


void setUpLU(cusparseMatDescr_t& descrLU, cusparseMatrixType_t matrixType, cusparseIndexBase_t indexBase, cusparseFillMode_t fillMode, cusparseDiagType_t diagType) {
    cusparseCreateMatDescr(&descrLU);
    cusparseSetMatType(descrLU, matrixType);
    cusparseSetMatIndexBase(descrLU, indexBase);
    cusparseSetMatFillMode(descrLU, fillMode);
    cusparseSetMatDiagType(descrLU, diagType);
}
 
void memoryLU(csrilu02Info_t& info_A, csrsv2Info_t& info_L, csrsv2Info_t& info_U, cusparseHandle_t handle, const int N, const int nnz, cusparseMatDescr_t descrA, cusparseMatDescr_t descr_L,
    cusparseMatDescr_t descr_U, double* d_A, int* d_A_RowIndices, int* d_A_ColIndices, cusparseOperation_t matrixOperation, void** pBuffer) {

    cusparseCreateCsrilu02Info(&info_A);
    cusparseCreateCsrsv2Info(&info_L);
    cusparseCreateCsrsv2Info(&info_U);

    int pBufferSize_M, pBufferSize_L, pBufferSize_U;
    cusparseDcsrilu02_bufferSize(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, &pBufferSize_M);
    cusparseDcsrsv2_bufferSize(handle, matrixOperation, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, &pBufferSize_L);
    cusparseDcsrsv2_bufferSize(handle, matrixOperation, N, nnz, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, info_U, &pBufferSize_U);

    int pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
    cudaMalloc((void**)pBuffer, pBufferSize);
    printf("razmer matric= %d %d %d ", pBufferSize_M, pBufferSize_L, pBufferSize_U);
    
}


void analysisLU(csrilu02Info_t& info_A, csrsv2Info_t& info_L, csrsv2Info_t& info_U, cusparseHandle_t handle, const int N, const int nnz, cusparseMatDescr_t descrA, cusparseMatDescr_t descr_L,
    cusparseMatDescr_t descr_U, double* d_A, int* d_A_RowIndices, int* d_A_ColIndices, cusparseOperation_t matrixOperation, cusparseSolvePolicy_t solvePolicy1, cusparseSolvePolicy_t solvePolicy2, void* pBuffer) {

    int structural_zero;

    cusparseDcsrilu02_analysis(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, solvePolicy1, pBuffer);
    cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, info_A, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) 
    { 
        printf("A(%d,%d) net\n", structural_zero, structural_zero);
    }

    cusparseDcsrsv2_analysis(handle, matrixOperation, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, solvePolicy1, pBuffer);
    
    cusparseDcsrsv2_analysis(handle, matrixOperation, N, nnz, descr_U, d_A, d_A_RowIndices, d_A_ColIndices, info_U, solvePolicy2, pBuffer);

}

void computeSparseLU(csrilu02Info_t& info_A, cusparseHandle_t handle, const int N, const int nnz, cusparseMatDescr_t descrA, double* d_A, int* d_A_RowIndices,
    int* d_A_ColIndices, cusparseSolvePolicy_t solutionPolicy, void* pBuffer) {

    int numerical_zero;

    cusparseDcsrilu02(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, solutionPolicy, pBuffer);
    cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, info_A, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) 
    { 
        printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero); 
    }

}


int main(void)
{
    
    cusparseCreate(&handle);

    const int Nrows = 4;    //kolvo strok                  
    const int Ncols = 4;    //kolvo stolbcov                      
    const int N = Nrows;

    
    FILE* right = fopen("right.txt", "r");
    double* h_x = (double*)malloc(Nrows * sizeof(*h_x));
    for(int i = 0; i<Nrows; i++)
    {
        fscanf(right, "%lf", &h_x[i]);
        //printf("%lf ", h_x[i]);
        
    }
    fclose(right);

    double* d_x;
    cudaMalloc(&d_x, Nrows * sizeof(*d_x));
    cudaMemcpy(d_x, h_x, Nrows * sizeof(*h_x), cudaMemcpyHostToDevice);

    

    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    

    FILE* data = fopen("data.txt", "r");
    FILE* ind = fopen("ind.txt", "r");
    FILE* ptr = fopen("ptr.txt", "r");
    
    double* h_A = (double*)malloc(Nrows * Ncols * sizeof(*h_A));
    
    int nnz=0, k = 0;
    while(fscanf(data,"%lf",&h_A[k])==1)
    {
        //printf("%lf ",h_A[k]);
        nnz++;
        k++;
            
    }
    printf("non zero = %d \n",nnz);

    double* d_A;       
    cudaMalloc(&d_A, nnz * sizeof(*d_A));
    cudaMemcpy(d_A, h_A, nnz * sizeof(*h_A), cudaMemcpyHostToDevice);

    
    
    int* d_ptr;    cudaMalloc(&d_ptr, (Nrows + 1) * sizeof(*d_ptr));
    int* d_ind;    cudaMalloc(&d_ind, nnz * sizeof(*d_ind));

    int* h_ptr = (int*)malloc((Nrows + 1) * sizeof(*h_ptr));
    int* h_ind = (int*)malloc(nnz * sizeof(*h_ind));
    k = 0;
    
    while (fscanf(ind, "%d", &h_ind[k]) == 1)
    {
        //printf("%lf ", h_ind[k]);
        k++;

    }
    k = 0;
    while (fscanf(ptr, "%d", &h_ptr[k]) == 1)
    {
        //printf("%lf ", h_ptr[k]);
        k++;

    }
    cudaMemcpy(d_ptr, h_ptr, (Nrows + 1) * sizeof(*h_ptr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ind, h_ind, nnz * sizeof(*h_ind), cudaMemcpyHostToDevice);
    fclose(data);fclose(ptr);fclose(ind);

    

    printf("\nCSR MATRIX:\n\n");
    for (int i = 0; i < nnz; ++i) 
        printf("A[%i] = %.0f ", i, h_A[i]); 
    printf("\n");

    printf("\n");
    for (int i = 0; i < (Nrows + 1); ++i) 
        printf("h_ptr[%i] = %i \n", i, h_ptr[i]); 
    printf("\n");

    for (int i = 0; i < nnz; ++i) 
        printf("h_ind[%i] = %i \n", i, h_ind[i]);
    
   
    
    setUpLU(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ONE, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_UNIT);
    setUpLU(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ONE, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);   
    memoryLU(info_A, info_L, info_U, handle, N, nnz, descrA, descr_L, descr_U, d_A, d_ptr, d_ind, CUSPARSE_OPERATION_NON_TRANSPOSE, &pBuffer);
    

    analysisLU(info_A, info_L, info_U, handle, N, nnz, descrA, descr_L, descr_U, d_A, d_ptr, d_ind, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_SOLVE_POLICY_NO_LEVEL,CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);
    
    /* PARASHA
    double* d_A1;
    double* h_A1 = (double*)malloc(nnz * sizeof(double));
    cudaMalloc(&d_A1, nnz * sizeof(*d_A1));
    cudaMemcpy(h_A1, d_A, nnz * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_A1, h_A1, nnz * sizeof(double), cudaMemcpyHostToDevice);
    for (int i = 0; i < nnz; i++)
        printf("%lf ", h_A1[i]);*/


    //A = L * U 
    computeSparseLU(info_A, handle, N, nnz, descrA, d_A, d_ptr, d_ind, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);//TUT OSHIBKA GDE TO!!!!!!!!!!!!!!!!!!!!!!!
    

    double* d_ztest = (double*)malloc(nnz * sizeof(*d_ztest));
    cudaMemcpy(d_ztest, d_A, nnz * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nnz; i++)
        printf("%lf ", d_ztest[i]);


    //L * z = x 
    const double alpha = 1.;
    
    double* d_z; 
    cudaMalloc(&d_z, N * sizeof(*d_z));
   
    cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, &alpha, descr_L, d_A, d_ptr, d_ind, info_L, d_x, d_z, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
    
    

    printf("\n");

    //U * y = z 
    double* d_y;        
    cudaMalloc(&d_y, Ncols * sizeof(*d_y));

    cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, &alpha, descr_U, d_A, d_ptr, d_ind, info_U, d_z, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);


    //resultat
    double* h_y = (double*)malloc(Ncols * sizeof(*h_y));
    cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("\n\nRESULTAT\n");

    for (int k = 0; k < N; k++) 
        printf("x[%i] = %f\n", k, h_y[k]);

    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyCsrilu02Info(info_A);
    cusparseDestroyCsrsv2Info(info_L);
    cusparseDestroyCsrsv2Info(info_U);
    cusparseDestroy(handle);
}