#include <stdio.h>
#include <stdlib.h>
#include <cstdio>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <helper_string.h>
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace std;

#define len1  1
#define len3  3
#define len33  9
#define len4  4
#define len9  9
#define len93  27
#define len94  36
#define len99  81

#define lambda 0.000001
#define TINY 1.0e-40

#define a(i,j) a[(i)*len9+(j)]

#define channel  3

//=========================================================================================================================================================//

//simple matrix operation
__device__ void rowFirst_printMat(double *m, int rows, int cols)
{
	for (int j = 0; j<rows; ++j){
		for (int i = 0; i<cols; ++i){
			//printf("%.6f\t", m[i + j*cols]);
		}
		//printf("\n");
	}
}

__device__ void rowFirst_mul(double *A, double *B, int m, int n, int k, double *C)
{
	double sum;
	int ia, ib;
	for (int i = 0; i<m; ++i){
		for (int j = 0; j<n; ++j){
			sum = 0;
			for (int t = 0; t<k; ++t){
				ia = t + j*m;
				ib = i + t*k;
				sum += A[ia] * B[ib];
				//sum += A[j + t*m] * B[t + i*n];
			}
			C[i + j*m] = sum;
		}
	}
}

__device__ void rowFirst_mul_opt_94(double *Xi, double *A, double *B)
{
	double sum;
	int ia, ib;
	for (int j = 0; j<len9; ++j)
	{
		for (int i = 0; i<len9; ++i)
		{
			sum = 0;
			sum =
				Xi[j * len4 + 0] * Xi[i * len4 + 0] +
				Xi[j * len4 + 1] * Xi[i * len4 + 1] +
				Xi[j * len4 + 2] * Xi[i * len4 + 2] +
				Xi[j * len4 + 3] * Xi[i * len4 + 3];

			B[i + j*len9] = sum;
			A[i + j*len9] = sum;
		}
		A[j + j*len9] += lambda;
	}
	A[len99 - 1] -= lambda;
}

__device__ void rowFirst_lapcoeff(double *I_F, double *lapcoeff)
{
	double sum;
	int ia, ib;
	for (int j = 0; j<len9; ++j){
		for (int i = 0; i<len9; ++i){
			sum = 0;
			for (int t = 0; t<len9; ++t){
				ia = j + t*len9;
				ib = i + t*len9;
				sum += I_F[ia] * I_F[ib];
			}
			lapcoeff[i + j*len9] = sum;
		}
	}
}

__device__ void rowFirst_cal_I_F(double *F)
{
	for (int i = 0; i<len99; ++i)
		F[i] = -F[i];
	for (int i = 0; i<len9; ++i)
		F[i + i * len9] += 1;
}

//============================================================================//

//lu based solver
__device__ void Doolittle(int d, double*S, double*D){
	for (int k = 0; k<d; ++k){
		for (int j = k; j<d; ++j){
			double sum = 0.;
			for (int p = 0; p<k; ++p)sum += D[k*d + p] * D[p*d + j];
			D[k*d + j] = (S[k*d + j] - sum); // not dividing by diagonals
		}
		for (int i = k + 1; i<d; ++i){
			double sum = 0.;
			for (int p = 0; p<k; ++p)sum += D[i*d + p] * D[p*d + k];
			D[i*d + k] = (S[i*d + k] - sum) / D[k*d + k];
		}
	}
}
__device__ void solveDoolittle(int d, double*LU, double*b, double*x){
	double y[len9];
	for (int i = 0; i<d; ++i){
		double sum = 0.;
		for (int k = 0; k<i; ++k)sum += LU[i*d + k] * y[k];
		y[i] = (b[i] - sum); // not dividing by diagonals
	}
	for (int i = d - 1; i >= 0; --i){
		double sum = 0.;
		for (int k = i + 1; k<d; ++k)sum += LU[i*d + k] * x[k];
		x[i] = (y[i] - sum) / LU[i*d + i];
	}
}

__device__ void coutMatrix(int d, double*m){
	//printf("\n");
	for (int i = 0; i<d; ++i){
		for (int j = 0; j<d; ++j)
			//printf("%.4f", m[i*d + j]);
		//printf("\n");
	}
}
__device__ void coutVector(int d, double*v){
	//printf("\n");
	for (int j = 0; j<d; ++j)
		//printf("%.4f", v[j]);
	//printf("\n");
}


//=========================================================================================================================================================//

//helper function
__device__ void d_printD(double *data, int rows, int cols){
	//
	//printf("\n");

	for (int j = 0; j < rows; ++j)
	{
		for (int i = 0; i < cols; ++i)
		{
			//printf("%.3f\t", data[j + i*rows]);
		}
		//printf("\n");
	}
}
__device__ void d_printI(int *data, int rows, int cols){
	//
	//printf("\n");

	for (int j = 0; j < rows; ++j)
	{
		for (int i = 0; i < cols; ++i)
		{
			//printf("%d\t", data[j + i*rows]);
		}
		//printf("\n");
	}
}


__device__ void rowFirst_assignXiMatrix(double *imdata, double *Xi, int tid, int cols, int x, int y)
{
	Xi[0] = imdata[(tid - cols - 1) * 3 + 0];		Xi[1] = imdata[(tid - cols - 1) * 3 + 1];	Xi[2] = imdata[(tid - cols - 1) * 3 + 2];	Xi[3] = 1;
	Xi[4] = imdata[(tid - 1) * 3 + 0];				Xi[5] = imdata[(tid - 1) * 3 + 1];			Xi[6] = imdata[(tid - 1) * 3 + 2];			Xi[7] = 1;
	Xi[8] = imdata[(tid + cols - 1) * 3 + 0];		Xi[9] = imdata[(tid + cols - 1) * 3 + 1];	Xi[10] = imdata[(tid + cols - 1) * 3 + 2];	Xi[11] = 1;



	Xi[12] = imdata[(tid - cols + 0) * 3 + 0];		Xi[13] = imdata[(tid - cols + 0) * 3 + 1];	Xi[14] = imdata[(tid - cols + 0) * 3 + 2];	Xi[15] = 1;
	Xi[16] = imdata[(tid + 0) * 3 + 0];				Xi[17] = imdata[(tid + 0) * 3 + 1];			Xi[18] = imdata[(tid + 0) * 3 + 2];			Xi[19] = 1;
	Xi[20] = imdata[(tid + cols + 0) * 3 + 0];		Xi[21] = imdata[(tid + cols + 0) * 3 + 1];	Xi[22] = imdata[(tid + cols + 0) * 3 + 2];	Xi[23] = 1;



	Xi[24] = imdata[(tid - cols + 1) * 3 + 0];		Xi[25] = imdata[(tid - cols + 1) * 3 + 1];	Xi[26] = imdata[(tid - cols + 1) * 3 + 2];	Xi[27] = 1;
	Xi[28] = imdata[(tid + 1) * 3 + 0];				Xi[29] = imdata[(tid + 1) * 3 + 1];     	Xi[30] = imdata[(tid + 1) * 3 + 2];	        Xi[31] = 1;
	Xi[32] = imdata[(tid + cols + 1) * 3 + 0];		Xi[33] = imdata[(tid + cols + 1) * 3 + 1];	Xi[34] = imdata[(tid + cols + 1) * 3 + 2];	Xi[35] = 1;

}

__device__ void mapDeviceMem(
	double *d_Xi, double *d_A, double *d_B,
	int *d_trimap, int *d_row_inds, int *d_col_inds, double *d_vals,
 	//
	double *&Xi, double *&A, double *&B,
	int *&row_inds, int *&col_inds, double *&vals,
	int tid)
{
	Xi = &d_Xi[tid*len94];

	A = &d_A[tid*len99];
	B = &d_B[tid*len99];

	vals	 = &d_vals[tid*len99];
	row_inds = &d_row_inds[tid*len99];
	col_inds = &d_col_inds[tid*len99];
}


__device__ void compLapcoeff_new(double *Xi, double *A, double *B)
{

	rowFirst_mul_opt_94(Xi, A, B);
	Doolittle(len9, A, A);

	for (int i = 0; i < len9; ++i)
	{
		solveDoolittle(len9, A, &B[len9*i], &B[len9*i]);
	}

	rowFirst_cal_I_F(B);

	rowFirst_lapcoeff(B, A);

	for (int i = 0; i < len99; ++i)
		B[i] = A[i];

}

__global__ void kernel_new(double *d_Xi, double *d_A, double *d_B,
	double *d_imdata, int *d_trimap, int *d_row_inds, int *d_col_inds, double *d_vals,
	int rows, int cols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int tid = x + y * cols;

	////printf("in the kernel\n");

	if ((x <= 1) || (y <= 1) || (x >= (cols - 1) ) || (y >= (rows - 1) ) || d_trimap[tid])	return;	


	double *Xi, *A, *B;

	int *row_inds, *col_inds;
	double *vals;


	mapDeviceMem(
		d_Xi, d_A, d_B, 
		d_trimap, d_row_inds, d_col_inds, d_vals,
		Xi, A, B,
		row_inds, col_inds, vals,
		tid);

	rowFirst_assignXiMatrix(d_imdata, Xi, tid, cols, x, y);

	compLapcoeff_new(Xi, A, B);

	//
	int winInds[9];
	winInds[0] = (y - 1) +	(x - 1)*rows;	winInds[3] = (y - 1) + (x    )*rows;	winInds[6] = (y - 1) + (x + 1)*rows;
	winInds[1] = (y    ) +	(x - 1)*rows;	winInds[4] = (y    ) + (x    )*rows;	winInds[7] = (y    ) + (x + 1)*rows;
	winInds[2] = (y + 1) +	(x - 1)*rows;	winInds[5] = (y + 1) + (x    )*rows;	winInds[8] = (y + 1) + (x + 1)*rows;

	//row
	for (int j = 0; j < len9; ++j)
		for (int i = 0; i < len9; ++i)
			row_inds[j*len9 + i] = winInds[i];

	//col
	for (int j = 0; j < len9; ++j)
		for (int i = 0; i < len9; ++i)
			col_inds[j*len9 + i] = winInds[j];

	//vals
	for (int i = 0; i < len99; ++i)
		vals[i] = B[i];

}

//=========================================================================================================================================================//

//memory allocation and free
void allocMem(
	int pixNum,
	double *&d_Xi, double *&d_A, double *&d_B,
	double *&d_imdata, double *&d_vals, int *&d_trimap, int *&d_row_inds, int *&d_col_inds)
{
	//d_Xi
	checkCudaErrors(cudaMalloc((void**)&d_Xi, pixNum * len94 * sizeof(double)));

	//d_A
	checkCudaErrors(cudaMalloc((void**)&d_A, pixNum * len99 * sizeof(double)));

	//d_B
	checkCudaErrors(cudaMalloc((void**)&d_B, pixNum * len99 * sizeof(double)));

	//d_imdata
	checkCudaErrors(cudaMalloc((void**)&d_imdata, pixNum * channel * sizeof(double)));

	//d_trimap
	checkCudaErrors(cudaMalloc((void**)&d_trimap, pixNum * sizeof(int)));

	//d_vals
	checkCudaErrors(cudaMalloc((void**)&d_vals, pixNum * len99 * sizeof(double)));
	
	//d_row_inds, d_col_inds
	checkCudaErrors(cudaMalloc((void**)&d_row_inds, pixNum * len99 * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_col_inds, pixNum * len99 * sizeof(int)));
}

void freeMem(double *&d_Xi, double *&d_A, double *&d_B,
	double *&d_imdata, double *&d_vals, int *&d_trimap, int *&d_row_inds, int *&d_col_inds)
{
	cudaFree(d_Xi);
	cudaFree(d_A);
	cudaFree(d_B);
	
	//
	cudaFree(d_imdata);
	cudaFree(d_vals);
	cudaFree(d_trimap);
	cudaFree(d_row_inds);
	cudaFree(d_col_inds);

}

void printMatrix(double* m, int rows, int cols){
	//
	cout.precision(3);
	cout << endl;
	for (int j = 0; j < rows; ++j){
		for (int i = 0; i < cols; ++i)
		{
			cout << m[j + i*rows] << "\t";
		}
		cout << endl;
	}
}

//=========================================================================================================================================================//

//interface
extern "C" void callKernel_step1(int rows, int cols, double *imdata, int *trimap, int *row_inds, int *col_inds, double *vals)
{
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	
	int pixNum = rows * cols;

	double *d_Xi, *d_A, *d_B;

	double *d_imdata, *d_vals;
	int *d_trimap, *d_row_inds, *d_col_inds;

	//
	allocMem(pixNum, 
		d_Xi, d_A, d_B,
		d_imdata, d_vals, d_trimap, d_row_inds, d_col_inds);

	checkCudaErrors(cudaMemcpy(d_imdata, imdata, pixNum * channel *sizeof(double), cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy(d_trimap, trimap, pixNum * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(d_row_inds, 0, pixNum * len99 * sizeof(int)));
	checkCudaErrors(cudaMemset(d_col_inds, 0, pixNum * len99 * sizeof(int)));
	checkCudaErrors(cudaMemset(d_vals,	   0, pixNum * len99 * sizeof(double)));
	

	//dim3 block(16, 16);
	//dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);


	checkCudaErrors(cudaEventRecord(start, 0));
	//kernel_new<<<grid, block>>>(d_Xi, d_A, d_B,
	//	d_imdata, d_trimap, d_row_inds, d_col_inds, d_vals,
	//	rows, cols);


	dim3 grid(1024, 1024);
	kernel_new<<<grid, 1>>>(d_Xi, d_A, d_B,
		d_imdata, d_trimap, d_row_inds, d_col_inds, d_vals,
		rows, cols);

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float timeCost = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&timeCost, start, stop));
	//printf("time consumption on GPU: %f\n", timeCost);
	
	checkCudaErrors(cudaMemcpy(row_inds, d_row_inds, pixNum * len99 * sizeof(int),		cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(col_inds, d_col_inds, pixNum * len99 * sizeof(int),		cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vals,     d_vals,     pixNum * len99 * sizeof(double),	cudaMemcpyDeviceToHost));


	freeMem(d_Xi, d_A, d_B,
		d_imdata, d_vals, d_trimap, d_row_inds, d_col_inds);
		
	
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	return ;

}

extern "C" void callKernel_step2(int *L_rows, int *L_cols, double *L_vals, double *alpha_star, double *res, int m, int nnz)
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	cusparseStatus_t status;

	double *d_csrVal;
	int *d_csrRowPtr, *d_csrColInd;

	double *h_x, *h_y, *h_z,
		   *d_x, *d_y, *d_z;

	cusparseMatDescr_t descr_M = 0;
	cusparseMatDescr_t descr_L = 0;
	cusparseMatDescr_t descr_U = 0;
	csrilu02Info_t	info_M = 0;
	csrsv2Info_t	info_L = 0;
	csrsv2Info_t	info_U = 0;
	int pBufferSize_M;
	int pBufferSize_L;
	int pBufferSize_U;
	int pBufferSize;
	char *pBuffer = 0;
	int structural_zero;
	int numerical_zero;
	const double alpha = 1.;
	const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
	const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
	
	//=====================================================================================================//

	//matrix device
	int * csrRowPtr = 0;
	int * cooRowIndex = 0;
	int * cooColIndex = 0;
	double * cooVal = 0;

	/* allocate GPU memory and copy the matrix and vectors into it */
	checkCudaErrors(cudaMalloc((void**)&cooRowIndex, nnz*sizeof(cooRowIndex[0])));
	checkCudaErrors(cudaMalloc((void**)&cooColIndex, nnz*sizeof(cooColIndex[0])));
	checkCudaErrors(cudaMalloc((void**)&cooVal, nnz*sizeof(cooVal[0])));

	checkCudaErrors(cudaMalloc((void**)&d_x, m * sizeof(d_x[0])));
	checkCudaErrors(cudaMalloc((void**)&d_y, m * sizeof(d_y[0])));
	checkCudaErrors(cudaMalloc((void**)&d_z, m * sizeof(d_z[0])));
	//printf("Device malloc succeeded\n");

	//copy stage
	checkCudaErrors(cudaMemcpy(cooRowIndex, L_rows, nnz*sizeof(L_rows[0]), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(cooColIndex, L_cols, nnz*sizeof(L_cols[0]), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(cooVal, L_vals, nnz*sizeof(L_vals[0]), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMemcpy(d_x, alpha_star, m * sizeof(d_x[0]), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_y, h_y, m * sizeof(d_y[0]), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_z, h_z, m * sizeof(d_z[0]), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_y, 0, m * sizeof(d_y[0])));
	checkCudaErrors(cudaMemset(d_z, 0, m * sizeof(d_z[0])));
	//printf("Memcpy from Host to Device succeeded\n");

	//change matrix format
	checkCudaErrors(cudaMalloc((void**)&csrRowPtr, (m+1) * sizeof(csrRowPtr[0])));
	checkCudaErrors(cusparseXcoo2csr(handle, cooRowIndex, nnz, m+1, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));
	//printf("Conversion from COO to CSR format succeeded\n");

	cudaDeviceSynchronize();


	////printf("the value is~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~:\n");
	//for (int i = 0; i < nnnz; ++i)
	//{
	//	////printf("i: %d row: %d\tcolumn: %d\tvalue:%f\n", i, L_rows[i], L_cols[i], L_vals[i]);
	//	//printf("%d\t%d\t%f\n", L_rows[i], L_cols[i], L_vals[i]);
	//}

	////printf("matrix size: %d\n", m);
	//for (int i = 0; i < m; ++i)
	//{
	//	////printf("i: %d as value:%f\n", i, alpha_star[i]);
	//	//printf("%f\n", alpha_star[i]);
	//}
	
	d_csrVal = cooVal;
	d_csrRowPtr = csrRowPtr;
	d_csrColInd = cooColIndex;

	//=====================================================================================================//

	cusparseCreateMatDescr(&descr_M);
	cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);


	cusparseCreateMatDescr(&descr_L);
	cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

	cusparseCreateMatDescr(&descr_U);
	cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

	// step 2: create a empty info structure 
	// we need one info for csrilu02 and two info's for csrsv2
	cusparseCreateCsrilu02Info(&info_M);
	cusparseCreateCsrsv2Info(&info_L);
	cusparseCreateCsrsv2Info(&info_U);

	// step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer 
	cusparseDcsrilu02_bufferSize(handle, m, nnz, descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M);
	cusparseDcsrsv2_bufferSize(handle, trans_L, m, nnz, descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, &pBufferSize_L);
	cusparseDcsrsv2_bufferSize(handle, trans_U, m, nnz, descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, &pBufferSize_U);
	pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
	// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes. 
	cudaMalloc((void**)&pBuffer, pBufferSize);

	// step 4: perform analysis of incomplete Cholesky on M 
	// perform analysis of triangular solve on L 
	// perform analysis of triangular solve on U
	// The lower(upper) triangular part of M has the same sparsity pattern as L(U), 
	// we can do analysis of csrilu0 and csrsv2 simultaneously.
	checkCudaErrors(cusparseDcsrilu02_analysis(handle, m, nnz, descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer));
	status = cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){
		//printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
	}
	cusparseDcsrsv2_analysis(handle, trans_L, m, nnz, descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, policy_L, pBuffer);
	cusparseDcsrsv2_analysis(handle, trans_U, m, nnz, descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, policy_U, pBuffer);

	// step 5: M = L * U
	//printf("begin to M = L * U\n");
	cusparseDcsrilu02(handle, m, nnz, descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);
	status = cusparseXcsrilu02_zeroPivot(handle, info_M, &numerical_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){
		//printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
	}

	// step 6: solve L*z = x 
	cusparseDcsrsv2_solve(handle, trans_L,
		m, nnz,
		&alpha, descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
		d_x, d_z, policy_L, pBuffer);

	// step 7: solve U*y = z 
	cusparseDcsrsv2_solve(handle, trans_U,
		m, nnz,
		&alpha, descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U,
		d_z, d_y, policy_U, pBuffer);

	checkCudaErrors(cudaMemcpy(res, d_y, m*sizeof(h_y[0]), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(h_y, d_y, m*sizeof(h_y[0]), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < m; ++i)
	//	//printf("%f\t", h_y[i]);

	// step 6: free resources 
	cudaFree(pBuffer);
	cusparseDestroyMatDescr(descr_M);
	cusparseDestroyMatDescr(descr_L);
	cusparseDestroyMatDescr(descr_U);
	cusparseDestroyCsrilu02Info(info_M);
	cusparseDestroyCsrsv2Info(info_L);
	cusparseDestroyCsrsv2Info(info_U);
	cusparseDestroy(handle);
}

extern "C" void callKernel_step2_iterative(int *L_rows, int *L_cols, double *L_vals, double *alpha_star, double *res, int m, int nnz)
{

	/* Create CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	checkCudaErrors( cublasCreate(&cublasHandle));

	/* Create CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	checkCudaErrors( cusparseCreate(&cusparseHandle));

	/* Description of the A matrix*/
	cusparseMatDescr_t descr = 0;
	checkCudaErrors( cusparseCreateMatDescr(&descr));

	/* Define the properties of the matrix */
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	//=====================================================================================================//
	//convert from coo format to csr format

	//matrix device
	int * h_csrRowPtr = 0;
	int * csrRowPtr = 0;
	int * cooRowIndex = 0;

	h_csrRowPtr = (int *)malloc((m+1)*sizeof(int));

	/* allocate GPU memory and copy the matrix and vectors into it */
	checkCudaErrors(cudaMalloc((void**)&cooRowIndex, nnz*sizeof(cooRowIndex[0])));
	//printf("Device malloc succeeded\n");

	//copy stage
	checkCudaErrors(cudaMemcpy(cooRowIndex, L_rows, nnz*sizeof(L_rows[0]), cudaMemcpyHostToDevice));
	//printf("Memcpy from Host to Device succeeded\n");

	//change matrix format
	checkCudaErrors(cudaMalloc((void**)&csrRowPtr, (m + 1) * sizeof(csrRowPtr[0])));
	checkCudaErrors(cusparseXcoo2csr(cusparseHandle, cooRowIndex, nnz, m + 1, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));
	//printf("Conversion from COO to CSR format succeeded\n");

	checkCudaErrors(cudaMemcpy(h_csrRowPtr, csrRowPtr, (m + 1) * sizeof(csrRowPtr[0]), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	//===========================================================================================================//

	const int max_iter = 100;		//max iteraton times
	int k, *I = NULL, *J = NULL;
	int *d_col, *d_row;
	const double precision = 1e-10f;
	double *x, *rhs;
	double r0, r1, alpha, beta;
	double *d_val, *d_x;
	double *d_r, *d_p, *d_omega, *d_y;
	double *val = NULL;
	double rsum, diff, err = 0.0;
	double qaerr1, qaerr2 = 0.0;
	double dot, numerator, denominator, nalpha;
	const double doubleone = 1.0;
	const double doublezero = 0.0;

	int nErrors = 0;

	//printf("conjugateGradientPrecond starting...\n");

	/* Generate a random tridiagonal symmetric matrix in CSR (Compressed Sparse Row) format */

	I = (int *)malloc(sizeof(int)*(m + 1));                              // csr row pointers for matrix A
	J = (int *)malloc(sizeof(int)*nnz);                                 // csr column indices for matrix A
	val = (double *)malloc(sizeof(double)*nnz);                           // csr values for matrix A
	x = (double *)malloc(sizeof(double)*m);
	rhs = (double *)malloc(sizeof(double)*m);

	for (int i = 0; i < m; i++)
	{
		rhs[i] = 0.0;                                                  // Initialize RHS
		x[i] = 0.0;                                                    // Initial approximation of solution
	}

	//
	memcpy(I, h_csrRowPtr, (m + 1)*sizeof(int));
	memcpy(J, L_cols, nnz * sizeof(int));
	memcpy(val, L_vals, nnz * sizeof(double));
	memcpy(rhs, alpha_star, m * sizeof(double));
	
	//memcpy(I, h_csrRowPtr, (m+1)*sizeof(int));

	////I = csrRowPtr;
	//J = L_cols;
	//val = L_vals;

	//rhs = alpha_star;
	//
	
	/* Allocate required memory */
	checkCudaErrors(cudaMalloc((void **)&d_col, nnz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (m + 1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nnz*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_x, m*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_y, m*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_r, m*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_p, m*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_omega, m*sizeof(double)));

	cudaMemcpy(d_col, J, nnz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (m + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nnz*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, m*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, m*sizeof(double), cudaMemcpyHostToDevice);

	/* Conjugate gradient without preconditioning.
	------------------------------------------
	Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Section 10.2.6  */

	//printf("Convergence of conjugate gradient without preconditioning: \n");
	k = 0;
	r0 = 0;
	cublasDdot(cublasHandle, m, d_r, 1, d_r, 1, &r1);

	while (r1 > precision*precision && k <= max_iter)
	{
		k++;

		if (k == 1)
		{
			cublasDcopy(cublasHandle, m, d_r, 1, d_p, 1);
		}
		else
		{
			beta = r1 / r0;
			cublasDscal(cublasHandle, m, &beta, d_p, 1);
			cublasDaxpy(cublasHandle, m, &doubleone, d_r, 1, d_p, 1);
		}

		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, m, nnz, &doubleone, descr, d_val, d_row, d_col, d_p, &doublezero, d_omega);
		cublasDdot(cublasHandle, m, d_p, 1, d_omega, 1, &dot);
		alpha = r1 / dot;
		cublasDaxpy(cublasHandle, m, &alpha, d_p, 1, d_x, 1);
		nalpha = -alpha;
		cublasDaxpy(cublasHandle, m, &nalpha, d_omega, 1, d_r, 1);
		r0 = r1;
		cublasDdot(cublasHandle, m, d_r, 1, d_r, 1, &r1);
	}

	//printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

	cudaMemcpy(x, d_x, m*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(res, d_x, m*sizeof(double), cudaMemcpyDeviceToHost);

	/* check result */
	err = 0.0;

	for (int i = 0; i < m; i++)
	{
		rsum = 0.0;

		for (int j = I[i]; j < I[i + 1]; j++)
		{
			rsum += val[j] * x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err)
		{
			err = diff;
		}
	}

	//printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
	nErrors += (k > max_iter) ? 1 : 0;
	qaerr1 = err;


	/* Destroy contexts */
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	/* Free device memory */
	free(h_csrRowPtr);
	//free(I);
	//free(J);
	//free(val);
	//free(x);
	//free(rhs);
	//free(valsILU0);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_omega);

	cudaDeviceReset();

	//printf("  Test Summary:\n");
	//printf("     Counted total of %d errors\n", nErrors);
	//printf("     qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2));

}

extern "C" void callKernel_step2_iterative_lu(int *L_rows, int *L_cols, double *L_vals, double *alpha_star, double *res, int m, int nnz)
{

	/* Create CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	checkCudaErrors(cublasCreate(&cublasHandle));

	/* Create CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	checkCudaErrors(cusparseCreate(&cusparseHandle));

	/* Description of the A matrix*/
	cusparseMatDescr_t descr = 0;
	checkCudaErrors(cusparseCreateMatDescr(&descr));

	/* Define the properties of the matrix */
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	//=====================================================================================================//
	//convert from coo format to csr format

	//matrix device
	int * h_csrRowPtr = 0;
	int * csrRowPtr = 0;
	int * cooRowIndex = 0;

	h_csrRowPtr = (int *)malloc((m + 1)*sizeof(int));

	/* allocate GPU memory and copy the matrix and vectors into it */
	checkCudaErrors(cudaMalloc((void**)&cooRowIndex, nnz*sizeof(cooRowIndex[0])));
	//printf("Device malloc succeeded\n");

	//copy stage
	checkCudaErrors(cudaMemcpy(cooRowIndex, L_rows, nnz*sizeof(L_rows[0]), cudaMemcpyHostToDevice));
	//printf("Memcpy from Host to Device succeeded\n");

	//change matrix format
	checkCudaErrors(cudaMalloc((void**)&csrRowPtr, (m + 1) * sizeof(csrRowPtr[0])));
	checkCudaErrors(cusparseXcoo2csr(cusparseHandle, cooRowIndex, nnz, m + 1, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));
	//printf("Conversion from COO to CSR format succeeded\n");

	checkCudaErrors(cudaMemcpy(h_csrRowPtr, csrRowPtr, (m + 1) * sizeof(csrRowPtr[0]), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	//===========================================================================================================//

	const int max_iter = 500;
	int k, *I = NULL, *J = NULL;
	int *d_col, *d_row;
	int qatest = 0;
	const double tol = 1e-1f;
	double *x, *rhs;
	double r0, r1, alpha, beta;
	double *d_val, *d_x;
	double *d_zm1, *d_zm2, *d_rm2;
	double *d_r, *d_p, *d_omega, *d_y;
	double *val = NULL;
	double *d_valsILU0;
	double rsum, diff, err = 0.0;
	double qaerr1, qaerr2 = 0.0;
	double dot, numerator, denominator, nalpha;
	const double doubleone = 1.0;
	const double doublezero = 0.0;

	int nErrors = 0;

	//printf("conjugateGradientPrecond starting...\n");


	I = (int *)malloc(sizeof(int)*(m + 1));                              // csr row pointers for matrix A
	J = (int *)malloc(sizeof(int)*nnz);                                 // csr column indices for matrix A
	val = (double *)malloc(sizeof(double)*nnz);                           // csr values for matrix A
	x = (double *)malloc(sizeof(double)*m);
	rhs = (double *)malloc(sizeof(double)*m);

	for (int i = 0; i < m; i++)
	{
		rhs[i] = 0.0;                                                  // Initialize RHS
		x[i] = 0.0;                                                    // Initial approximation of solution
	}

	//prepareData(I, J, val, M, N, nnz, rhs);
	//printf("data assignment\n");
	
	//I = csrRowPtr;
	//J = L_cols;
	//val = L_vals;
	//rhs = alpha_star;
	memcpy(I, h_csrRowPtr, (m + 1)*sizeof(int));
	memcpy(J, L_cols, nnz * sizeof(int));
	memcpy(val, L_vals, nnz * sizeof(double));
	memcpy(rhs, alpha_star, m * sizeof(double));
	


	/* Allocate required memory */
	checkCudaErrors(cudaMalloc((void **)&d_col, nnz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (m + 1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nnz*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_x, m*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_y, m*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_r, m*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_p, m*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_omega, m*sizeof(double)));

	cudaMemcpy(d_col, J, nnz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (m + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nnz*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, m*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, m*sizeof(double), cudaMemcpyHostToDevice);
	

	/* Preconditioned Conjugate Gradient using ILU.
	--------------------------------------------
	Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Algorithm 10.3.1  */

	//printf("\nConvergence of conjugate gradient using incomplete LU preconditioning: \n");

	int nnzILU0 = nnz;
	//int nnzILU0 = 2*N-1;
	//valsILU0 = (double *) malloc(nnz*sizeof(double));

	checkCudaErrors(cudaMalloc((void **)&d_valsILU0, nnz*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_zm1, (m)*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_zm2, (m)*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_rm2, (m)*sizeof(double)));

	/* create the analysis info object for the A matrix */
	cusparseSolveAnalysisInfo_t infoA = 0;
	cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);

	checkCudaErrors(cusparseStatus);

	/* Perform the analysis for the Non-Transpose case */
	cusparseStatus = cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		m, nnz, descr, d_val, d_row, d_col, infoA);

	checkCudaErrors(cusparseStatus);

	/* Copy A data to ILU0 vals as input*/
	cudaMemcpy(d_valsILU0, d_val, nnz*sizeof(double), cudaMemcpyDeviceToDevice);

	/* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */
	cusparseStatus = cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, descr, d_valsILU0, d_row, d_col, infoA);

	checkCudaErrors(cusparseStatus);

	/* Create info objects for the ILU0 preconditioner */
	cusparseSolveAnalysisInfo_t info_u;
	cusparseCreateSolveAnalysisInfo(&info_u);

	cusparseMatDescr_t descrL = 0;
	cusparseStatus = cusparseCreateMatDescr(&descrL);
	cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);

	cusparseMatDescr_t descrU = 0;
	cusparseStatus = cusparseCreateMatDescr(&descrU);
	cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);

	cusparseStatus = cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnz, descrU, d_val, d_row, d_col, info_u);

	/* reset the initial guess of the solution to zero */
	for (int i = 0; i < m; i++)
	{
		x[i] = 0.0;
	}

	checkCudaErrors(cudaMemcpy(d_r, rhs, m*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_x, x, m*sizeof(double), cudaMemcpyHostToDevice));

	k = 0;
	cublasDdot(cublasHandle, m, d_r, 1, d_r, 1, &r1);

	while (r1 > tol*tol && k <= max_iter)
	{
		// Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
		cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, &doubleone, descrL,
			d_valsILU0, d_row, d_col, infoA, d_r, d_y);
		checkCudaErrors(cusparseStatus);

		// Back Substitution
		cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, &doubleone, descrU,
			d_valsILU0, d_row, d_col, info_u, d_y, d_zm1);
		checkCudaErrors(cusparseStatus);

		k++;

		if (k == 1)
		{
			cublasDcopy(cublasHandle, m, d_zm1, 1, d_p, 1);
		}
		else
		{
			cublasDdot(cublasHandle, m, d_r, 1, d_zm1, 1, &numerator);
			cublasDdot(cublasHandle, m, d_rm2, 1, d_zm2, 1, &denominator);
			beta = numerator / denominator;
			cublasDscal(cublasHandle, m, &beta, d_p, 1);
			cublasDaxpy(cublasHandle, m, &doubleone, d_zm1, 1, d_p, 1);
		}

		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, m, nnzILU0, &doubleone, descrU, d_val, d_row, d_col, d_p, &doublezero, d_omega);
		cublasDdot(cublasHandle, m, d_r, 1, d_zm1, 1, &numerator);
		cublasDdot(cublasHandle, m, d_p, 1, d_omega, 1, &denominator);
		alpha = numerator / denominator;
		cublasDaxpy(cublasHandle, m, &alpha, d_p, 1, d_x, 1);
		cublasDcopy(cublasHandle, m, d_r, 1, d_rm2, 1);
		cublasDcopy(cublasHandle, m, d_zm1, 1, d_zm2, 1);
		nalpha = -alpha;
		cublasDaxpy(cublasHandle, m, &nalpha, d_omega, 1, d_r, 1);
		cublasDdot(cublasHandle, m, d_r, 1, d_r, 1, &r1);
	}

	//printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

	cudaMemcpy(x, d_x, m*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(res, d_x, m*sizeof(double), cudaMemcpyDeviceToHost);
	/* check result */
	err = 0.0;

	for (int i = 0; i < m; i++)
	{
		rsum = 0.0;

		for (int j = I[i]; j < I[i + 1]; j++)
		{
			rsum += val[j] * x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err)
		{
			err = diff;
		}
	}

	//printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
	nErrors += (k > max_iter) ? 1 : 0;
	qaerr2 = err;

	/* Destroy paramters */
	cusparseDestroySolveAnalysisInfo(infoA);
	cusparseDestroySolveAnalysisInfo(info_u);

	/* Destroy contexts */
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	/* Free device memory */
	free(I);
	free(J);
	free(val);
	free(x);
	free(rhs);
	//free(valsILU0);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_omega);
	cudaFree(d_valsILU0);
	cudaFree(d_zm1);
	cudaFree(d_zm2);
	cudaFree(d_rm2);

	cudaDeviceReset();

	//printf("  Test Summary:\n");
	//printf("     Counted total of %d errors\n", nErrors);
	//printf("     qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2));
	//exit((nErrors == 0 && fabs(qaerr1)<1e-5 && fabs(qaerr2) < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE));

}

//