#include <stdio.h>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <helper_string.h>
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace std;

const int len1 = 1;
const int len3 = 3;
const int len4 = 4;
const int len9 = 9;
const int len93 = 27;
const int len94 = 36;
const int len99 = 81;

int channel = 3;

//const int effThreads = 1000*900;
//const int effThreads = 800 * 600;
//const int effThreads = 1;
//const int effThreads = 420000;

//=========================================================================================================================================================//

__device__ void d_printD(float *data, int rows, int cols){
	//
	printf("\n");

	for (int j = 0; j < rows; ++j)
	{
		for (int i = 0; i < cols; ++i)
		{
			printf("%.3f\t", data[j + i*rows]);
		}
		printf("\n");
	}
}
__device__ void d_printI(int *data, int rows, int cols){
	//
	printf("\n");

	for (int j = 0; j < rows; ++j)
	{
		for (int i = 0; i < cols; ++i)
		{
			printf("%d\t", data[j + i*rows]);
		}
		printf("\n");
	}
}

__device__ void makePivotMatrix(float *pivot_arr, int *pivot_vec)
{
	int pc[len9];

	//
	for (int i = 0; i < len9; ++i)
		pc[i] = i;

	int tmp;
	for (int i = 0; i < len9 - 1; ++i)
	{
		int j = pivot_vec[i] - 1;

		tmp = pc[i];
		pc[i] = pc[j];
		pc[j] = tmp;
	}

	for (int i = 0; i < len99; ++i)
	{
		pivot_arr[i] = 0.f;
	}

	for (int row = 0; row < len9; ++row)
	{
		int col = pc[row];
		pivot_arr[col*len9 + row] = 1.f;
	}


}

__device__ void assignLUMatrix(float *A, float *L, float *U)
{
	//获取下三角矩阵L
	//L[0] = 1.0f;	L[3] = 0.0f;	L[6] = 0.0f;
	//L[1] = A[1];	L[4] = 1.0f;	L[7] = 0.0f;
	//L[2] = A[2];	L[5] = A[5];	L[8] = 1.0f;

	// 行循环
	for (int j = 0; j < len9; ++j)
	{
		for (int i = 0; i < len9; ++i)
		{
			if (j>i)	//行号较大，处于下方，赋值A【】
				L[i*len9 + j] = A[i*len9 + j];
			else if (i == j)
				L[i*len9 + j] = 1.f;
			else
				L[i*len9 + j] = 0.f;
		}
	}


	//获取上三角矩阵U
	//U[0] = A[0];	U[3] = A[3];	U[6] = A[6];
	//U[1] = 0.0f;	U[4] = A[4];	U[7] = A[7];
	//U[2] = 0.0f;	U[5] = 0.0f;	U[8] = A[8];

	for (int j = 0; j < len9; ++j)
	{
		for (int i = 0; i < len9; ++i)
		{
			if (j <= i)	//行号较小，处于上方，赋值A【】
				U[i*len9 + j] = A[i*len9 + j];
			else
				U[i*len9 + j] = 0.f;
		}
	}
	//printf("\nA: \n %f  %f  %f  \n%f  %f  %f  \n%f  %f  %f  \n", A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]);

	//printf("\nL: \n %f  %f  %f  \n%f  %f  %f  \n%f  %f  %f  \n", L[0], L[3], L[6], L[1], L[4], L[7], L[2], L[5], L[8]);

	//printf("\nU: \n %f  %f  %f  \n%f  %f  %f  \n%f  %f  %f  \n", U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8]);

}

__device__ void assignXiMatrix(float *imdata, float *Xi, int tid, int cols, int x, int y )
{
	//int shift1[len3*len3] = 
	//{
	//	-cols-1,	-cols,	-cols+1,
	//		 -1,		0,		 +1,
	//	+cols-1,	+cols,	+cols+1
	//};

	//int shift2[2] = 
	//{
	//	9, 18
	//};

	//Xi[0] = imdata[tid * 3 - cols * 3 - 1];		Xi[9] = imdata[tid * 3 - cols * 3 + 0];		Xi[18] = imdata[tid * 3 - cols * 3 + 1];	Xi[27] = 1;
	//Xi[1] = imdata[tid * 3            - 1];		Xi[10] = imdata[tid * 3           + 0];		Xi[19] = imdata[tid * 3            + 1];	Xi[28] = 1;
	//Xi[2] = imdata[tid * 3 + cols * 3 - 1];		Xi[11] = imdata[tid * 3 + cols * 3 + 0];	Xi[20] = imdata[tid * 3 + cols * 3 + 1];	Xi[29] = 1;
	//


	//Xi[3] = imdata[tid * 3 - cols * 3 + 0];		Xi[12] = imdata[tid * 3 - cols * 3 + 1];	Xi[21] = imdata[tid * 3 - cols * 3 + 2];	Xi[30] = 1;
	//Xi[4] = imdata[tid * 3            + 0];		Xi[13] = imdata[tid * 3            + 1];	Xi[22] = imdata[tid * 3            + 2];	Xi[31] = 1;
	//Xi[5] = imdata[tid * 3 + cols * 3 + 0];		Xi[14] = imdata[tid * 3 + cols * 3 + 1];	Xi[23] = imdata[tid * 3 + cols * 3 + 2];	Xi[32] = 1;
	//


	//Xi[6] = imdata[tid * 3 - cols * 3 + 1];		Xi[15] = imdata[tid * 3 - cols * 3 + 2];	Xi[24] = imdata[tid * 3 - cols * 3 + 3];	Xi[33] = 1;
	//Xi[7] = imdata[tid * 3            + 1];		Xi[16] = imdata[tid * 3            + 2];	Xi[25] = imdata[tid * 3            + 3];	Xi[34] = 1;
	//Xi[8] = imdata[tid * 3 + cols * 3 + 1];		Xi[17] = imdata[tid * 3 + cols * 3 + 2];	Xi[26] = imdata[tid * 3 + cols * 3 + 3];	Xi[35] = 1;




	Xi[0] = imdata[(tid - cols - 1) * 3 + 0];		Xi[9]  = imdata[(tid - cols - 1) * 3 + 1];	Xi[18] = imdata[(tid - cols - 1) * 3 + 2];	Xi[27] = 1;
	Xi[1] = imdata[(tid        - 1) * 3 + 0];		Xi[10] = imdata[(tid        - 1) * 3 + 1];	Xi[19] = imdata[(tid        - 1) * 3 + 2];	Xi[28] = 1;
	Xi[2] = imdata[(tid + cols - 1) * 3 + 0];		Xi[11] = imdata[(tid + cols - 1) * 3 + 1];	Xi[20] = imdata[(tid + cols - 1) * 3 + 2];	Xi[29] = 1;



	Xi[3] = imdata[(tid - cols + 0) * 3 + 0];		Xi[12] = imdata[(tid - cols + 0) * 3 + 1];	Xi[21] = imdata[(tid - cols + 0) * 3 + 2];	Xi[30] = 1;
	Xi[4] = imdata[(tid        + 0) * 3 + 0];		Xi[13] = imdata[(tid        + 0) * 3 + 1];	Xi[22] = imdata[(tid        + 0) * 3 + 2];	Xi[31] = 1;
	Xi[5] = imdata[(tid + cols + 0) * 3 + 0];		Xi[14] = imdata[(tid + cols + 0) * 3 + 1];	Xi[23] = imdata[(tid + cols + 0) * 3 + 2];	Xi[32] = 1;



	Xi[6] = imdata[(tid - cols + 1) * 3 + 0];		Xi[15] = imdata[(tid - cols + 1) * 3 + 1];	Xi[24] = imdata[(tid - cols + 1) * 3 + 2];	Xi[33] = 1;
	Xi[7] = imdata[(tid        + 1) * 3 + 0];		Xi[16] = imdata[(tid        + 1) * 3 + 1];	Xi[25] = imdata[(tid        + 1) * 3 + 2];	Xi[34] = 1;
	Xi[8] = imdata[(tid + cols + 1) * 3 + 0];		Xi[17] = imdata[(tid + cols + 1) * 3 + 1];	Xi[26] = imdata[(tid + cols + 1) * 3 + 2];	Xi[35] = 1;


	//if (tid != 121)
	//	return;

	//printf("\n position: x:%d y:%d\n", x, y);
	//d_printD(Xi, 9, 4);

}

__device__ void solve(float **A, float *B, int *pivot_vec, float *pivot_arr, int *info, float *L, float *U)
{
	float alpha = 1.0f;
	float beta = 0.0f;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasStatus_t status;

	//lu分解
	cublasSgetrfBatched(handle,
		len9, &A[0],
		len9, pivot_vec,
		info, 1);

	cudaDeviceSynchronize();

	//还原枢纽矩阵
	makePivotMatrix(pivot_arr, pivot_vec);

	////获取L、U矩阵
	assignLUMatrix(A[0], L, U);

	//计算B = PB
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		len9, len9, len9,
		&alpha,
		pivot_arr, len9,
		B, len9,
		&beta,
		B, len9);

	//cudaDeviceSynchronize();

	// 求解LY = B, 注意到Y存储于B矩阵
	cublasStrsm(handle,
		cublasSideMode_t::CUBLAS_SIDE_LEFT,
		cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
		cublasOperation_t::CUBLAS_OP_N,
		cublasDiagType_t::CUBLAS_DIAG_UNIT,
		len9, len9,
		&alpha,
		L, len9,
		B, len9);

	//cudaDeviceSynchronize();

	// 求解UX = Y, 注意到X存储于Y矩阵,即B矩阵
	cublasStrsm(handle,
		cublasSideMode_t::CUBLAS_SIDE_LEFT,
		cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
		cublasOperation_t::CUBLAS_OP_N,
		cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
		len9, len9,
		&alpha,
		U, len9,
		B, len9);

	//cudaDeviceSynchronize();

	// 最后，结果B要做一次转置！！！！！！！！
	// B = B^T + 0
	alpha = 0.0f;
	beta = 1.0f;
	status = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T,
		len9, len9,
		&alpha,
		A[0], len9,
		&beta, B, len9,
		A[0], len9);

	//if (status != CUBLAS_STATUS_SUCCESS)
	//	printf("transpose 1 error! error code %d\n", status);
	//else
	//	printf("transpose 1 success!\n");

	alpha = 0.0f;
	beta = 1.0f;
	status = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		len9, len9,
		&alpha,
		B, len9,
		&beta, A[0], len9,
		B, len9);

	//if (status != CUBLAS_STATUS_SUCCESS)
	//	printf("transpose 2 error! error code %d\n", status);
	//else
	//	printf("transpose 2 success!\n");


	cublasDestroy(handle);

}

__device__ void mapDeviceMem(
	float *d_Xi, float *d_A, float **d_pA, float *d_B, int *d_pivot_vec, float *d_pivot_arr, int *d_info, float *d_L, float *d_U, 
	int *d_trimap, int *d_row_inds, int *d_col_inds, float *d_vals,
 	//
	float *&Xi, float **&A, float *&B, int *&pivot_vec, float *&pivot_arr, int *&info, float *&L, float *&U,
	int *&row_inds, int *&col_inds, float *&vals,
	int tid)
{
	Xi = &d_Xi[tid*len94];

	A = &d_pA[tid];
	d_pA[tid] = &d_A[tid*len99];

	B = &d_B[tid*len99];

	pivot_vec	= &d_pivot_vec[tid*len99];
	pivot_arr	= &d_pivot_arr[tid*len99];
	info		= &d_info[tid*len99];
	L	= &d_L[tid*len99];
	U	= &d_U[tid*len99];

	vals	 = &d_vals[tid*len99];
	row_inds = &d_row_inds[tid*len99];
	col_inds = &d_col_inds[tid*len99];
}

__device__ void createDiagnal(float *I, float val, int len)
{
	unsigned sz = sizeof(float)* len * len;

	for (int i = 0; i < len*len; ++i)
		I[i] = 0.f;

	for (int j = 0; j < len; ++j)
		I[j*len + j] = val;
}

__device__ void compLapcoeff(float *Xi, float **A, float *B, int *pivot_vec, float *pivot_arr, int *info, float *L, float *U){
	//
	float lambda = 0.000001f;

	float alpha = 1.0f;
	float beta = 0.0f;

	cublasHandle_t handle;
	cublasCreate(&handle);

	// A = lambda * I
	createDiagnal(A[0], lambda, len9);

	// 右下角元素置0
	A[0][len99 - 1] = 0;

	// B = Xi * Xi^T
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
		len9, len9, len4,
		&alpha,
		Xi, len9,
		Xi, len9, &beta,
		B, len9);

	// A = Xi*Xi^T + lambda*I =  A + B
	alpha = 1.0f;
	beta = 1.0f;	//beta 为正表明加法
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		len9, len9,
		&alpha,
		A[0], len9,
		&beta, B, len9,
		A[0], len9);

	// F = (Xi*Xi^T + lambda*I)^(-1) * Xi*Xi^T
	// 结果存储于B矩阵
	solve(A, B, pivot_vec, pivot_arr, info, L, U);

	//// 生成单位矩阵，此处复用A[0]
	createDiagnal(A[0], 1.0f, len9);

	// 计算 B = I-F
	alpha = 1.0f;
	beta = -1.0f;	//beta 为负表明加法
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		len9, len9,
		&alpha,
		A[0], len9,
		&beta, B, len9,
		B, len9);

	// 计算 B = (I-F)^T*(I-F) = B^T*B
	alpha = 1.0f;
	beta = 0.0f;
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
		len9, len9, len9,
		&alpha,
		B, len9,
		B, len9, &beta,
		B, len9);

	cublasDestroy(handle);

}

//__global__ void kernel(float *d_Xi, float *d_A, float **d_pA, float *d_B, int *d_pivot_vec, float *d_pivot_arr, int *d_info, float *d_L, float *d_U)
//{
//	int tid = threadIdx.x + blockIdx.x*blockDim.x;
//	if (tid > effThreads - 1)
//		return;
//
//	float *Xi, **A, *B;
//	int *pivot_vec;
//	float *pivot_arr;
//	int *info;
//	float *L, *U;
//
//	mapDeviceMem(d_Xi, d_A, d_pA, d_B, d_pivot_vec, d_pivot_arr, d_info, d_L, d_U,
//		Xi, A, B, pivot_vec, pivot_arr, info, L, U, tid);
//
//	// 公式中Xi存储于B矩阵中，请在此进行赋值
//	//B[0] = 1;		B[3] = 2;		B[6] = 6;
//	//B[1] = 4;		B[4] = 8;		B[7] = -1;
//	//B[2] = -2;	B[5] = 3;		B[8] = -5;
//
//	//Xi[0] = 1;		Xi[9] = 2;		Xi[18] = 3;		Xi[27] = 1;
//	//Xi[1] = 2;		Xi[10] = 3;		Xi[19] = 4;		Xi[28] = 1;
//	//Xi[2] = 3;		Xi[11] = 4;		Xi[20] = 5;		Xi[29] = 1;
//	//Xi[3] = 4;		Xi[12] = 5;		Xi[21] = 6;		Xi[30] = 1;
//	//Xi[4] = 5;		Xi[13] = 6;		Xi[22] = 7;		Xi[31] = 1;
//	//Xi[5] = 6;		Xi[14] = 7;		Xi[23] = 8;		Xi[32] = 1;
//	//Xi[6] = 7;		Xi[15] = 8;		Xi[24] = 9;		Xi[33] = 1;
//	//Xi[7] = 8;		Xi[16] = 9;		Xi[25] = 1;		Xi[34] = 1;
//	//Xi[8] = 9;		Xi[17] = 1;		Xi[26] = 2;		Xi[35] = 1;
//
//
//	//Xi[0] = 0.3725;  Xi[9] = 0.3137;   Xi[18] = 0.3137;  Xi[27] = 1;
//	//Xi[1] = 0.3882;  Xi[10] = 0.3333;  Xi[19] = 0.3294;  Xi[28] = 1;
//	//Xi[2] = 0.4157;  Xi[11] = 0.3569;  Xi[20] = 0.3569;  Xi[29] = 1;
//	//Xi[3] = 0.5216;  Xi[12] = 0.4549;  Xi[21] = 0.4667;  Xi[30] = 1;
//	//Xi[4] = 0.4824;  Xi[13] = 0.4157;  Xi[22] = 0.4235;  Xi[31] = 1;
//	//Xi[5] = 0.4588;  Xi[14] = 0.3922;  Xi[23] = 0.3961;  Xi[32] = 1;
//	//Xi[6] = 0.6078;  Xi[15] = 0.5373;  Xi[24] = 0.5529;  Xi[33] = 1;
//	//Xi[7] = 0.6706;  Xi[16] = 0.5961;  Xi[25] = 0.6235;  Xi[34] = 1;
//	//Xi[8] = 0.6;     Xi[17] = 0.5294;  Xi[26] = 0.549;   Xi[35] = 1;
//
//	//
//	//Xi[0] = 0.5569;  Xi[9] = 0.4941;   Xi[18] = 0.5098;   Xi[27] = 1;
//	//Xi[1] = 0.5529;  Xi[10] = 0.4902;  Xi[19] = 0.5176;   Xi[28] = 1;
//	//Xi[2] = 0.4235;  Xi[11] = 0.3686;  Xi[20] = 0.3765;   Xi[29] = 1;
//	//Xi[3] = 0.5647;  Xi[12] = 0.5059;  Xi[21] = 0.5294;   Xi[30] = 1;
//	//Xi[4] = 0.4253;  Xi[13] = 0.3804;  Xi[22] = 0.3922;   Xi[31] = 1;
//	//Xi[5] = 0.4588;  Xi[14] = 0.3922;  Xi[23] = 0.4;      Xi[32] = 1;
//	//Xi[6] = 0.5608;  Xi[15] = 0.502;   Xi[24] = 0.5176;   Xi[33] = 1;
//	//Xi[7] = 0.4706;  Xi[16] = 0.4118;  Xi[25] = 0.4235;	  Xi[34] = 1;
//	//Xi[8] = 0.5216;  Xi[17] = 0.4588;  Xi[26] = 0.4667;   Xi[35] = 1;
//
//
//	compLapcoeff(Xi, A, B, pivot_vec, pivot_arr, info, L, U);
//}

__global__ void kernel_new(float *d_Xi, float *d_A, float **d_pA, float *d_B, int *d_pivot_vec, float *d_pivot_arr, int *d_info, float *d_L, float *d_U,
	float *d_imdata, float *d_vals, int *d_trimap, int *d_row_inds, int *d_col_inds,
	int rows, int cols)
{
	//int x = blockIdx.x;
	//int y = blockIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//printf("x: %d y: %d\n", x, y);


	int tid = x + y * cols;

	//if ( x>30 && x<35 && y>30 && y<35)
	//if (x==41 && y==1)
	//	printf("%d x:%d y:%d r:%f g:%f b:%f trimap:%d cols:%d\n", tid, x, y, d_imdata[tid * 3], d_imdata[tid * 3+1], d_imdata[tid * 3+2], d_trimap[tid], cols);

	if ((x <= 0) || (y <= 1) || (x >= cols - 1) || (y >= rows - 1) || d_trimap[tid])
		return;
	

	float *Xi, **A, *B, *pivot_arr, *L, *U;
	int *pivot_vec, *info;

	int *row_inds, *col_inds;
	float *vals;

	mapDeviceMem(
		d_Xi, d_A, d_pA, d_B, d_pivot_vec, d_pivot_arr, d_info, d_L, d_U,
		d_trimap, d_row_inds, d_col_inds, d_vals,
		Xi, A, B, pivot_vec, pivot_arr, info, L, U, 
		row_inds, col_inds, vals,
		tid);

	assignXiMatrix(d_imdata, Xi, tid, cols, x, y);

	compLapcoeff(Xi, A, B, pivot_vec, pivot_arr, info, L, U);

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

	//if (tid == 121)
	//{
	//	printf("row: ");
	//	d_printI(row_inds, len9, len9);
	//	
	//	printf("col: ");
	//	d_printI(col_inds, len9, len9);

	//	printf("B: ");
	//	d_printD(B, len9, len9);
	//}
	//	

	////row
	//for (int j = 0; j < len9; ++j)s
	//	for (int i = 0; i < len9; ++i)
	//		row_inds[j*len9 + i] = i;
	////col
	//for (int j = 0; j < len9; ++j)
	//	for (int i = 0; i < len9; ++i)
	//		row_inds[j*len9 + i] = j;
	////vals
	//for (int i = 0; i < len99; ++i)
	//	vals[i] = B[i];

}

//=========================================================================================================================================================//

void allocMem(
	int pixNum,
	float *&d_Xi, float **&d_pA, float *&d_A, float *&d_B, int *&d_pivot_vec, float *&d_pivot_arr, int *&d_info, float *&d_L, float *&d_U,
	float *&d_imdata, float *&d_vals, int *&d_trimap, int *&d_row_inds, int *&d_col_inds)
{
	//d_Xi
	checkCudaErrors(cudaMalloc((void**)&d_Xi, pixNum * len94 * sizeof(float)));

	//d_pA和d_A
	checkCudaErrors(cudaMalloc((void**)&d_pA, pixNum * sizeof(float *)));
	checkCudaErrors(cudaMalloc((void**)&d_A, pixNum * len99 * sizeof(float)));

	//d_B
	checkCudaErrors(cudaMalloc((void**)&d_B, pixNum * len99 * sizeof(float)));

	//d_pivot_vec,	d_pivot_arr
	checkCudaErrors(cudaMalloc((void**)&d_pivot_vec, pixNum * len9 * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_pivot_arr, pixNum * len99 * sizeof(float)));

	//d_info
	checkCudaErrors(cudaMalloc((void**)&d_info, pixNum * sizeof(int)));

	//L和U
	checkCudaErrors(cudaMalloc((void**)&d_L, pixNum * len99 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_U, pixNum * len99 * sizeof(float)));

	//d_imdata
	checkCudaErrors(cudaMalloc((void**)&d_imdata, pixNum * channel * sizeof(float)));
	
	//d_vals
	checkCudaErrors(cudaMalloc((void**)&d_vals, pixNum * len99 * sizeof(float)));

	//d_trimap
	checkCudaErrors(cudaMalloc((void**)&d_trimap, pixNum * sizeof(int)));

	//d_row_inds, d_col_inds
	checkCudaErrors(cudaMalloc((void**)&d_row_inds, pixNum * len99 * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_col_inds, pixNum * len99 * sizeof(int)));
}

void freeMem(float *&d_Xi, float **&d_pA, float *&d_A, float *&d_B, int *&d_pivot_vec, float *&d_pivot_arr, int *&d_info, float *&d_L, float *&d_U,
	float *&d_imdata, float *&d_vals, int *&d_trimap, int *&d_row_inds, int *&d_col_inds)
{
	cudaFree(d_Xi);
	cudaFree(d_pA);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_pivot_vec);
	cudaFree(d_pivot_arr);
	cudaFree(d_info);
	cudaFree(d_L);
	cudaFree(d_U);
	
	//
	cudaFree(d_imdata);
	cudaFree(d_vals);
	cudaFree(d_trimap);
	cudaFree(d_row_inds);
	cudaFree(d_col_inds);

}

void printMatrix(float* m, int rows, int cols){
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

extern "C" void callKernel(int rows, int cols, float *imdata, int *trimap, int *row_inds, int *col_inds, float *vals)
{
	int pixNum = rows * cols;

	float *d_Xi, *d_A, **d_pA, *d_B, *d_pivot_arr, *d_L, *d_U;
	int *d_pivot_vec, *d_info;

	float *d_imdata, *d_vals;
	int *d_trimap, *d_row_inds, *d_col_inds;

	//
	allocMem(pixNum, 
		d_Xi, d_pA, d_A, d_B, d_pivot_vec, d_pivot_arr, d_info, d_L, d_U,
		d_imdata, d_vals, d_trimap, d_row_inds, d_col_inds);

	checkCudaErrors(cudaMemcpy(d_imdata, imdata, pixNum * channel *sizeof(float), cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy(d_trimap, trimap, pixNum * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(d_row_inds, 0, pixNum * len99 * sizeof(int)));
	checkCudaErrors(cudaMemset(d_col_inds, 0, pixNum * len99 * sizeof(int)));
	checkCudaErrors(cudaMemset(d_vals,	0, pixNum * len99 * sizeof(float)));         
	

	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));
	//dim3 grid(2048, 2048);
	//dim3 block(3, 3, 1);
	//dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	dim3 grid(1024, 1024);
	kernel_new <<<grid, 1>>>(d_Xi, d_A, d_pA, d_B, d_pivot_vec, d_pivot_arr, d_info, d_L, d_U,
		d_imdata, d_vals, d_trimap, d_row_inds, d_col_inds,
		rows, cols);

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, NULL));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	printf("\ncopy time consumption: %f\n", msecTotal);

	//float xi[len9*len9];
	//checkCudaErrors(cublasGetMatrix(len9, len4, sizeof(float), d_Xi, len9, xi, len9));
	//checkCudaErrors(cublasGetMatrix(len9, len9, sizeof(float), d_A, len9, a, len9));
	//checkCudaErrors(cublasGetMatrix(len9, len9, sizeof(float), d_B, len9, b, len9));
	//checkCudaErrors(cublasGetMatrix(len9, len9, sizeof(float), d_L, len9, l, len9));
	//checkCudaErrors(cublasGetMatrix(len9, len9, sizeof(float), d_U, len9, u, len9));
	//checkCudaErrors(cublasGetMatrix(len9, len9, sizeof(float), d_pivot_arr, len9, p, len9));



	checkCudaErrors(cudaMemcpy(row_inds, d_row_inds, pixNum * len99 * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(col_inds, d_col_inds, pixNum * len99 *sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vals, d_vals, pixNum * len99 * sizeof(float), cudaMemcpyDeviceToHost));





	//printf("row_inds:\n");
	//for (int i = 0; i < pixNum *len99; ++i)
	//	printf("%d\t", row_inds[i]);

	//cout << "xi is: " << endl;
	//printMatrix(xi, len9, len4);
	//cout << "b is: " << endl;
	//printMatrix(b, len9, len9);
	//cout << "a is: " << endl;
	//printMatrix(a, len9, len9);
	//cout << "l is: " << endl;
	//printMatrix(l, len9, len9);
	//cout << "u is: " << endl;
	//printMatrix(u, len9, len9);


	freeMem(d_Xi, d_pA, d_A, d_B, d_pivot_vec, d_pivot_arr, d_info, d_L, d_U,
		d_imdata, d_vals, d_trimap, d_row_inds, d_col_inds);
		
	return ;
}