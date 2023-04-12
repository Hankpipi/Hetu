#include "gpu_runtime.h"
#include "gpu_reduce.h"
#define pi 3.14159265358979323846f
#define e  2.71828182845904523536f
#define SQRT_1_2  0.70710678118654757274f  // sqrt(1/2)
#include <stdio.h>

// Fuse layernorm in it.
__global__ void fused_gather_kernel(const float *x,
                                   const float *index,
                                   const float *scale,
                                   const float *shift,
                                   float* y,
                                   const float eps, const int C,
                                   const int activation_mode = 0) {
    __shared__ float var_share;
    __shared__ float mean_share;
    __shared__ float shared_var[32];
    __shared__ float shared_mean[32];

    int input_x = int(index[blockIdx.x]);
    int output_begin = blockIdx.x * C + threadIdx.x;
    int begin = input_x * C + threadIdx.x;
    int end = (input_x + 1) * C;

    float mean_thread = 0, var_thread = 0;
    for (int i = begin; i < end; i += blockDim.x) {
        mean_thread += x[i];
        var_thread += (x[i] * x[i]);
    }

    BlockReduceSum(mean_thread, shared_mean);
    BlockReduceSum(var_thread, shared_var);
    if (threadIdx.x == 0) {
        mean_share = mean_thread / C;
        var_share = var_thread / C - mean_share * mean_share;
        if (var_share < 0) var_share = 0;
    }
    __syncthreads();

    mean_thread = mean_share;
    var_thread = var_share;
    float tmp = 1.0f / sqrtf(var_thread + eps);
    for (int i = begin, j = threadIdx.x, k = output_begin; 
        i < end; i += blockDim.x, j += blockDim.x, k += blockDim.x)
        y[k] = (x[i] - mean_thread) * tmp * scale[j] + shift[j];
}

__global__ void broadcast_linear_bias(const float *input_data, float *output_data,
    size_t input_size, size_t output_size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= output_size)
    return;
    output_data[id] = input_data[id % input_size];
}

__global__ void gather_kernel(const float *input, const float *index,
                              float *output, size_t size, int col) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;

    int nr = ind / col;
    int nc = ind % col;
    int ind_new = int(index[nr]) * col + nc;
    output[ind] = input[ind_new];
}

// Fuse gelu_half in it.
__global__ void scatter_kernel(float* target_data, const float* index_data,
                               const float* src_data, size_t size, int col,
                               int activation_mode) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;

    int nr = ind / col;
    int nc = ind % col;
    int ind_new = int(index_data[nr]) * col + nc;
    float ans = src_data[ind];
    
    // half_gelu
    if (activation_mode == 1 && nc >= (col / 2))
        ans = ans * 0.5f * (1.0f + erff(ans * SQRT_1_2));

    target_data[ind_new] = ans;
}

int DLGpuLinear(const DLArrayHandle matA, bool transposeA,
                const DLArrayHandle matB, bool transposeB,
                const DLArrayHandle bias,
                DLArrayHandle matC,
                DLStreamHandle stream_handle = NULL) {
    // cublas assume matrix is column major
    assert(matB->ndim == 2);
    assert(bias->ndim == 1);
    assert(matA->ndim == matC->ndim);

    size_t input_size = bias->shape[0];
    size_t size = input_size;
    for(int i = 0; i < matC->ndim - 1; ++i)
        size *= matC->shape[i];

    dim3 blocks;
    dim3 threads;
    if (size <= THREADS_PER_BLOCK) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = THREADS_PER_BLOCK;
        blocks.x = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    }
    cudaStream_t* s = nullptr;
    if (stream_handle) {
        s = (cudaStream_t*)(stream_handle->handle);
        broadcast_linear_bias<<<blocks, threads, 0, *s>>>(
            (const float *)(bias->data), (float *)(matC->data), input_size, size);
    } else {
        broadcast_linear_bias<<<blocks, threads>>>(
            (const float *)(bias->data), (float *)(matC->data), input_size, size);
    }

    int dev_id = (matA->ctx).device_id;
    cublas_init(dev_id, stream_handle);

    float one = 1.0f;
    int m = matC->shape[matC->ndim - 1], n = 1;
    for(int i = 0; i < matC->ndim - 1; ++i)
        n *= matC->shape[i];
    int k = transposeB ? matB->shape[1] : matB->shape[0];
    cudaDataType_t data_type = CUDA_R_32F;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    if (CUDART_VERSION >= 11000)
        algo = CUBLAS_GEMM_DEFAULT;

    CUBLAS_CALL(cublasGemmEx(cublas_map[dev_id], transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one,
                (const float *)matB->data, data_type, !transposeB ? m : k,
                (const float *)matA->data, data_type, !transposeA ? k : n, &one,
                (float *)matC->data, data_type, m, data_type, algo));
    return 0;
}


 /**
  * @brief matA --fused_gather--> matA_sparse --linear--> matC_sparse --scatter--> matC
  * 
  * @param matA 
  * @param transposeA 
  * @param matB 
  * @param transposeB 
  * @param bias 
  * @param matC 
  * @param index 
  * @param matA_sparse 
  * @param matC_sparse 
  * @param scale 
  * @param shift 
  * @param eps              
  * @param activation_mode  fuse activation function (1 stands for GeLU)
  * @param stream_handle 
  * @return int 
  */
int DLGpuLinearSparse(const DLArrayHandle matA, bool transposeA,
                const DLArrayHandle matB, bool transposeB,
                const DLArrayHandle bias,
                DLArrayHandle matC, DLArrayHandle index,
                DLArrayHandle matA_sparse, DLArrayHandle matC_sparse,
                DLArrayHandle scale = NULL, DLArrayHandle shift = NULL,
                const float eps = 0, const int activation_mode = 0,
                DLStreamHandle stream_handle = NULL) {
    // cublas assume matrix is column major
    assert(matB->ndim == 2);
    assert(bias->ndim == 1);
    assert(matA->ndim == matC->ndim);
    assert(matA_sparse->ndim == 2);

    // gather
    size_t size = 1;
    int col = matA_sparse->shape[1];

    for (index_t i = 0; i < matA_sparse->ndim; i++) {
        size *= matA_sparse->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    cudaStream_t* s = nullptr;
    // Fuse layernorm
    if (scale != NULL) {
        blocks.x = matA_sparse->shape[0];
        threads = GetThreadNum(col);
        if (stream_handle) {
            s = (cudaStream_t*)(stream_handle->handle);
            fused_gather_kernel<<<blocks, threads, 0, *s>>>(
                (const float *)(matA->data), (const float *)(index->data), 
                (const float *)(scale->data), (const float *)(shift->data), 
                (float *)(matA_sparse->data),
                eps, col, activation_mode);
        }
        else
            fused_gather_kernel<<<blocks, threads>>>(
                (const float *)(matA->data), (const float *)(index->data), 
                (const float *)(scale->data), (const float *)(shift->data), 
                (float *)(matA_sparse->data),
                eps, col, activation_mode);
    }
    // Do not fuse
    else {
        if (size <= THREADS_PER_BLOCK) {
            threads.x = size;
            blocks.x = 1;
        } else {
            threads.x = THREADS_PER_BLOCK;
            blocks.x = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        }
        if (stream_handle) {
            s = (cudaStream_t*)(stream_handle->handle);
            gather_kernel<<<blocks, threads, 0, *s>>>(
                (const float *)(matA->data), (const float *)(index->data), (float *)(matA_sparse->data),
                size, col);
        }
        else
            gather_kernel<<<blocks, threads>>>(
                (const float *)(matA->data), (const float *)(index->data), (float *)(matA_sparse->data),
                size, col);
    }

    // cudnn matmul
    size_t input_size = bias->shape[0];
    size = input_size;
    for(int i = 0; i < matC_sparse->ndim - 1; ++i)
        size *= matC_sparse->shape[i];

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        broadcast_linear_bias<<<blocks, threads, 0, *s>>>(
            (const float *)(bias->data), (float *)(matC_sparse->data), input_size, size);
    } else {
        broadcast_linear_bias<<<blocks, threads>>>(
            (const float *)(bias->data), (float *)(matC_sparse->data), input_size, size);
    }

    int dev_id = (matA->ctx).device_id;
    cublas_init(dev_id, stream_handle);

    float one = 1.0f;
    int m = matC_sparse->shape[matC_sparse->ndim - 1], n = 1;
    for(int i = 0; i < matC_sparse->ndim - 1; ++i)
        n *= matC_sparse->shape[i];
    int k = transposeB ? matB->shape[1] : matB->shape[0];
    cudaDataType_t data_type = CUDA_R_32F;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    if (CUDART_VERSION >= 11000)
        algo = CUBLAS_GEMM_DEFAULT;

    CUBLAS_CALL(cublasGemmEx(cublas_map[dev_id], transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one,
                (const float *)matB->data, data_type, !transposeB ? m : k,
                (const float *)matA_sparse->data, data_type, !transposeA ? k : n, &one,
                (float *)matC_sparse->data, data_type, m, data_type, algo));

    // scatter
    col = matC_sparse->shape[1];
    size = matC_sparse->shape[0] * col;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if(stream_handle){
        scatter_kernel<<<blocks, threads, 0, *s>>>(
            (float *)matC->data, (const float *)index->data, (const float *)matC_sparse->data, 
            size, col, activation_mode);
    }else{
        scatter_kernel<<<blocks, threads>>>(
            (float *)matC->data, (const float *)index->data, (const float *)matC_sparse->data, 
            size, col, activation_mode);
    }
    return 0;
}
