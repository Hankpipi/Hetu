#include "gpu_runtime.h"

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

__global__ void scatter_kernel(float* target_data, const float* index_data,
                               const float* src_data, size_t size, int col) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;

    int nr = ind / col;
    int nc = ind % col;
    int ind_new = int(index_data[nr]) * col + nc;
    target_data[ind_new] = src_data[ind];
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
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
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

    CUBLAS_CALL(cublasGemmEx(cublas_map[dev_id], transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one,
                (const float *)matB->data, data_type, !transposeB ? m : k,
                (const float *)matA->data, data_type, !transposeA ? k : n, &one,
                (float *)matC->data, data_type, m, data_type, algo));
    return 0;
}

int DLGpuLinearSparse(const DLArrayHandle matA, bool transposeA,
                const DLArrayHandle matB, bool transposeB,
                const DLArrayHandle bias,
                DLArrayHandle matC, DLArrayHandle index,
                DLArrayHandle matA_sparse, DLArrayHandle matC_sparse,
                DLStreamHandle stream_handle = NULL) {
    // cublas assume matrix is column major
    assert(matB->ndim == 2);
    assert(bias->ndim == 1);
    assert(matA->ndim == matC->ndim);

    // gather
    size_t size = 1;
    int col = matA_sparse->shape[1];

    for (index_t i = 0; i < matA_sparse->ndim; i++) {
        size *= matA_sparse->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    cudaStream_t* s = nullptr;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
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
            (float *)matC->data, (const float *)index->data, (const float *)matC_sparse->data, size, col);
    }else{
        scatter_kernel<<<blocks, threads>>>(
            (float *)matC->data, (const float *)index->data, (const float *)matC_sparse->data, size, col);
    }
    return 0;
}
