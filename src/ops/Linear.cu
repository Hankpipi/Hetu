#include "gpu_runtime.h"

__global__ void broadcast_linear_bias(const float *input_data, float *output_data,
    size_t input_size, size_t output_size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= output_size)
    return;
    output_data[id] = input_data[id % input_size];
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
