#include "gpu_runtime.h"

int DLGpuBaddbmm(const DLArrayHandle input, const DLArrayHandle matA,
                 const DLArrayHandle matB, float alpha, float beta,
                 DLArrayHandle matC, DLStreamHandle stream_handle = NULL) {
    assert(matA->ndim == matB->ndim);
    assert(matA->ndim == matC->ndim);

    int dev_id = (matA->ctx).device_id;
    cublas_init(dev_id, stream_handle);

    int ndim = matA->ndim;
    int m = matC->shape[ndim - 1];
    int n = matC->shape[ndim - 2];
    int k = matA->shape[ndim - 1];
    long long int strideA = matA->shape[ndim - 2] * matA->shape[ndim - 1];
    long long int strideB = matB->shape[ndim - 2] * matB->shape[ndim - 1];
    long long int strideC = matC->shape[ndim - 2] * matC->shape[ndim - 1];

    int batchCount = 1;
    for (int i = 0; i < ndim - 2; ++i) {
        assert(matA->shape[i] == matB->shape[i]);
        assert(matA->shape[i] == matC->shape[i]);
        batchCount *= matA->shape[i];
    }

    float *input_data = (float *)input->data;
    float *output_data = (float *)matC->data;
    int size = 1;
    for (int i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }

    cudaMemcpy((void *)output_data, (void *)input_data, size * sizeof(float),
               cudaMemcpyDeviceToDevice);

    cudaDataType_t data_type = CUDA_R_32F;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    cublasStatus_t res = cublasGemmStridedBatchedEx(
        cublas_map[dev_id], CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        (const float *)matB->data, data_type, m, strideB,
        (const float *)matA->data, data_type, k, strideA, &beta,
        (float *)matC->data, data_type, m, strideC, batchCount, data_type, algo);
    assert(res == CUBLAS_STATUS_SUCCESS);
    return 0;
}
