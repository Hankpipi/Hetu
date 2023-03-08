#include "gpu_reduce.h"

__global__ void group_norm_forward(const float *x,
                                   const float *scale,
                                   const float *bias,
                                   float* y,
                                   float *mean, float *var,
                                   const float eps, const int M, const int HW, const int C) {
    __shared__ float var_share;
    __shared__ float mean_share;
    __shared__ float shared_var[32];
    __shared__ float shared_mean[32];

    int begin = blockIdx.x * M + threadIdx.x;
    int end = (blockIdx.x + 1) * M;

    float mean_thread = 0, var_thread = 0;
    for (int i = begin; i < end; i += blockDim.x) {
        mean_thread += x[i];
        var_thread += x[i] * x[i];
    }

    BlockReduceSum(mean_thread, shared_mean);
    BlockReduceSum(var_thread, shared_var);
    if (threadIdx.x == 0) {
        mean[blockIdx.x] = mean_share = mean_thread / M;
        var_share = var_thread / M  - mean_share * mean_share;
        if (var_share < 0) var_share = 0;
        var[blockIdx.x] = var_share;
    }
    __syncthreads();    

    mean_thread = mean_share;
    var_thread = var_share;
    float tmp = 1.0f / sqrtf(var_thread + eps);
    for (int i = begin; i < end; i += blockDim.x) {
        int cid = i / HW % C;
        y[i] = (x[i] - mean_thread) * tmp * scale[cid] + bias[cid];
    }
}

int DLGpuGroupNormalization(const DLArrayHandle in_arr,
                            const DLArrayHandle ln_scale,
                            const DLArrayHandle ln_bias, 
                            int num_groups, DLArrayHandle mean_arr,
                            DLArrayHandle var_arr, DLArrayHandle out_arr,
                            float eps, DLStreamHandle stream_handle) {
    int ndim = in_arr->ndim;
    int B = in_arr->shape[0] * num_groups, M = 1, HW = 1;
    for(int i = 1; i < ndim; ++i)
        M *= in_arr->shape[i];
    HW = M / in_arr->shape[1];
    M /= num_groups;
    int threads = GetThreadNum(M);

    if (stream_handle)
        group_norm_forward<<<B, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                (const float *)in_arr->data, (const float *)ln_scale->data,
                (const float *)ln_bias->data, (float *)out_arr->data,
                (float *)mean_arr->data, (float *)var_arr->data, eps, M, HW, in_arr->shape[1]);
    else
        group_norm_forward<<<B, threads, 0>>>(
                (const float *)in_arr->data, (const float *)ln_scale->data,
                (const float *)ln_bias->data, (float *)out_arr->data,
                (float *)mean_arr->data, (float *)var_arr->data, eps, M, HW, in_arr->shape[1]);
    return 0;
}
