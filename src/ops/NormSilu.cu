#include "gpu_runtime.h"

__global__ void norm_silu_kernel(float *input, float *output,
        float *scale, float *bias,
        size_t size, size_t hw, size_t C, int block_sum, int activation_mode) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    int output_pos = ind;
    if (ind >= size)
        return;
    float res = input[ind];
    ind /= hw;
    int cid = ind % C;
    int bid = ind / C / block_sum;
    int pos = bid * C + cid;
    res = res * scale[pos] + bias[pos];
    if (activation_mode == 1)
        res = res / (1.0f + exp(-res));
    output[output_pos] = res;
}

int DLGpuNormSilu(const DLArrayHandle input, DLArrayHandle output,
              DLArrayHandle scale, DLArrayHandle bias,
              int block_sum, int activation_mode,
              DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    size_t hw = input->shape[2] * input->shape[3];
    size_t C = input->shape[1];
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *scale_data = (float *)scale->data;
    float *bias_data = (float *)bias->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        norm_silu_kernel<<<blocks, threads, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, scale_data, bias_data, size, 
            hw, C, block_sum, activation_mode);
    else
        norm_silu_kernel<<<blocks, threads>>>(input_data, output_data, 
            scale_data, bias_data, size, hw, C, block_sum, activation_mode);
    return 0;
}
