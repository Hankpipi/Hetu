#include "gpu_runtime.h"

__global__ void gather_kernel(const float *input, const float *index,
                              float *output, size_t size, int dim_b_input,
                              int dim_c_input, int dim_b_output,
                              int dim_c_output) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int na = ind / (dim_b_output * dim_c_output);
    int tmp = ind % (dim_b_output * dim_c_output);
    int nc = tmp % dim_c_output;
    int ind_new =
        na * dim_b_input * dim_c_input + int(index[ind]) * dim_c_input + nc;
    output[ind] = input[ind_new];
}

__global__ void gather_gradient_kernel(const float *input, const float *index,
                                       float *output, size_t size,
                                       int dim_b_input, int dim_c_input,
                                       int dim_b_output, int dim_c_output) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float val = input[ind];
    int na = ind / (dim_b_output * dim_c_output);
    int tmp = ind % (dim_b_output * dim_c_output);
    int nc = tmp % dim_c_output;
    int ind_new =
        na * dim_b_input * dim_c_input + int(index[ind]) * dim_c_input + nc;
    atomicAdd(&output[ind_new], val);
}

int DLGpuGather(const DLArrayHandle input, const DLArrayHandle index,
                DLArrayHandle output, int dim,
                DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    int dim_b_input = input->shape[dim];
    int dim_c_input = 1;
    int dim_b_output = output->shape[dim];
    int dim_c_output = 1;

    for (index_t i = 0; i < input->ndim; i++) {
        size *= output->shape[i];
        if (i > dim) {
            dim_c_input *= input->shape[i];
            dim_c_output *= output->shape[i];
        }
    }

    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *index_data = (float *)index->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        gather_kernel<<<blocks, threads, 0,
                        *(cudaStream_t *)stream_handle->handle>>>(
            input_data, index_data, output_data, size, dim_b_input, dim_c_input,
            dim_b_output, dim_c_output);
    else
        gather_kernel<<<blocks, threads>>>(input_data, index_data, output_data,
                                           size, dim_b_input, dim_c_input,
                                           dim_b_output, dim_c_output);
    return 0;
}

int DLGpuGatherGradient(const DLArrayHandle input, const DLArrayHandle index,
                        DLArrayHandle output, int dim,
                        DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    int dim_b_input = input->shape[dim];
    int dim_c_input = 1;
    int dim_b_output = output->shape[dim];
    int dim_c_output = 1;

    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
        if (i > dim) {
            dim_c_input *= input->shape[i];
            dim_c_output *= output->shape[i];
        }
    }

    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *index_data = (float *)index->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        gather_gradient_kernel<<<blocks, threads, 0,
                                 *(cudaStream_t *)stream_handle->handle>>>(
            input_data, index_data, output_data, size, dim_b_output,
            dim_c_output, dim_b_input, dim_c_input);
    else
        gather_gradient_kernel<<<blocks, threads>>>(
            input_data, index_data, output_data, size, dim_b_output,
            dim_c_output, dim_b_input, dim_c_input);
    return 0;
}

/**
Gather: input [N, C, H, W] ---> output [N * block_sum, C, block_h, block_w].
Already implemented: fuse activation in it.
Already implemented: fuse GroupNorm (scale + shift) in it.
*/
__global__ void gather_for_conv_kernel(const int nthreads, const float *input, const float *index, float *output, 
                                const int N, const int C, const int H, const int W, 
                                const int block_sum, const int block_h, const int block_w, 
                                const float *scale = NULL, const float *shift = NULL) {

    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int output_pos = ind;

    if (ind >= nthreads)
        return;
    int offset_w = ind % block_w;
    ind /= block_w;
    int offset_h = ind % block_h;
    ind /= block_h;
    int channel_num = ind % C;
    ind /= C;
    int block_num = ind % block_sum;
    int batch_num = ind / block_sum;

    int input_h = index[block_num] + offset_h;
    if (input_h < 0 || input_h >= H)
    {
        output[output_pos] = 0;
        return;
    }
    int input_w = index[block_sum + block_num] + offset_w;
    if (input_w < 0 || input_w >= W)
    {
        output[output_pos] = 0;
        return;
    }
    int input_pos = batch_num * C * H * W + channel_num * H * W + input_h * W + input_w;
    float res = input[input_pos];
    // Fuse GN
    if (scale != NULL)
    {
        int pos = batch_num * C + channel_num;
        res *= scale[pos];
        res += shift[pos];
    }
    // Fuse activation function
    // SiLU
    res = res / (1.0f + exp(-res));

    output[output_pos] = res;
}


int DLGpuGatherForConv(const DLArrayHandle input, DLArrayHandle output,
                      DLArrayHandle gather_index, DLArrayHandle scale, DLArrayHandle shift,
                      const int block_sum, int block_h, int block_w,
                      DLStreamHandle stream_handle = NULL) {
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    // GN
    float *scale_data = NULL, *shift_data = NULL;
    if (scale != NULL)
    {
        assert (scale->ndim == 2);
        assert (scale->shape[0] == input->shape[0]);
        assert (scale->shape[1] == input->shape[1]);
        scale_data = (float *)scale->data;
        shift_data = (float *)shift->data;
    }

    // Gather
    int in_N = input->shape[0];
    int in_C = input->shape[1];
    int in_H = input->shape[2];
    int in_W = input->shape[3];
    int nthreads = in_N * block_sum * in_C * block_h * block_w;
    int BLOCKS = (nthreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        gather_for_conv_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            nthreads, input_data, (const float *)(gather_index->data), output_data, 
            in_N, in_C, in_H, in_W, block_sum, block_h, block_w,
            scale_data, shift_data);
    else
        gather_for_conv_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
            nthreads, input_data, (const float *)(gather_index->data), output_data, 
            in_N, in_C, in_H, in_W, block_sum, block_h, block_w,
            scale_data, shift_data);

    return 0;
}
