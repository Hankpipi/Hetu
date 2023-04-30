#include "gpu_runtime.h"
#include "gpu_reduce.h"

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

/*
__global__ void gather_for_linear_kernel(const float *input, const float *index,
                                        float *output, size_t size, int col) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int nr = ind / col;
    int nc = ind % col;
    int ind_new = int(index[nr]) * col + nc;
    output[ind] = input[ind_new];
}
*/


__global__ void gather_for_linear_kernel(const float *x,
                                   const float *index,
                                   const float *scale,
                                   const float *shift,
                                   float* y,
                                   const float eps, const int C) {
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


int DLGpuGatherForLinear(const DLArrayHandle input, DLArrayHandle output,
                        DLArrayHandle index, DLArrayHandle scale, DLArrayHandle shift,
                        const float eps = 0, DLStreamHandle stream_handle = NULL) {
    assert (output->ndim == 3);
    dim3 blocks;
    dim3 threads;
    cudaStream_t* s = nullptr;
    blocks.x = output->shape[0] * output->shape[1];
    int col = output->shape[2];
    threads = GetThreadNum(col);
    if (stream_handle) {
        s = (cudaStream_t*)(stream_handle->handle);
        gather_for_linear_kernel<<<blocks, threads, 0, *s>>>(
            (const float *)(input->data), (const float *)(index->data), 
            (const float *)(scale->data), (const float *)(shift->data), 
            (float *)(output->data),
            eps, col);
    }
    else {
        gather_for_linear_kernel<<<blocks, threads>>>(
            (const float *)(input->data), (const float *)(index->data), 
            (const float *)(scale->data), (const float *)(shift->data), 
            (float *)(output->data),
            eps, col);
    }
    return 0;
}

__global__ void gather_for_linear_simple_kernel(const float *input, const float *index,
                              float *output, size_t size, int col) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;

    int nr = ind / col;
    int nc = ind % col;
    int ind_new = int(index[nr]) * col + nc;
    output[ind] = input[ind_new];
}

int DLGpuGatherForLinearSimple(const DLArrayHandle input, DLArrayHandle output,
                        DLArrayHandle index, DLStreamHandle stream_handle = NULL) {
    assert (output->ndim == 3);
    dim3 blocks;
    dim3 threads;
    cudaStream_t* s = nullptr;
    size_t size = 1;
    int col = output->shape[1];
    for (index_t i = 0; i < output->ndim; i++) {
        size *= output->shape[i];
    }

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        s = (cudaStream_t*)(stream_handle->handle);
        gather_for_linear_simple_kernel<<<blocks, threads, 0, *s>>>(
            (const float *)(input->data), (const float *)(index->data), (float *)(output->data),
            size, col);
    }
    else
        gather_for_linear_simple_kernel<<<blocks, threads>>>(
            (const float *)(input->data), (const float *)(index->data), (float *)(output->data),
            size, col);
    return 0;
}
