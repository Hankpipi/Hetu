#include "gpu_runtime.h"
#include "conv2d_utils.h"

__global__ void conv2d_add_bias(size_t nthreads,
    const float *input_data,
    float *output_data,
    size_t input_size,
    size_t output_size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nthreads)
    return;
    size_t input_id = id % input_size / output_size;
    output_data[id] += input_data[input_id];
}

/**
Gather: input [N, C, H, W] ---> output [N * block_sum, C, block_h, block_w].
Already implemented: fuse activation in it.
Already implemented: fuse GroupNorm (scale + shift) in it.
*/
__global__ void gather_for_conv_kernel(const int nthreads, const float *input, const float *index, float *output, 
                                const int N, const int C, const int H, const int W, 
                                const int block_sum, const int block_h, const int block_w, 
                                const int activation_mode = 0, const float *scale = NULL, const float *shift = NULL) {

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
    if (activation_mode == 1)
        res = res / (1.0f + exp(-res));
    output[output_pos] = res;
}


/**
Note that the C, block_h and block_w is different now (after Conv).
Scatter: src [N * block_sum, C, block_h, block_w] ---> target [B, C, H, W].
Already implemented: fuse add bias in it.
To be implemented: fuse residual block in it.
*/
__global__ void scatter_for_conv_kernel(const int nthreads, float* target, const float* index, const float* src,
                        const int N, const int C, const int H, const int W, 
                        const int block_sum, const int block_h, const int block_w,
                        const int stride_h, const int stride_w,
                        const float *bias = NULL, const float *residual = NULL) {

    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int src_pos = ind;

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

    int target_h = index[block_num] + offset_h;
    int target_w = index[block_sum + block_num] + offset_w;
    int target_pos = batch_num * C * H * W + channel_num * H * W + target_h * W + target_w;
    target[target_pos] = src[src_pos] + bias[channel_num];
}


int Cudnn_Conv2dAddBias(const DLArrayHandle input_x, const DLArrayHandle input_f,
                      const DLArrayHandle bias, DLArrayHandle output,
                      const int padding_h, const int padding_w,
                      const int stride_h, const int stride_w,
                      DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_x->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    const float *input_data = (const float *)input_x->data;
    const float *filter_data = (const float *)input_f->data;
    float *output_data = (float *)output->data;
    
    Conv2dArgs conv_args(input_x, input_f, padding_h, padding_w,
                         stride_h, stride_w);

    if (conv2d_cache.find(conv_args) == conv2d_cache.end()) {
        conv2d_cache[conv_args] = std::unique_ptr<Conv2dDesc>(
            new Conv2dDesc(input_x, input_f, output,
                            padding_h, padding_w,
                            stride_h, stride_w));
    }
    Conv2dDesc* desc = conv2d_cache[conv_args].get();
    void* work_data = find_chunk(desc->workspace_size, dev_id);

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn_map[dev_id], &alpha, desc->input_desc, input_data,
        desc->filter_desc, filter_data, desc->conv_desc, desc->algo,
        work_data, desc->workspace_size,
        &beta, desc->out_desc, output_data));

    del_chunk(work_data, dev_id);
    size_t out_N = output->shape[0];
    size_t out_C = output->shape[1];
    size_t out_H = output->shape[2];
    size_t out_W = output->shape[3];
    // add bias
    const float *bias_data = (const float*)bias->data;
    size_t nthreads = out_N * out_C * out_H * out_W;
    size_t BLOCKS = (nthreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t bias_output_size = out_H * out_W;
    size_t bias_input_size = out_C * bias_output_size;
    if (stream_handle)
        conv2d_add_bias<<<BLOCKS, THREADS_PER_BLOCK, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            nthreads, bias_data, output_data, bias_input_size, bias_output_size);
    else
        conv2d_add_bias<<<BLOCKS, THREADS_PER_BLOCK>>>(
            nthreads, bias_data, output_data, bias_input_size, bias_output_size);
    return 0;
}

/**
 * @brief input_x --gather--> gather_map --conv--> scatter_map --scatter--> output
 * 
 * @param input_x           input
 * @param input_f           conv kernel
 * @param bias              conv bias
 * @param output            output
 * @param gather_index      the beginning coordinate of the block to gather, [x1, x2 ..., xn, y1, y2, ..., yn]
 * @param scatter_index     the beginning coordinate of the block to scatter, [x1, x2 ..., xn, y1, y2, ..., yn]
 * @param block_sum         number of blocks to do sparse conv
 * @param block_h           the ***overlapped height of the block to gather 
 * @param block_w           the ***overlapped width of the block to gather 
 * @param gather_map        the array after gather     
 * @param scatter_map       the array before scatter
 * @param padding_h         not used
 * @param padding_w         not used
 * @param stride_h          height stride of conv
 * @param stride_w          width stride of conv
 * @param activation_mode   fuse activation
 * @param scale             fuse groupnorm
 * @param shift             fuse groupnorm
 * @param stream_handle     use for sync
 * @return int           
 */
int Cudnn_Conv2dAddBiasSparse(const DLArrayHandle input_x, const DLArrayHandle input_f,
                      const DLArrayHandle bias, DLArrayHandle output,
                      DLArrayHandle gather_index, DLArrayHandle scatter_index,
                      const int block_sum, int block_h, int block_w, 
                      DLArrayHandle gather_map, DLArrayHandle scatter_map,
                      int padding_h, int padding_w,
                      const int stride_h, const int stride_w,
                      const int activation_mode = 0, DLArrayHandle scale = NULL, DLArrayHandle shift = NULL,
                      DLStreamHandle stream_handle = NULL) {

    int dev_id = (input_x->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    const float *input_data = (const float *)input_x->data;
    const float *filter_data = (const float *)input_f->data;
    const float *bias_data = (const float*)bias->data;
    float *gather_data = (float *)gather_map->data;
    float *scatter_data = (float *)scatter_map->data;
    float *output_data = (float *)output->data;

    // GN
    float *scale_data = NULL, *shift_data = NULL;
    if (scale != NULL)
    {
        assert (scale->ndim == 2);
        assert (scale->shape[0] == input_x->shape[0]);
        assert (scale->shape[1] == input_x->shape[1]);
        scale_data = (float *)scale->data;
        shift_data = (float *)shift->data;
    }

    // Gather
    int in_N = input_x->shape[0];
    int in_C = input_x->shape[1];
    int in_H = input_x->shape[2];
    int in_W = input_x->shape[3];
    int nthreads = in_N * block_sum * in_C * block_h * block_w;
    int BLOCKS = (nthreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        gather_for_conv_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            nthreads, input_data, (const float *)(gather_index->data), gather_data, 
            in_N, in_C, in_H, in_W, block_sum, block_h, block_w,
            activation_mode, scale_data, shift_data);
    else
        gather_for_conv_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
            nthreads, input_data, (const float *)(gather_index->data), gather_data, 
            in_N, in_C, in_H, in_W, block_sum, block_h, block_w,
            activation_mode, scale_data, shift_data);

    // Conv
    // Do not need padding (already calculated in index).
    padding_h = 0;
    padding_w = 0;
    Conv2dArgs conv_args(gather_map, input_f, padding_h, padding_w,
        stride_h, stride_w);

    if (conv2d_cache.find(conv_args) == conv2d_cache.end()) {
        conv2d_cache[conv_args] = std::unique_ptr<Conv2dDesc>(
            new Conv2dDesc(gather_map, input_f, scatter_map,
                            padding_h, padding_w,
                            stride_h, stride_w));
    }
    Conv2dDesc* desc = conv2d_cache[conv_args].get();
    void* work_data = find_chunk(desc->workspace_size, dev_id);

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn_map[dev_id], &alpha, desc->input_desc, gather_data,
        desc->filter_desc, filter_data, desc->conv_desc, desc->algo,
        work_data, desc->workspace_size,
        &beta, desc->out_desc, scatter_data));

    
    // Scatter
    del_chunk(work_data, dev_id);
    int out_N = output->shape[0];
    int out_C = output->shape[1];
    int out_H = output->shape[2];
    int out_W = output->shape[3];
    block_h = scatter_map->shape[2];
    block_w = scatter_map->shape[3];  
    nthreads = out_N * block_sum * out_C * block_h * block_w;
    BLOCKS = (nthreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  
    if (stream_handle)
        scatter_for_conv_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0,
                                    *(cudaStream_t *)stream_handle->handle>>>(
            nthreads, output_data, (const float *)(scatter_index->data), scatter_data, 
            out_N, out_C, out_H, out_W, block_sum, 
            block_h, block_w, stride_h, stride_w, bias_data);
    else
        scatter_for_conv_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
            nthreads, output_data, (const float *)(scatter_index->data), scatter_data, 
            out_N, out_C, out_H, out_W, block_sum, 
            block_h, block_w, stride_h, stride_w, bias_data);

    return 0;
}