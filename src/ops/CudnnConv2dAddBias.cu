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
