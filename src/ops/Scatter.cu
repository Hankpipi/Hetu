#include "gpu_runtime.h"
#define pi 3.14159265358979323846f
#define e  2.71828182845904523536f
#define SQRT_1_2  0.70710678118654757274f  // sqrt(1/2)

__global__ void scatter_kernel(float* target_data, float* index_data, float* src_data, int tgt_col, int src_col, int row){
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if(offset >= row){
        return ;
    }
    float* target_data_start = target_data + offset*tgt_col;
    float* index_data_start = index_data + offset*src_col;
    float* src_data_start = src_data + offset*src_col;

    for(int i=0; i<src_col; i++){
        target_data_start[int(index_data_start[i])]=src_data_start[i];
    }
}

int DLGpuScatter(const DLArrayHandle target, int dim, DLArrayHandle index, DLArrayHandle src, DLStreamHandle stream_handle = NULL){
    assert(target->ndim == 2);
    assert(src->ndim == 2);
    assert(index->ndim == 2);
    
    int ROW = target->shape[0];
    int COL = target->shape[1];
    int SRC_COL = src->shape[1];

    float* target_data = (float*) target->data;
    float* src_data = (float*) src->data;
    float* index_data = (float*) index->data;

    dim3 blocks;
    dim3 threads;

    assert(dim == 1);
    // dim = 0 not implemented yet
    // will implement it later

    if(dim == 1){
        if(ROW<=1024){
            blocks.x = 1;
            threads.x = ROW;
        }else{
            blocks.x = (ROW+1023)/1024;
            threads.x = 1024;
        }
        if(stream_handle){
            scatter_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(target_data, index_data, src_data, COL, SRC_COL, ROW);
        }else{
            scatter_kernel<<<blocks, threads>>>(target_data, index_data, src_data, COL, SRC_COL, ROW);
        }
    }
    return 0;
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
                        const int stride_h, const int stride_w, const float *residual = NULL) {

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
    target[target_pos] = src[src_pos];
}

int DLGpuScatterForConv(const DLArrayHandle input, DLArrayHandle output,
                      DLArrayHandle scatter_index,
                      const int block_sum, const int stride_h, const int stride_w,
                      DLStreamHandle stream_handle = NULL) {
    int out_N = output->shape[0];
    int out_C = output->shape[1];
    int out_H = output->shape[2];
    int out_W = output->shape[3];
    int block_h = input->shape[2];
    int block_w = input->shape[3];  
    int nthreads = out_N * block_sum * out_C * block_h * block_w;
    int blocks = (nthreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  
    if (stream_handle)
        scatter_for_conv_kernel<<<blocks, THREADS_PER_BLOCK, 0,
                                    *(cudaStream_t *)stream_handle->handle>>>(
            nthreads, (float*)(output->data), (const float *)(scatter_index->data),
            (const float *)(input->data), out_N, out_C, out_H, out_W, block_sum, 
            block_h, block_w, stride_h, stride_w);
    else
        scatter_for_conv_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            nthreads, (float*)(output->data), (const float *)(scatter_index->data),
            (const float *)(scatter_index->data), out_N, out_C, out_H, out_W, block_sum,
            block_h, block_w, stride_h, stride_w);
    return 0;
}



__global__ void scatter_for_linear_kernel(float* target_data, const float* index_data,
                               const float* src_data, const float* add_data, size_t size, int col,
                               int activation_mode) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;

    int nr = ind / col;
    int nc = ind % col;
    int ind_new = int(index_data[nr]) * col + nc;
    float temp = src_data[ind];

    target_data[ind_new] = add_data[ind_new] + temp;
}


int DLGpuScatterForLinear(const DLArrayHandle src, const DLArrayHandle add, DLArrayHandle target, 
                DLArrayHandle index,
                const int activation_mode = 0,
                DLStreamHandle stream_handle = NULL) {
    assert (src->ndim == 3);
    int B = src->shape[0];
    int L = src->shape[1];
    int col = src->shape[2];
    int size = B * L * col;
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
    if(stream_handle){
        s = (cudaStream_t*)(stream_handle->handle);
        scatter_for_linear_kernel<<<blocks, threads, 0, *s>>>(
            (float *)target->data, (const float *)index->data, (const float *)src->data, (const float *)add->data, 
            size, col, activation_mode);
    }else{
        scatter_for_linear_kernel<<<blocks, threads>>>(
            (float *)target->data, (const float *)index->data, (const float *)src->data, (const float *)add->data, 
            size, col, activation_mode);
    }
    return 0;
}
