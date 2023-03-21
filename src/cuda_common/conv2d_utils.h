#include "gpu_runtime.h"

struct Conv2dArgs {
    size_t input_n, input_c, input_h, input_w;
    size_t filter_n, filter_c, filter_h, filter_w;
    size_t padding_h, padding_w;
    size_t stride_h, stride_w;

    bool operator < (const Conv2dArgs& args) const {
        if (input_n != args.input_n) return input_n < args.input_n;
        if (input_c != args.input_c) return input_c < args.input_c;
        if (input_h != args.input_h) return input_h < args.input_h;
        if (input_w != args.input_w) return input_w < args.input_w;
        if (filter_n != args.filter_n) return filter_n < args.filter_n;
        if (filter_c != args.filter_c) return filter_c < args.filter_c;
        if (filter_h != args.filter_h) return filter_h < args.filter_h;
        if (filter_w != args.filter_w) return filter_w < args.filter_w;
        if (padding_h != args.padding_h) return padding_h < args.padding_h;
        if (padding_w != args.padding_w) return padding_w < args.padding_w;
        if (stride_h != args.stride_h) return stride_h < args.stride_h;
        return stride_w < args.stride_w;
    }
    
    bool operator == (const Conv2dArgs& args) const {
        return input_n == args.input_n && input_c == args.input_c
        && input_h == args.input_h && input_w == args.input_w
        && filter_n == args.filter_n && filter_c == args.filter_c
        && filter_h == args.filter_h && filter_w == args.filter_w
        && padding_h == args.padding_h && padding_w == args.padding_w
        && stride_h == args.stride_h && stride_w == args.stride_w;
    }

    Conv2dArgs(const DLArrayHandle input_x, const DLArrayHandle input_f,
                      const int _padding_h, const int _padding_w,
                      const int _stride_h, const int _stride_w):
                      padding_h(_padding_h), padding_w(_padding_w),
                      stride_h(_stride_h), stride_w(_stride_w) {

        input_n = input_x->shape[0];
        input_c = input_x->shape[1];
        input_h = input_x->shape[2];
        input_w = input_x->shape[3];
        filter_n = input_f->shape[0];
        filter_c = input_f->shape[1];
        filter_h = input_f->shape[2];
        filter_w = input_f->shape[3];
    }
};


struct Conv2dDesc {
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t input_desc, out_desc;
    
    int dev_id;
    size_t workspace_size;
    cudnnConvolutionFwdAlgo_t algo;

    Conv2dDesc(const DLArrayHandle input_x, const DLArrayHandle input_f, 
               DLArrayHandle output,
               const int padding_h, const int padding_w,
               const int stride_h, const int stride_w) {

        dev_id = (input_x->ctx).device_id;

        size_t input_n = input_x->shape[0];
        size_t input_c = input_x->shape[1];
        size_t input_h = input_x->shape[2];
        size_t input_w = input_x->shape[3];

        // input
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, input_n, input_c,
                                            input_h, input_w));
        size_t filter_n = input_f->shape[0];
        size_t filter_c = input_f->shape[1];
        size_t filter_h = input_f->shape[2];
        size_t filter_w = input_f->shape[3];

        // filter
        CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW, filter_n, filter_c,
                                            filter_h, filter_w));

        // convolution
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(
            conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        size_t out_N = output->shape[0];
        size_t out_C = output->shape[1];
        size_t out_H = output->shape[2];
        size_t out_W = output->shape[3];

        // output
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, out_N, out_C, out_H,
                                            out_W));

        int request_cnt = CUDNN_CONVOLUTION_FWD_ALGO_COUNT, return_cnt;
        cudnnConvolutionFwdAlgoPerf_t algo_perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

        // Heuristic Search
        CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
            cudnn_map[dev_id], input_desc, filter_desc, conv_desc, out_desc,
            request_cnt, &return_cnt, algo_perf));

        if (is_chunk_init(dev_id) == false)
            chunk_init(dev_id);

        void *work_data = nullptr;
        for(int i = 0; i < return_cnt; ++i) {
            CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
                cudnn_map[dev_id], input_desc, filter_desc, conv_desc, out_desc, algo_perf[i].algo,
                &workspace_size));
            work_data = find_chunk(workspace_size, dev_id, false);
            if (work_data) {
                del_chunk(work_data, dev_id);
                algo = algo_perf[i].algo;
                break;
            }
        }
    }

    ~Conv2dDesc() {
        CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
        CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
        CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
        CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    }
};

std::map<Conv2dArgs, std::unique_ptr<Conv2dDesc> > conv2d_cache;