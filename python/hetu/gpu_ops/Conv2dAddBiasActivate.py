from __future__ import absolute_import
import numpy as np

import torch
import hetu as ht
from .Node import Op
from .Conv2d import conv2d_gradient_of_data_op, conv2d_gradient_of_filter_op
from .ReduceSum import reduce_sum_op
from ..gpu_links import CuDNN_conv2d_with_bias, CuDNN_conv2d_with_bias_sparse, group_normalization, \
                        silu, norm_silu

from timeit import default_timer as timer
import pickle


class Conv2dAddBiasActivateOp(Op):

    profile_dict = {}

    workspace_cache = {}

    def __init__(self, node_A, node_B, bias, padding=0, stride=1, activation_mode=0,
                gn_weight=None, gn_bias=None, num_groups=32, eps=0.01, height=None, width=None, config=None, ctx=None):
        self.op_name = node_B.name
        self.cache_ctx = None
        self.mask = None
        self.limit_1 = None
        self.limit_2 = None
        self.config = config
        if gn_weight == None:
            super().__init__(Conv2dAddBiasActivateOp, [node_A, node_B, bias], ctx)
        else:
            super().__init__(Conv2dAddBiasActivateOp, [node_A, node_B, bias, gn_weight, gn_bias], ctx)
        if not isinstance(padding, tuple):
            assert isinstance(padding, int)
            padding = (padding, padding)
        self.padding = padding
        if not isinstance(stride, tuple):
            assert isinstance(stride, int)
            stride = (stride, stride)
        self.stride = stride

        # Conv
        self.block_division_h = height // 2
        self.block_division_w = width // 2
        # self.block_division_h = 12
        # self.block_division_w = 12
        self.latent_scale = height * width

        self.round = 0
        self.outdeg = 0
        self.d2h_stream = None
        self.use_sparse = False
        self.activation_mode = None
        self.scale = None
        self.shift = None
        self.output_cache = []
        self.should_init_cache = True

        # Activation
        # 0 stands for Identity function.
        # 1 stands for SiLU.
        self.activation_mode = activation_mode

        # GN
        self.fuse_gn = (gn_weight != None)
        self.eps = eps
        self.num_groups = num_groups
        self.gn_scale_cache = []
        self.gn_shift_cache = []
        self.mean = None
        self.var = None
        self.gn_output = None

    def im2col(self, X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding[0] - filter_H) % stride[0] == 0
        assert (W + 2 * padding[1] - filter_W) % stride[1] == 0
        out_H = (H + 2 * padding[0] - filter_H) // stride[0] + 1
        out_W = (W + 2 * padding[1] - filter_W) // stride[1] + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride[0] - padding[0]
                in_x = out_x * stride[1] - padding[1]
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                Y[batch_index, row_idx,
                                    col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y

    def np_conv2d(self, X, Filter, padding=(0, 0), stride=(1, 1)):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding[0] - filter_H) % stride[0] == 0
        assert (W + 2 * padding[1] - filter_W) % stride[1] == 0
        out_H = (H + 2 * padding[0] - filter_H) // stride[0] + 1
        out_W = (W + 2 * padding[1] - filter_W) // stride[1] + 1

        im2col_matrix = self.im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)


    def compute_edit(self, input_vals, output_val, stream_handle=None):
        # print("Sparse Conv:", self.fuse_gn, "input", input_vals[0].shape, "kernel", input_vals[1].shape, "output", output_val.shape)
        ctx = input_vals[0].ctx
        in_N, in_C, in_H, in_W = input_vals[0].shape
        filter_out_C, filter_in_C, filter_H, filter_W = input_vals[1].shape
        out_N, out_C, out_H, out_W = output_val.shape
        key = (in_N, in_C, out_C, out_H, out_W, self.stride, self.padding, filter_H, filter_W)

        if key not in Conv2dAddBiasActivateOp.workspace_cache:

            if self.mask == None:
                mask = torch.load(f'runtime/mask.pt')
            else:
                mask = self.mask
            
            flops_input = (out_C * out_H * out_W) * (filter_in_C * filter_H * filter_W)

            '''
            Note that the scatter_map is aligned with the default
            block_h and block_w, while the gather_map should be 
            little larger to match the conv kernel.
            The handling of the padding and stride in the original conv 
            should be considerd carefully in this section.
            '''

            # Scatter settings.
            # By default, we divide it into 12 * 12 blocks.
            # Assume the original latent's size is no less than 96 * 96.
            block_h = out_H // self.block_division_h
            block_w = out_W // self.block_division_w
            block_mask = torch.nn.MaxPool2d(kernel_size=(mask.shape[-2] // self.block_division_h, mask.shape[-1] // self.block_division_w))(mask.float().repeat(1, 1, 1, 1)) 
            assert (block_mask.shape[-2] == self.block_division_h and block_mask.shape[-1] == self.block_division_w)
            block_mask = (block_mask > 0.5)
            block_mask = block_mask.numpy()[0][0]
            index_arr = list(np.where(block_mask == True))
            index_arr[0] *= block_h
            index_arr[1] *= block_w
            block_sum = len(index_arr[0])

            # Gather settings.
            # We need to calculate the overlapped block size considering the conv kernel.
            overlapped_block_h = (block_h - 1) * self.stride[0] + filter_H
            overlapped_block_w = (block_w - 1) * self.stride[1] + filter_W
            flops_new = (block_sum * out_C * block_h * block_w) * (filter_in_C * filter_H * filter_W)
            # print(f'origin flops={flops_input}, new flops={flops_new}, rate={flops_new / flops_input}')
            if self.latent_scale >= self.config.latent_scale_conv:
                new_index_arr = index_arr.copy()
                new_index_arr[0] = new_index_arr[0] * self.stride[0] - self.padding[0]
                new_index_arr[1] = new_index_arr[1] * self.stride[1] - self.padding[1]

                scatter_index = ht.array(np.concatenate((index_arr[0], index_arr[1])), ctx=ctx)
                scatter_map = ht.empty((out_N * block_sum, out_C, block_h, block_w), ctx=ctx)
                gather_index = ht.array(np.concatenate((new_index_arr[0], new_index_arr[1])), ctx=ctx)
                gather_map = ht.empty((in_N * block_sum, in_C, overlapped_block_h, overlapped_block_w), ctx=ctx)

                Conv2dAddBiasActivateOp.workspace_cache[key] = (scatter_index, scatter_map, 
                                    gather_index, gather_map, overlapped_block_h, overlapped_block_w, block_sum)
            else:
                Conv2dAddBiasActivateOp.workspace_cache[key] = 'origin'
        
        value = Conv2dAddBiasActivateOp.workspace_cache[key]

        if self.config.profile:
            torch.cuda.synchronize() 
            time_start = timer()

        if not self.config.turn_off_h2d:
            if self.cache_ctx == self.ctx:
                pass
                # self.output_cache[self.round].copyto(output_val)
            elif self.cache_ctx == ht.cpu():
                self.event.sync()

        # Also need to load the scale & shift of the GN layer.
        scale, shift = None, None
        if self.fuse_gn:
            scale = self.gn_scale_cache[self.round]
            shift = self.gn_shift_cache[self.round]
        
        if value == 'origin':
            gn_output = input_vals[0]
            if self.fuse_gn:
                norm_silu(input_vals[0], self.gn_output, scale, shift, 
                        block_sum=1, activation_mode=self.activation_mode, stream=stream_handle)
                gn_output = self.gn_output
            CuDNN_conv2d_with_bias(gn_output, input_vals[1], input_vals[2],
                            output_val, self.padding, self.stride, stream_handle)
        else:
            scatter_index, scatter_map, gather_index, gather_map, overlapped_block_h, \
                overlapped_block_w, block_sum = value
            # A trade-off of peak memory and latency.
            # Store them as self.scatter_map and self.gather_map will get faster but need more memory.
            # scatter_map = ht.empty((out_N * self.block_sum, out_C, self.block_h, self.block_w), ctx=ctx)
            # gather_map = ht.empty((in_N * self.block_sum, in_C, self.overlapped_block_h, self.overlapped_block_w), ctx=ctx)
            CuDNN_conv2d_with_bias_sparse(
                input_vals[0], input_vals[1], input_vals[2], output_val,
                gather_index, scatter_index, block_sum,
                overlapped_block_h, overlapped_block_w, 
                gather_map, scatter_map, self.padding, self.stride,
                self.activation_mode, scale, shift, stream_handle)

        if self.config.profile:  
            torch.cuda.synchronize() 
            time_end = timer()
            time_elapse = time_end - time_start
            time_key = (self.op_name, self.latent_scale)
            if time_key not in Conv2dAddBiasActivateOp.profile_dict:
                Conv2dAddBiasActivateOp.profile_dict[time_key] = time_elapse
            else:
                Conv2dAddBiasActivateOp.profile_dict[time_key] += time_elapse
            if self.round == self.limit_2 - 1:
                f_save = open('profile_conv.pkl', 'wb')
                pickle.dump(Conv2dAddBiasActivateOp.profile_dict, f_save)
                f_save.close()

        self.round += 1



    def compute(self, input_vals, output_val, stream_handle=None):
        ctx = input_vals[0].ctx

        if self.use_sparse and self.round >= self.limit_1 and self.round < self.limit_2:
            self.compute_edit(input_vals, output_val, stream_handle)
            return  

        # Fuse GN
        if self.fuse_gn:
            # [N, C, H, W] ---> mean/var:[N, num_groups, num_res]
            gn_shape = list(input_vals[0].shape)
            assert gn_shape[1] % self.num_groups == 0
            num_per_group = gn_shape[1] // self.num_groups
            if self.mean is None:
                num_res = num_per_group
                for s in gn_shape[2:]:
                    num_res *= s
                gn_shape = gn_shape[:1] + [self.num_groups, num_res]
                gn_shape = tuple(gn_shape)
                self.mean = ht.empty(gn_shape[:2], ctx=ctx)
                self.var = ht.empty(gn_shape[:2], ctx=ctx)
            if self.gn_output is None: 
                self.gn_output = ht.empty(input_vals[0].shape, ctx=input_vals[0].ctx)
            mean, var, gn_output = self.mean, self.var, self.gn_output

            group_normalization(input_vals[0], input_vals[3], input_vals[4], self.num_groups,
                        mean, var, gn_output, self.eps, stream_handle)

            if not self.use_sparse:
                # transform the GN to scale + shift.
                mean = torch.tensor(mean.asnumpy())
                var = torch.tensor(np.sqrt(var.asnumpy() + self.eps))
                # Damn! Should use repeat_interleave rather than repeat!
                mean = mean.repeat_interleave(num_per_group, -1)
                var = var.repeat_interleave(num_per_group, -1)
                weight = torch.tensor(input_vals[3].asnumpy()).view(-1)
                bias = torch.tensor(input_vals[4].asnumpy()).view(-1)
                gn_scale = weight / var
                gn_shift = -(mean / var) * weight + bias
            
                gn_scale = ht.array(gn_scale.numpy(), ctx=ctx)
                gn_shift = ht.array(gn_shift.numpy(), ctx=ctx)
                self.gn_scale_cache.append(gn_scale)
                self.gn_shift_cache.append(gn_shift)
        else:
            gn_output = input_vals[0]

        # Fuse activation Func
        if self.activation_mode == 1:
            silu(gn_output, gn_output, stream_handle)

        # Conv
        CuDNN_conv2d_with_bias(gn_output, input_vals[1], input_vals[2],
                        output_val, self.padding, self.stride, stream_handle)

        if not self.use_sparse and self.cache_ctx == self.ctx:
            pass
            # output_val.copyto(self.output_cache[self.round])
        elif not self.use_sparse and self.cache_ctx == ht.cpu() and self.d2h_stream is not None:
            self.output_cache[self.round].async_d2h(output_val, stream_handle=self.d2h_stream)

        self.round += 1

    def gradient(self, output_grad):
        return [conv2d_gradient_of_data_op(self.inputs[1], output_grad, self.inputs[0], self.padding, self.stride, ctx=self.raw_ctx),
                conv2d_gradient_of_filter_op(
                    self.inputs[0], output_grad, self.inputs[1], self.padding, self.stride, ctx=self.raw_ctx),
                reduce_sum_op(output_grad, [0, 2, 3], ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        if self.fuse_gn:
            assert len(input_shapes) == 5
        else:
            assert len(input_shapes) == 3
        N, _, H, W = input_shapes[0]
        f_O, _, f_H, f_W = input_shapes[1]
        assert len(input_shapes[2]) == 1 and input_shapes[2][0] == f_O
        padding = self.padding
        stride = self.stride
        filter_H = input_shapes[1][2]
        filter_W = input_shapes[1][3]
        out_H = (H + 2 * padding[0] - filter_H) // stride[0] + 1
        out_W = (W + 2 * padding[1] - filter_W) // stride[1] + 1
        self.output_shape = (N, f_O, out_H, out_W)
        return (N, f_O, out_H, out_W)


def conv2d_add_bias_activate_op(node_A, node_B, bias, padding=0, stride=1, 
                       activation_mode=0, gn_weight=None, gn_bias=None, 
                       num_groups=32, eps=0.01, height=None, width=None, config=None, ctx=None):
    """Conv2d-with-bias node.

    Parameters:
    ----
    node_A : Node
        Input data node.
    node_B : Node
        Input filter node.
    bias : Node
        Bias node.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2dAddBiasActivateOp(node_A, node_B, bias, padding, stride, activation_mode,
                            gn_weight, gn_bias, num_groups, eps, height, width, config, ctx=ctx)
