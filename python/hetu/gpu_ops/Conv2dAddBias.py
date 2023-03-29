from __future__ import absolute_import
import numpy as np
from scipy import interpolate

import os
import torch
import hetu as ht
from .Node import Op
from .Conv2d import conv2d_gradient_of_data_op, conv2d_gradient_of_filter_op
from .ReduceSum import reduce_sum_op
from ..gpu_links import CuDNN_conv2d_with_bias, CuDNN_conv2d_with_bias_sparse


class Conv2dAddBiasOp(Op):
    def __init__(self, node_A, node_B, bias, padding=0, stride=1, ctx=None):
        super().__init__(Conv2dAddBiasOp, [node_A, node_B, bias], ctx)
        if not isinstance(padding, tuple):
            assert isinstance(padding, int)
            padding = (padding, padding)
        self.padding = padding
        if not isinstance(stride, tuple):
            assert isinstance(stride, int)
            stride = (stride, stride)
        self.stride = stride

        self.round = 0
        self.outdeg = 0
        self.d2h_stream = None
        self.settings = False
        self.use_sparse = False
        self.gather_index = None
        self.scatter_index = None
        self.block_division_h = 12
        self.block_division_w = 12
        self.block_sum = None
        self.block_h = None
        self.block_w = None
        self.overlapped_block_h = None
        self.overlapped_block_w = None
        self.activation_mode = None
        self.scale = None
        self.shift = None
        self.output_cache = []

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

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = self.np_conv2d(
                input_vals[0].asnumpy(), input_vals[1].asnumpy(), self.padding, self.stride) +\
                input_vals[2].asnumpy().reshape((input_vals[2].shape[0], 1, 1))
        
        else:
            if self.use_sparse and self.round >= 10:

                ctx = input_vals[0].ctx
                in_N, in_C, in_H, in_W = input_vals[0].shape
                filter_out_C, filter_in_C, filter_H, filter_W = input_vals[1].shape
                out_N, out_C, out_H, out_W = output_val.shape

                if self.settings is False:
                    self.settings = True
                    mask = torch.load(f'runtime/mask.pt')

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
                    self.block_h = out_H // self.block_division_h
                    self.block_w = out_W // self.block_division_w
                    mask = torch.nn.functional.interpolate(
                        mask.repeat(1, 1, 1, 1), size=(out_H, out_W)
                    )
                    block_mask = torch.nn.MaxPool2d(kernel_size=(self.block_h, self.block_w))(mask.float()) 
                    assert (block_mask.shape[-2] == self.block_division_h and block_mask.shape[-1] == self.block_division_w)
                    block_mask = (block_mask > 0.5)
                    block_mask = block_mask.numpy()[0][0]
                    index_arr = list(np.where(block_mask == True))
                    index_arr[0] *= self.block_h
                    index_arr[1] *= self.block_w
                    self.block_sum = len(index_arr[0])
                    self.scatter_index = ht.array(np.concatenate((index_arr[0], index_arr[1])), ctx=ctx)
                    self.scatter_map = ht.empty((out_N * self.block_sum, out_C, self.block_h, self.block_w), ctx=ctx)

                    # Gather settings.
                    # We need to calculate the overlapped block size considering the conv kernel.
                    self.overlapped_block_h = (self.block_h - 1) * self.stride[0] + filter_H
                    self.overlapped_block_w = (self.block_w - 1) * self.stride[1] + filter_W
                    new_index_arr = index_arr.copy()
                    new_index_arr[0] = new_index_arr[0] * self.stride[0] - self.padding[0]
                    new_index_arr[1] = new_index_arr[1] * self.stride[1] - self.padding[1]
                    self.gather_index = ht.array(np.concatenate((new_index_arr[0], new_index_arr[1])), ctx=ctx)
                    self.gather_map = ht.empty((in_N * self.block_sum, in_C, self.overlapped_block_h, self.overlapped_block_w), ctx=ctx)

                self.event.sync()
                CuDNN_conv2d_with_bias_sparse(
                    input_vals[0], input_vals[1], input_vals[2], output_val,
                    self.gather_index, self.scatter_index, self.block_sum,
                    self.overlapped_block_h, self.overlapped_block_w, 
                    self.gather_map, self.scatter_map, self.padding, self.stride,
                    0, None, None, stream_handle)   

            else:
                CuDNN_conv2d_with_bias(input_vals[0], input_vals[1], input_vals[2],
                                output_val, self.padding, self.stride, stream_handle)

                if not self.use_sparse and self.d2h_stream is not None:
                    output_cached = ht.empty(output_val.shape, ctx=ht.cpu())
                    output_cached.async_d2h(output_val, stream_handle=self.d2h_stream)
                    self.output_cache.append(output_cached)

            self.round += 1

    def gradient(self, output_grad):
        return [conv2d_gradient_of_data_op(self.inputs[1], output_grad, self.inputs[0], self.padding, self.stride, ctx=self.raw_ctx),
                conv2d_gradient_of_filter_op(
                    self.inputs[0], output_grad, self.inputs[1], self.padding, self.stride, ctx=self.raw_ctx),
                reduce_sum_op(output_grad, [0, 2, 3], ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
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
        return (N, f_O, out_H, out_W)


def conv2d_add_bias_op(node_A, node_B, bias, padding=0, stride=1, ctx=None):
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
    return Conv2dAddBiasOp(node_A, node_B, bias, padding, stride, ctx=ctx)
