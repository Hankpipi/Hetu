from __future__ import absolute_import
import torch
import numpy as np

from ..ndarray import array, empty, cpu
from .Node import Op
from ..gpu_links import silu, group_normalization, CuDNN_conv2d_with_bias, matmul_with_bias, \
                        array_reshape, repeat, matrix_elementwise_add, matrix_elementwise_multiply_by_const, \
                        gather_for_conv, scatter_for_conv, norm_silu



class ResNet(Op):
    def __init__(self, node_A, temb, resnet, config, ctx=None):
        super().__init__(ResNet, [node_A, temb], ctx)
        self.round = 0
        self.outdeg = 0
        self.mask_rate = 0
        self.built = False
        self.use_sparse = False
        self.output_scale_factor = resnet.output_scale_factor
        self.use_in_shortcut = resnet.use_in_shortcut
        self.num_groups1 = resnet.norm1.num_groups
        self.num_groups2 = resnet.norm2.num_groups
        self.eps1 = resnet.norm1.eps
        self.eps2 = resnet.norm2.eps
        self.config = config
        self.stride = (1, 1)
        self.padding = (1, 1)
        assert(resnet.time_embedding_norm == "default")

        self.gn1_weights = array(resnet.norm1.weight, ctx=ctx)
        self.gn1_bias = array(resnet.norm1.bias, ctx=ctx)

        self.conv1_weights = array(resnet.conv1.weight, ctx=ctx)
        self.conv1_bias = array(resnet.conv1.bias, ctx=ctx)

        self.gn2_weights = array(resnet.norm2.weight, ctx=ctx)
        self.gn2_bias = array(resnet.norm2.bias, ctx=ctx)

        self.conv2_weights = array(resnet.conv2.weight, ctx=ctx)
        self.conv2_bias = array(resnet.conv2.bias, ctx=ctx)

        self.temb_proj_weights = array(resnet.time_emb_proj.weight, ctx=ctx)
        self.temb_proj_bias = array(resnet.time_emb_proj.bias, ctx=ctx)

        if self.use_in_shortcut:
            self.conv_shortcut_weights = array(resnet.conv_shortcut.weight, ctx=ctx)
            self.conv_shortcut_bias = array(resnet.conv_shortcut.bias, ctx=ctx)
        
        # sparse config
        self.d2h_stream = None
        if config.height is None or config.height <= 24:
            self.block_division_h = 12
            self.block_division_w = 12
        else:
            self.block_division_h = config.height // 2
            self.block_division_w = config.width // 2
        self.gn_scale_cache = []
        self.gn_shift_cache = []
        self.output_cache = []


    def init_gn_workspace(self, input_shape, num_groups):
        gn_shape = list(input_shape)
        assert gn_shape[1] % num_groups == 0
        num_per_group = gn_shape[1] // num_groups
        num_res = num_per_group
        for s in gn_shape[2:]:
            num_res *= s
        gn_shape = gn_shape[:1] + [num_groups, num_res]
        gn_shape = tuple(gn_shape)
        mean = empty(gn_shape[:2], ctx=self.ctx)
        var = empty(gn_shape[:2], ctx=self.ctx)
        return mean, var

    def build_sparse(self, input_vals, output_val):
        if self.built:
            return 
        self.built = True
        ctx = self.ctx
        mask = torch.load(f'runtime/mask.pt')

        in_N, in_C, in_H, in_W = input_vals[0].shape
        filter_out_C, filter_in_C, filter_H, filter_W = self.conv1_weights.shape
        out_N, out_C, out_H, out_W = output_val.shape
        # Scatter settings.
        # By default, we divide it into 12 * 12 blocks.
        # Assume the original latent's size is no less than 96 * 96.
        block_h = out_H // self.block_division_h
        block_w = out_W // self.block_division_w
        mask = torch.nn.functional.interpolate(
            mask.repeat(1, 1, 1, 1), size=(out_H, out_W)
        )
        block_mask = torch.nn.MaxPool2d(kernel_size=(block_h, block_w))(mask.float()) 
        assert (block_mask.shape[-2] == self.block_division_h and block_mask.shape[-1] == self.block_division_w)
        block_mask = (block_mask > 0.5)
        block_mask = block_mask.numpy()[0][0]
        index_arr = list(np.where(block_mask == True))
        index_arr[0] *= block_h
        index_arr[1] *= block_w
        block_sum = len(index_arr[0])
        self.scatter_index = array(np.concatenate((index_arr[0], index_arr[1])), ctx=ctx)
        self.scatter_map2 = empty((out_N * block_sum, out_C, block_h, block_w), ctx=ctx)

        # Gather settings.
        # We need to calculate the overlapped block size considering the conv kernel.
        overlapped_block_h = (block_h - 1) * self.stride[0] + filter_H
        overlapped_block_w = (block_w - 1) * self.stride[1] + filter_W
        self.scatter_map1 = empty((in_N * block_sum, filter_out_C,
                                   overlapped_block_h, overlapped_block_w), ctx=ctx)
        self.temb_sparse = empty((in_N * block_sum, self.temb_proj.shape[1],
                                  overlapped_block_h, overlapped_block_w), ctx=self.ctx)
        overlapped_block_h += filter_H - self.stride[0]
        overlapped_block_w += filter_W - self.stride[1]

        new_index_arr = index_arr.copy()
        new_index_arr[0] = new_index_arr[0] * self.stride[0] - self.padding[0]
        new_index_arr[1] = new_index_arr[1] * self.stride[1] - self.padding[1]
        self.gather_index = array(np.concatenate((new_index_arr[0], new_index_arr[1])), ctx=ctx)
        self.gather_map = empty((in_N * block_sum, in_C, overlapped_block_h, overlapped_block_w), ctx=ctx)

        self.block_sum = block_sum
        self.block_h = overlapped_block_h
        self.block_w = overlapped_block_w


    def compute_sparse(self, input_vals, output_val, stream_handle=None):
        self.build_sparse(input_vals, output_val)
        scale1, scale2 = self.gn_scale_cache[self.round]
        shift1, shift2 = self.gn_shift_cache[self.round]

        gather_for_conv(input_vals[0], self.gather_index, self.gather_map, scale1, shift1, self.block_sum,
                        self.block_h, self.block_w, stream_handle)
        CuDNN_conv2d_with_bias(self.gather_map, self.conv1_weights, self.conv1_bias, self.scatter_map1,
                               padding=(0, 0), stride=(1, 1), stream=stream_handle)

        # # temb
        silu(input_vals[1], self.temb0, stream_handle)
        matmul_with_bias(self.temb0, False, self.temb_proj_weights, True, self.temb_proj_bias, self.temb_proj, stream_handle)
        array_reshape(self.temb_proj, self.temb_reshape, stream_handle)
        repeat(self.temb_reshape, self.temb_sparse, stream_handle)
        matrix_elementwise_add(self.scatter_map1, self.temb_sparse, self.scatter_map1, stream=stream_handle)

        norm_silu(self.scatter_map1, self.scatter_map1, scale2, shift2, self.block_sum, 1, stream_handle)
        CuDNN_conv2d_with_bias(self.scatter_map1, self.conv2_weights, self.conv2_bias, self.scatter_map2,
                               padding=(0, 0), stride=(1, 1), stream=stream_handle)

        shortcut = input_vals[0]
        if self.use_in_shortcut:
            CuDNN_conv2d_with_bias(shortcut, self.conv_shortcut_weights, self.conv_shortcut_bias, self.shortcut,
                                   padding=(0, 0), stride=(1, 1), stream=stream_handle)
            shortcut = self.shortcut

        self.event.sync()
        scatter_for_conv(self.scatter_map2, output_val, self.scatter_index, self.block_sum, self.stride, stream_handle)

        matrix_elementwise_add(output_val, shortcut, output_val, stream=stream_handle)
        matrix_elementwise_multiply_by_const(output_val, 1 / self.output_scale_factor, output_val, stream_handle)

        self.round += 1

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        if self.use_sparse and self.round >= 10:
            self.compute_sparse(input_vals, output_val, stream_handle)
            return 

        group_normalization(input_vals[0], self.gn1_weights, self.gn1_bias, self.num_groups1,
                            self.mean1, self.var1, self.input_buffer, self.eps1, stream_handle)
        silu(self.input_buffer, self.input_buffer, stream_handle)
        CuDNN_conv2d_with_bias(self.input_buffer, self.conv1_weights, self.conv1_bias, self.output_conv1,
                               padding=(1, 1), stride=(1, 1), stream=stream_handle)
        
        # temb
        silu(input_vals[1], self.temb0, stream_handle)
        matmul_with_bias(self.temb0, False, self.temb_proj_weights, True, self.temb_proj_bias, self.temb_proj, stream_handle)
        array_reshape(self.temb_proj, self.temb_reshape, stream_handle)
        repeat(self.temb_reshape, self.temb, stream_handle)
        matrix_elementwise_add(self.output_conv1, self.temb, self.output_conv1, stream=stream_handle)

        group_normalization(self.output_conv1, self.gn2_weights, self.gn2_bias, self.num_groups2,
                            self.mean2, self.var2, self.output_conv1, self.eps2, stream_handle)
        silu(self.output_conv1, self.output_conv1, stream_handle)
        CuDNN_conv2d_with_bias(self.output_conv1, self.conv2_weights, self.conv2_bias, output_val,
                               padding=(1, 1), stride=(1, 1), stream=stream_handle)

        # save checkpoints
        if not self.use_sparse and self.d2h_stream is not None:
            output_cached = empty(output_val.shape, ctx=cpu())
            output_cached.async_d2h(output_val, stream_handle=self.d2h_stream)
            self.output_cache.append(output_cached)

            def transform(mean, var, num_per_group, weight, bias, eps, ctx):
                # transform the GN to scale + shift.
                mean = torch.tensor(mean.asnumpy())
                var = torch.tensor(np.sqrt(var.asnumpy() + eps))
                # Damn! Should use repeat_interleave rather than repeat!
                mean = mean.repeat_interleave(num_per_group, -1)
                var = var.repeat_interleave(num_per_group, -1)
                weight = torch.tensor(weight.asnumpy()).view(-1)
                bias = torch.tensor(bias.asnumpy()).view(-1)
                gn_scale = weight / var
                gn_shift = -(mean / var) * weight + bias
            
                gn_scale = array(gn_scale.numpy(), ctx=ctx)
                gn_shift = array(gn_shift.numpy(), ctx=ctx)

                return gn_scale, gn_shift

            gn1_scale, gn1_shift = transform(self.mean1, self.var1, input_vals[0].shape[1] // self.num_groups1,
                                             self.gn1_weights, self.gn1_bias, self.eps1, self.ctx)
            gn2_scale, gn2_shift = transform(self.mean2, self.var2, self.output_conv1.shape[1] // self.num_groups2,
                                             self.gn2_weights, self.gn2_bias, self.eps2, self.ctx)
            self.gn_scale_cache.append((gn1_scale, gn2_scale))
            self.gn_shift_cache.append((gn1_shift, gn2_shift))

        shortcut = input_vals[0]
        if self.use_in_shortcut:
            CuDNN_conv2d_with_bias(shortcut, self.conv_shortcut_weights, self.conv_shortcut_bias, self.shortcut,
                                   padding=(0, 0), stride=(1, 1), stream=stream_handle)
            shortcut = self.shortcut
        matrix_elementwise_add(output_val, shortcut, output_val, stream=stream_handle)
        matrix_elementwise_multiply_by_const(output_val, 1 / self.output_scale_factor, output_val, stream_handle)

        self.round += 1

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        HW = input_shapes[0][2:]
        output_shape = (self.config.batch, self.conv2_bias.shape[0]) + HW
        output_conv1_shape = (self.config.batch, self.conv1_bias.shape[0]) + HW
        self.mean1, self.var1 = self.init_gn_workspace(input_shapes[0], self.num_groups1)
        self.mean2, self.var2 = self.init_gn_workspace(output_conv1_shape, self.num_groups2)
        self.output_conv1 = empty(output_conv1_shape, ctx=self.ctx)
        self.input_buffer = empty(input_shapes[0], ctx=self.ctx)
        if self.use_in_shortcut:
            self.shortcut = empty(output_shape, ctx=self.ctx)

        self.temb0 = empty(input_shapes[1], ctx=self.ctx)
        self.temb_proj = empty((self.config.batch, self.temb_proj_bias.shape[0]), ctx=self.ctx)
        self.temb_reshape = empty(self.temb_proj.shape + (1, 1), ctx=self.ctx)
        self.temb = empty(self.temb_proj.shape + HW, ctx=self.ctx)
        return output_shape


def resnet(node_A, temb, resnet, config, ctx=None):
    """Fused resnet node.

    Parameters:
    ----
    node_A : Node
        Input data node.
    temb : Node
        Time embedding.
    resnet : Torch.nn.Module
        Pytorch module.
    config :
        Hetu unet config.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ResNet(node_A, temb, resnet, config, ctx=ctx)
