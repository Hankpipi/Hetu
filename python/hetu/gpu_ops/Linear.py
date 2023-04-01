from __future__ import absolute_import
import os
import math
import numpy as np
from scipy import interpolate

import torch
import hetu as ht
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import matmul_with_bias, matmul_with_bias_sparse, layer_normalization, gelu_half


class LinearOp(Op):

    index_pool = {}
    input_pool = {}
    output_pool = {}

    def __init__(self, node_A, node_B, bias, trans_A=False, trans_B=False, name=None,
                 activation_mode=0, ln_weight=None, ln_bias=None, eps=0.01, ctx=None):
        if ln_weight == None:
            super().__init__(LinearOp, [node_A, node_B, bias], ctx)
        else:
            super().__init__(LinearOp, [node_A, node_B, bias, ln_weight, ln_bias], ctx)

        # Linear
        self.name = name
        self.matmul_attr_trans_A = trans_A
        self.matmul_attr_trans_B = trans_B
        self.round = 0
        self.outdeg = 0
        self.use_sparse = False
        self.d2h_stream = None
        self.output_cache = []

        # Activation
        # 0 stands for Identity function.
        # 1 stands for GeLU.
        self.activation_mode = activation_mode

        # LN
        self.mean = None
        self.var = None
        self.ln_output = None
        self.fuse_ln = (ln_weight != None)
        self.eps = eps

        # CrossAttn key and value reuse.
        self.crossattn_reuse = None

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            input_vals = [n.asnumpy() for n in input_vals]
            if ((self.matmul_attr_trans_A is False) and
                    (self.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    input_vals[0], input_vals[1]) + input_vals[2]
            elif ((self.matmul_attr_trans_A is True) and
                    (self.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), input_vals[1]) + input_vals[2]
            elif ((self.matmul_attr_trans_A is False) and
                    (self.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    input_vals[0], np.transpose(input_vals[1])) + input_vals[2]
            elif ((self.matmul_attr_trans_A is True) and
                    (self.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), np.transpose(input_vals[1])) + input_vals[2]
        else:

            if self.crossattn_reuse != None:
                # It will be much slower if we use copy!
                # self.crossattn_reuse.copyto(output_val)
                output_val = self.crossattn_reuse
                return 
            
            ctx = input_vals[0].ctx
            if self.use_sparse and self.round >= 10:
                B, L, input_channel = input_vals[0].shape
                output_channel = output_val.shape[-1]
                if L not in LinearOp.index_pool:
                    mask = torch.load(f'runtime/mask.pt')
                    width = int(math.sqrt(output_val.shape[-2]))
                    rate = 96 // width

                    mask = torch.nn.functional.interpolate(
                        mask.repeat(B, 1, 1, 1), size=(96, 96)
                    )
                    
                    mask = torch.nn.MaxPool2d(kernel_size=rate)(mask.float())
                    mask = (mask > 0.5)

                    mask = mask.numpy().reshape(-1)

                    index = np.where(mask == True)
                    index = ht.array(index[0], ctx=ctx)
                    LinearOp.index_pool[L] = index
                    input_sparse = ht.empty(index.shape + (input_channel, ), ctx=ctx)
                    output_sparse = ht.empty(index.shape + (output_channel, ), ctx=ctx)
                else:
                    index = LinearOp.index_pool[L]

                input_shape = index.shape + (input_channel, )
                output_shape = index.shape + (output_channel, )
                if input_shape in LinearOp.input_pool:
                    input_sparse = LinearOp.input_pool[input_shape]
                else:
                    input_sparse = ht.empty(input_shape, ctx=ctx)
                    LinearOp.input_pool[input_shape] = input_sparse

                if output_shape in LinearOp.output_pool:
                    output_sparse = LinearOp.output_pool[output_shape]
                else:
                    output_sparse = ht.empty(output_shape, ctx=ctx)
                    LinearOp.output_pool[output_shape] = output_sparse

                self.event.sync()
                ln_scale_curr = None
                ln_shift_curr = None 
                if self.fuse_ln:
                    ln_scale_curr = input_vals[3]
                    ln_shift_curr = input_vals[4]

                matmul_with_bias_sparse(
                    input_vals[0], self.matmul_attr_trans_A,
                    input_vals[1], self.matmul_attr_trans_B, input_vals[2],
                    output_val, index, input_sparse, output_sparse, 
                    ln_scale_curr, ln_shift_curr, self.eps, 
                    self.activation_mode, stream_handle)
            else:
                # Fuse LN
                if self.fuse_ln:
                    if self.mean is None:
                        self.mean = ht.empty(input_vals[0].shape[:2], ctx=ctx)
                        self.var = ht.empty(input_vals[0].shape[:2], ctx=ctx)
                        self.ln_output = ht.empty(input_vals[0].shape, ctx=ctx)
                    ln_output = self.ln_output
                    layer_normalization(input_vals[0], input_vals[3], input_vals[4],
                                self.mean, self.var, ln_output, self.eps, stream_handle)
                else:
                    ln_output = input_vals[0]

                # Linear
                matmul_with_bias(
                    ln_output, self.matmul_attr_trans_A,
                    input_vals[1], self.matmul_attr_trans_B, input_vals[2],
                    output_val, stream_handle)
                
                # Fuse activation Func
                # GeLU (half of the hidden_states)
                if self.activation_mode == 1:
                    gelu_half(output_val, output_val, stream_handle)

                if not self.use_sparse and self.d2h_stream is not None:
                    output_cached = ht.empty(output_val.shape, ctx=ht.cpu())
                    output_cached.async_d2h(output_val, stream_handle=self.d2h_stream)
                    self.output_cache.append(output_cached)

            if (self.name == 'CrossAttn_k' or self.name == 'CrossAttn_v') and self.crossattn_reuse == None:
                self.crossattn_reuse = ht.empty(output_val.shape, ctx=ctx)
                output_val.copyto(self.crossattn_reuse)
            self.round += 1

    def gradient(self, output_grad):
        from .MatrixMult import matmul_op
        from .ReduceSum import reduce_sum_op
        if ((self.matmul_attr_trans_A is False) and
                (self.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = matmul_op(
                output_grad, self.inputs[1], trans_A=False, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                self.inputs[0], output_grad, trans_A=True, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is True) and
                (self.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            lhs_grad = matmul_op(
                self.inputs[1], output_grad, trans_A=False, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                self.inputs[0], output_grad, trans_A=False, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is False) and
                (self.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                output_grad, self.inputs[1], trans_A=False, trans_B=False, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                output_grad, self.inputs[0], trans_A=True, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is True) and
                (self.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B)^T=B^T dY^T, dB=(A dY)^T=dY^T A^T
            lhs_grad = matmul_op(
                self.inputs[1], output_grad, trans_A=True, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                output_grad, self.inputs[0], trans_A=True, trans_B=True, ctx=self.raw_ctx)
        bias_grad = reduce_sum_op(
            output_grad, [0], keepdims=False, ctx=self.raw_ctx)
        return [lhs_grad, rhs_grad, bias_grad]

    def infer_shape(self, input_shapes):
        if self.fuse_ln:
            assert len(input_shapes) == 5
        else:
            assert len(input_shapes) == 3
        assert len(input_shapes[1]) == 2
        assert len(input_shapes[2]) == 1
        A = input_shapes[0]
        B = input_shapes[1]
        bias_shape = input_shapes[2]
        shape_A = A[:-1]
        shape_B = (B[1], )
        if self.matmul_attr_trans_A == True:
            assert(len(input_shapes[0]) == 2)
            shape_A = (A[1], )
        if self.matmul_attr_trans_B == True:
            shape_B = (B[0], )
        assert bias_shape == shape_B
        return shape_A + shape_B


def linear_op(node_A, node_B, bias, trans_A=False, trans_B=False, name=None,
              activation_mode=0, ln_weight=None, ln_bias=None, eps=0.01, ctx=None):
    """Make a new instance of Matrix Multiplication with bias and call the instance.

    Parameters:
    ----
    node_A : Node
        The left operand of the matrix multiplication.
    node_B : Node
        The right operand of the matrix multiplication.
    bias : Node
        The bias of linear operation.
    trans_A : Boolean
        Whether node_A to be transposed
    trans_B : Boolean
        Whether node_B to be transposed

    Returns:
    ----
    A new Node instance created by Op.

    """
    return LinearOp(node_A, node_B, bias, trans_A, trans_B, name,
                    activation_mode, ln_weight, ln_bias, eps, ctx=ctx)
