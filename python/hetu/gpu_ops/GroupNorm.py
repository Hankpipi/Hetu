from __future__ import absolute_import
from .Node import Op
import numpy as np
from .. import ndarray
from ..gpu_links import group_normalization


class Group_NormalizationOp(Op):
    def __init__(self, node_in, ln_scale, ln_bias, num_groups, eps=0.01, ctx=None):
        super().__init__(Group_NormalizationOp,
                         [node_in, ln_scale, ln_bias], ctx)
        self.eps = eps
        self.num_groups = num_groups
        self.save_mean = None
        self.save_var = None
        self.data_shape = None

    def compute(self, input_vals, output_val, stream_handle=None):
        # default input shape is NCHW
        origin_shape = input_vals[0].shape
        local_shape = list(input_vals[0].shape)
        assert local_shape[1] % self.num_groups == 0
        num_per_group = int(local_shape[1] / self.num_groups)
        num_res = num_per_group
        for s in local_shape[2:]:
            num_res *= s
        local_shape = local_shape[:1] + [self.num_groups, num_res]
        local_shape = tuple(local_shape)

        if self.on_cpu:
            input_vals = [n.asnumpy() for n in input_vals]
            data_type = input_vals[0].dtype
            input_vals[0] = input_vals[0].reshape(local_shape)

            if self.data_shape is None:
                self.save_mean = np.empty(local_shape, dtype=np.float32)
                self.save_var = np.empty(local_shape, dtype=np.float32)
                self.data_shape = local_shape
            elif self.data_shape != local_shape:
                del self.save_mean
                del self.save_var
                self.save_mean = np.empty(local_shape, dtype=np.float32)
                self.save_var = np.empty(local_shape, dtype=np.float32)
                self.data_shape = local_shape
            self.save_mean[:] = input_vals[0].mean(
                axis=2, dtype=data_type, keepdims=True)
            self.save_var[:] = input_vals[0].var(
                axis=2, dtype=data_type, keepdims=True)
            std = np.sqrt(self.save_var + self.eps, dtype=data_type)
            centered_input = input_vals[0] - self.save_mean
            normed_input = centered_input / std

            bc_shape = [1] * len(origin_shape)
            bc_shape[1] = self.num_groups * num_per_group
            normed_input = normed_input.reshape(origin_shape)

            output_val[:] = input_vals[1].reshape(bc_shape) * normed_input + \
                input_vals[2].reshape(bc_shape)

        else:
            if self.data_shape is None:
                dev_id = input_vals[0].handle.contents.ctx.device_id
                self.save_mean = ndarray.empty(
                    local_shape, ctx=ndarray.gpu(dev_id))
                self.save_var = ndarray.empty(
                    local_shape, ctx=ndarray.gpu(dev_id))
                self.data_shape = local_shape
            elif self.data_shape != local_shape:
                del self.save_mean
                del self.save_var
                dev_id = input_vals[0].handle.contents.ctx.device_id
                self.save_mean = ndarray.empty(
                    local_shape, ctx=ndarray.gpu(dev_id))
                self.save_var = ndarray.empty(
                    local_shape, ctx=ndarray.gpu(dev_id))
                self.data_shape = local_shape
            group_normalization(input_vals[0], input_vals[1], input_vals[2], self.num_groups,
                                self.save_mean, self.save_var, output_val, self.eps, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def group_normalization_op(node_in, ln_scale, ln_bias, num_groups, eps=0.01, ctx=None):
    """Layer normalization node.

    Parameters:
    ----
    node_in : Node
        Input data.
    ln_scale : float
        scaling parameter
    ln_bias :
        learnable bias parameter
    eps : float
        Epsilon value for numerical stability.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Group_NormalizationOp(node_in, ln_scale, ln_bias, num_groups, eps, ctx=ctx)
