from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import silu

class SiluOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(SiluOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = input_vals[0].asnumpy()
            output_val = output_val / (1 + np.exp(-output_val))
        else:
            silu(input_vals[0], output_val, stream_handle)
        if stream_handle:
            stream_handle.sync()

    def gradient(self, output_grad):
        assert False, "not implement"

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def silu_op(node, ctx=None):
    """Rectified Linear Unit.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SiluOp(node, ctx=ctx)

