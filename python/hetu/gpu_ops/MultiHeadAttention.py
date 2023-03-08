from __future__ import absolute_import
import numpy as np

from .Node import Op
from ..gpu_links import fused_multi_head_attention


class MultiHeadAttentionOp(Op):
    # node_A: query, node_B: key, node_C: value 
    def __init__(self, node_A, node_B, node_C, num_heads, ctx=None):
        super().__init__(MultiHeadAttentionOp, [node_A, node_B, node_C], ctx)
        self.num_heads = num_heads

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            assert NotImplementedError
        else:
            fused_multi_head_attention(input_vals[0], input_vals[1], input_vals[2],
                                       output_val, self.num_heads, stream_handle)


    def gradient(self, output_grad):
        assert NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def multi_head_attention_op(node_A, node_B, node_C, num_heads, ctx=None):
    """Conv2d-with-bias node.

    Parameters:
    ----
    node_A : Node
        Query node.
    node_B : Node
        Key node.
    node_C : Node
        Value node.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return MultiHeadAttentionOp(node_A, node_B, node_C, num_heads, ctx=ctx)
