from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def fused_multi_head_attention(query, key, value, out_arr, num_heads, stream=None):
    assert isinstance(query, _nd.NDArray)
    assert isinstance(key, _nd.NDArray)
    assert isinstance(value, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuFusedMultiHeadAttention(query.handle, key.handle, value.handle,
                             out_arr.handle, num_heads, stream.handle if stream else None)
