from __future__ import absolute_import

from .._base import _LIB
from .. import ndarray as _nd


def silu(in_arr, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSilu(in_arr.handle, out_arr.handle,
                   stream.handle if stream else None)

def norm_silu(in_arr, out_arr, scale, bias, block_sum, activation_mode, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(scale, _nd.NDArray)
    assert isinstance(bias, _nd.NDArray)
    _LIB.DLGpuNormSilu(in_arr.handle, out_arr.handle, scale.handle, bias.handle,
                   block_sum, activation_mode, stream.handle if stream else None)
