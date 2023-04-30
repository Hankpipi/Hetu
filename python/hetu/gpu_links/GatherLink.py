
from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def gather(in_arr, index, out_arr, dim, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuGather(in_arr.handle, index.handle, out_arr.handle,
                     ctypes.c_int(dim), stream.handle if stream else None)


def gather_gradient(in_arr, index, out_arr, dim, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuGatherGradient(in_arr.handle, index.handle, out_arr.handle, ctypes.c_int(
        dim), stream.handle if stream else None)


def gather_for_conv(in_arr, index, out_arr, scale, shift, block_sum, block_h, block_w, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuGatherForConv(in_arr.handle, out_arr.handle, index.handle,
                     scale.handle, shift.handle, block_sum, block_h, block_w,
                     stream.handle if stream else None)

def gather_for_linear(in_arr, index, out_arr, scale, shift, eps, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuGatherForLinear(in_arr.handle, out_arr.handle, index.handle,
                     scale.handle, shift.handle, ctypes.c_float(eps),
                     stream.handle if stream else None)
    
def gather_for_linear_simple(in_arr, index, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuGatherForLinearBoth(in_arr.handle, out_arr.handle, index.handle,
                     stream.handle if stream else None)