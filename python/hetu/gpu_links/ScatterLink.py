from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def scatter(target_mat, dim, index_mat, src_mat, stream=None):
    assert isinstance(target_mat, _nd.NDArray)
    assert isinstance(index_mat, _nd.NDArray)
    assert isinstance(src_mat, _nd.NDArray)

    _LIB.DLGpuScatter(
            target_mat.handle, dim, index_mat.handle, src_mat.handle, stream.handle if stream else None)


def scatter_for_conv(in_arr, out_arr, index, blocksum, stride, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)

    _LIB.DLGpuScatterForConv(
            in_arr.handle, out_arr.handle, index.handle,
            blocksum, stride[0], stride[1], stream.handle if stream else None)
