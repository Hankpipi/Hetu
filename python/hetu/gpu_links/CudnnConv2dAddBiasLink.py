from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def CuDNN_conv2d_with_bias(in_arr_x, in_arr_f, bias, out_arr, padding=(0, 0), stride=(1, 1), stream=None):
    assert isinstance(in_arr_x, _nd.NDArray)
    assert isinstance(in_arr_f, _nd.NDArray)
    assert isinstance(bias, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.Cudnn_Conv2dAddBias(in_arr_x.handle, in_arr_f.handle, bias.handle,
                             out_arr.handle, padding[0], padding[1], stride[0], stride[1], stream.handle if stream else None)

def CuDNN_conv2d_with_bias_sparse(in_arr_x, in_arr_f, bias, out_arr, 
                    gather_index, scatter_index, block_sum, block_h, block_w,
                    gather_map, scatter_map, padding=(0, 0), stride=(1, 1), 
                    activation_mode=0, scale=None, shift=None, stream=None):
    assert isinstance(in_arr_x, _nd.NDArray)
    assert isinstance(in_arr_f, _nd.NDArray)
    assert isinstance(bias, _nd.NDArray)
    assert isinstance(gather_index, _nd.NDArray)
    assert isinstance(scatter_index, _nd.NDArray)
    assert isinstance(gather_map, _nd.NDArray)
    assert isinstance(scatter_map, _nd.NDArray)
    _LIB.Cudnn_Conv2dAddBiasSparse(in_arr_x.handle, in_arr_f.handle, bias.handle,
                             out_arr.handle, gather_index.handle, scatter_index.handle,
                             block_sum, block_h, block_w,
                             gather_map.handle, scatter_map.handle,
                             padding[0], padding[1], stride[0], stride[1], 
                             activation_mode, scale.handle if scale else None,
                             shift.handle if shift else None,
                             stream.handle if stream else None)
