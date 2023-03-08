from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def group_normalization(in_arr, ln_scale, ln_bias, num_groups, mean, var, out_arr, eps, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(ln_scale, _nd.NDArray)
    assert isinstance(ln_bias, _nd.NDArray)
    assert isinstance(mean, _nd.NDArray)
    assert isinstance(var, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuGroupNormalization(in_arr.handle, ln_scale.handle, ln_bias.handle, ctypes.c_long(num_groups), mean.handle,
                                 var.handle, out_arr.handle, ctypes.c_float(eps), stream.handle if stream else None)
