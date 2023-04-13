from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matmul_with_bias(matA, transA, matB, transB, bias, matC, stream=None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(bias, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuLinear(
        matA.handle, transA, matB.handle, transB, bias.handle, matC.handle, stream.handle if stream else None)

def matmul_with_bias_sparse(matA, transA, matB, transB, bias, matC, index, matA_sparse, matC_sparse, 
                            scale=None, shift=None, eps=0.01, activation_mode=0, stream=None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(bias, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuLinearSparse(
        matA.handle, transA, matB.handle, transB, bias.handle, matC.handle, 
            index.handle, matA_sparse.handle, matC_sparse.handle, 
            scale.handle if scale else None, shift.handle if shift else None,
            ctypes.c_float(eps), activation_mode, stream.handle if stream else None)


def matmul_qkv(matA, transA, matB, transB, bias, matC, matQ, matK, matV, stream=None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(bias, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    assert isinstance(matQ, _nd.NDArray)
    assert isinstance(matK, _nd.NDArray)
    assert isinstance(matV, _nd.NDArray)
    _LIB.DLGpuLinearQKV(
        matA.handle, transA, matB.handle, transB, bias.handle, matC.handle, 
            matQ.handle, matK.handle, matV.handle, stream.handle if stream else None)

