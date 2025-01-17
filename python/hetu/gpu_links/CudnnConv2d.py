from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def CuDNN_conv2d(in_arr_x, in_arr_f, out_arr, padding=0, stride=1, stream=None):
    assert isinstance(in_arr_x, _nd.NDArray)
    assert isinstance(in_arr_f, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.CuDNN_DLGpuConv2d(in_arr_x.handle, in_arr_f.handle,
                           out_arr.handle, padding, stride, stream.handle if stream else None)


def CuDNN_conv2d_gradient_of_filter(in_arr_x, in_gradient_y, out_gradient_f, padding=0, stride=1, stream=None):
    assert isinstance(in_arr_x, _nd.NDArray)
    assert isinstance(in_gradient_y, _nd.NDArray)
    assert isinstance(out_gradient_f, _nd.NDArray)
    _LIB.CuDNN_DLGpuConv2d_Gradient_of_Filter(
        in_arr_x.handle, in_gradient_y.handle, out_gradient_f.handle, padding, stride, stream.handle if stream else None)


def CuDNN_conv2d_gradient_of_data(in_arr_f, in_gradient_y, out_gradient_x, padding=0, stride=1, stream=None):
    assert isinstance(in_arr_f, _nd.NDArray)
    assert isinstance(in_gradient_y, _nd.NDArray)
    assert isinstance(out_gradient_x, _nd.NDArray)
    _LIB.CuDNN_DLGpuConv2d_Gradient_of_Data(
        in_arr_f.handle, in_gradient_y.handle, out_gradient_x.handle, padding, stride, stream.handle if stream else None)
