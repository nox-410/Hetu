from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from .Softmax import softmax_func
from ..gpu_links import cross_entropy
from ..gpu_links import cross_entropy_gradient


class CrossEntropyOp(Op):
    def __init__(self, node_y, node_y_, ctx=None):
        super().__init__(CrossEntropyOp, [node_y, node_y_], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        y = input_vals[0]
        y_ = input_vals[1]
        if self.on_cpu:
            output_val[:] = -np.sum(y_.asnumpy() * np.log(y.asnumpy()), axis=1)
        else:
            cross_entropy(y, y_, output_val, stream_handle)

    def gradient(self, output_grad):
        grad_A = crossentropy_gradient_op(
            output_grad, self.inputs[0], self.inputs[1], ctx=self.raw_ctx)
        return [grad_A, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[0]) >= 2
        return input_shapes[0][:-1]

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        # only allow data parallel in softmax cross entropy
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses:
            if nst.valid(deduce_order):
                nst.check_state(1, deduce_order)
                status.copy_from(nst, deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        # only allow data parallel in softmax cross entropy
        assert len(input_statuses) == len(self.inputs)
        if status.valid(deduce_order):
            status.check_state(1, deduce_order)
            for nst in input_statuses:
                nst.copy_from(status, deduce_order)


class CrossEntropyGradientOp(Op):
    def __init__(self, node_grad, node_y, node_y_, ctx=None):
        super().__init__(CrossEntropyGradientOp,
                         [node_grad, node_y, node_y_], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            grad = input_vals[0].asnumpy()
            y = input_vals[1].asnumpy()
            y_ = input_vals[2].asnumpy()
            output_val[:] = - y_ / y * np.expand_dims(grad, -1)
        else:
            crossentropy_gradient_op(
                input_vals[0], input_vals[1], input_vals[2], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[1]

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        # only allow data parallel in softmax cross entropy
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses:
            if nst.valid(deduce_order):
                nst.check_state(1, deduce_order)
                status.copy_from(nst, deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        # only allow data parallel in softmax cross entropy
        assert len(input_statuses) == len(self.inputs)
        if status.valid(deduce_order):
            status.check_state(1, deduce_order)
            for nst in input_statuses:
                nst.copy_from(status, deduce_order)


def crossentropy_op(node_y, node_y_, ctx=None):
    """Computes cross entropy loss for pre-softmax activations.

    Parameters:
    ----
    node_A : Node
        Predicted probability.
    node_B : Node
        Labels.

    Returns:
    ----
    A new Node instance created by Op.

    """

    return CrossEntropyOp(node_y, node_y_, ctx=ctx)


def crossentropy_gradient_op(node_grad, node_y, node_y_, ctx=None):
    return CrossEntropyGradientOp(node_grad, node_y, node_y_, ctx=ctx)
