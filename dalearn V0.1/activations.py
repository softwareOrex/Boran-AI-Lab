from .tensor import Tensor


def relu(x):

    result = Tensor(
        max(0, x.data),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="relu"
    )

    def backward():

        if x.requires_grad:

            if x.data > 0:
                x.grad += result.grad

    result._backward = backward

    return result
