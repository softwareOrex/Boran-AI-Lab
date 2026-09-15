import math


class Tensor:

    def __init__(
        self,
        data,
        requires_grad=False,
        _children=(),
        _op=""
    ):
        self.data = float(data)
        self.grad = 0.0

        self.requires_grad = requires_grad

        self._prev = set(_children)
        self._op = _op

        self._backward = lambda: None

    def __repr__(self):
        return (
            f"Tensor("
            f"data={self.data:.6f}, "
            f"grad={self.grad:.6f}"
            f")"
        )

    # =========================================
    # ADD
    # =========================================

    def __add__(self, other):

        other = self._to_tensor(other)

        result = Tensor(
            self.data + other.data,
            requires_grad=(
                self.requires_grad
                or other.requires_grad
            ),
            _children=(self, other),
            _op="+"
        )

        def backward():

            if self.requires_grad:
                self.grad += result.grad

            if other.requires_grad:
                other.grad += result.grad

        result._backward = backward

        return result

    def __radd__(self, other):
        return self + other

    # =========================================
    # SUB
    # =========================================

    def __sub__(self, other):

        other = self._to_tensor(other)

        result = Tensor(
            self.data - other.data,
            requires_grad=(
                self.requires_grad
                or other.requires_grad
            ),
            _children=(self, other),
            _op="-"
        )

        def backward():

            if self.requires_grad:
                self.grad += result.grad

            if other.requires_grad:
                other.grad -= result.grad

        result._backward = backward

        return result

    def __rsub__(self, other):
        return self._to_tensor(other) - self

    # =========================================
    # MUL
    # =========================================

    def __mul__(self, other):

        other = self._to_tensor(other)

        result = Tensor(
            self.data * other.data,
            requires_grad=(
                self.requires_grad
                or other.requires_grad
            ),
            _children=(self, other),
            _op="*"
        )

        def backward():

            if self.requires_grad:
                self.grad += (
                    other.data * result.grad
                )

            if other.requires_grad:
                other.grad += (
                    self.data * result.grad
                )

        result._backward = backward

        return result

    def __rmul__(self, other):
        return self * other

    # =========================================
    # DIV
    # =========================================

    def __truediv__(self, other):

        other = self._to_tensor(other)

        result = Tensor(
            self.data / other.data,
            requires_grad=(
                self.requires_grad
                or other.requires_grad
            ),
            _children=(self, other),
            _op="/"
        )

        def backward():

            if self.requires_grad:
                self.grad += (
                    result.grad / other.data
                )

            if other.requires_grad:
                other.grad -= (
                    self.data
                    / (other.data ** 2)
                    * result.grad
                )

        result._backward = backward

        return result

    # =========================================
    # POWER
    # =========================================

    def __pow__(self, power):

        result = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{power}"
        )

        def backward():

            if self.requires_grad:

                self.grad += (
                    power
                    * self.data ** (power - 1)
                    * result.grad
                )

        result._backward = backward

        return result

    # =========================================
    # EXP
    # =========================================

    def exp(self):

        value = math.exp(self.data)

        result = Tensor(
            value,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp"
        )

        def backward():

            if self.requires_grad:
                self.grad += (
                    value * result.grad
                )

        result._backward = backward

        return result

    # =========================================
    # LOG
    # =========================================

    def log(self):

        result = Tensor(
            math.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log"
        )

        def backward():

            if self.requires_grad:
                self.grad += (
                    (1 / self.data)
                    * result.grad
                )

        result._backward = backward

        return result

    # =========================================
    # BACKWARD
    # =========================================

    def backward(self):

        if not self.requires_grad:
            return

        graph = []
        visited = set()

        def build(node):

            if node not in visited:

                visited.add(node)

                for child in node._prev:
                    build(child)

                graph.append(node)

        build(self)

        self.grad = 1.0

        for node in reversed(graph):
            node._backward()

    # =========================================
    # ZERO GRAD
    # =========================================

    def zero_grad(self):
        self.grad = 0.0

    # =========================================
    # HELPER
    # =========================================

    @staticmethod
    def _to_tensor(value):

        if isinstance(value, Tensor):
            return value

        return Tensor(value)
