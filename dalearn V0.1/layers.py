import random

from .tensor import Tensor


class Linear:

    def __init__(self, inputs, outputs):

        self.weights = [
            [
                Tensor(
                    random.uniform(-1, 1),
                    requires_grad=True
                )
                for _ in range(inputs)
            ]
            for _ in range(outputs)
        ]

        self.biases = [
            Tensor(
                0,
                requires_grad=True
            )
            for _ in range(outputs)
        ]

    def forward(self, inputs):

        result = []

        for weights, bias in zip(
            self.weights,
            self.biases
        ):

            total = bias

            for x, w in zip(inputs, weights):
                total = total + x * w

            result.append(total)

        return result

    def parameters(self):

        params = []

        for row in self.weights:
            params.extend(row)

        params.extend(self.biases)

        return params
