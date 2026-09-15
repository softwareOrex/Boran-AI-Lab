import random


class Neuron:
    def __init__(self, inputs_count):
        self.weights = [
            random.uniform(-1, 1)
            for _ in range(inputs_count)
        ]

        self.bias = random.uniform(-1, 1)

        
        self.last_inputs = []
        self.last_output = 0

    def forward(self, inputs):
        self.last_inputs = inputs

        total = self.bias

        for x, w in zip(inputs, self.weights):
            total += x * w

        self.last_output = total

        return total

    def backward(self, gradient, learning_rate):
        """
        gradient — насколько ошибка зависит
        от выхода этого нейрона.
        """

        
        input_gradients = []

        for i in range(len(self.weights)):
            input_gradient = gradient * self.weights[i]
            input_gradients.append(input_gradient)

    
        for i in range(len(self.weights)):
            weight_gradient = gradient * self.last_inputs[i]

            self.weights[i] -= learning_rate * weight_gradient

        
        self.bias -= learning_rate * gradient

        return input_gradients
