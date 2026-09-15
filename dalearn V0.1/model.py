from .losses import mse, mse_derivative


class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def predict(self, inputs):
        return self.forward(inputs)

    def train(
        self,
        X,
        Y,
        epochs=1000,
        learning_rate=0.01
    ):
        for epoch in range(epochs):

            total_loss = 0

            for inputs, targets in zip(X, Y):

                # =====================
                # FORWARD
                # =====================

                predictions = self.forward(inputs)

                # =====================
                # LOSS
                # =====================

                loss = mse(
                    predictions,
                    targets
                )

                total_loss += loss

                # =====================
                # BACKWARD
                # =====================

                gradients = mse_derivative(
                    predictions,
                    targets
                )

                for layer in reversed(self.layers):

                    gradients = layer.backward(
                        gradients,
                        learning_rate
                    )

            # Показываем прогресс
            if epoch % 100 == 0:
                average_loss = total_loss / len(X)

                print(
                    f"Epoch {epoch} | "
                    f"Loss: {average_loss:.6f}"
                )
