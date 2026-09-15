def mse(predictions, targets):
    total = 0

    for prediction, target in zip(predictions, targets):
        total += (prediction - target) ** 2

    return total / len(targets)


def mse_derivative(predictions, targets):
    gradients = []

    for prediction, target in zip(predictions, targets):
        gradient = 2 * (prediction - target)

        gradients.append(gradient)

    return gradients
