from dalearn import Tensor
from dalearn import Linear
from dalearn import relu
from dalearn import SGD


# ====================================
# DATA
# ====================================

X = [
    [1],
    [2],
    [3],
    [4],
    [5]
]

Y = [
    2,
    4,
    6,
    8,
    10
]


# ====================================
# MODEL
# ====================================

layer1 = Linear(1, 4)
layer2 = Linear(4, 1)


parameters = (
    layer1.parameters()
    + layer2.parameters()
)


optimizer = SGD(
    parameters,
    learning_rate=0.001
)


# ====================================
# TRAINING
# ====================================

for epoch in range(5000):

    total_loss = 0

    for x, target in zip(X, Y):

        # Tensor
        inputs = [
            Tensor(
                value,
                requires_grad=False
            )
            for value in x
        ]

        target = Tensor(target)

        # Forward

        hidden = layer1.forward(inputs)

        hidden = [
            relu(value)
            for value in hidden
        ]

        prediction = layer2.forward(hidden)[0]

        # Loss

        loss = (
            prediction - target
        ) ** 2

        total_loss += loss.data

        # Backward

        optimizer.zero_grad()

        loss.backward()

        # Update

        optimizer.step()

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch} | "
            f"Loss: {total_loss:.6f}"
        )


# ====================================
# TEST
# ====================================

print("\nTEST")

for x in X:

    inputs = [
        Tensor(value)
        for value in x
    ]

    hidden = layer1.forward(inputs)

    hidden = [
        relu(value)
        for value in hidden
    ]

    prediction = layer2.forward(hidden)[0]

    print(
        f"{x[0]} -> "
        f"{prediction.data:.4f}"
    )


# New value

inputs = [Tensor(10)]

hidden = layer1.forward(inputs)

hidden = [
    relu(value)
    for value in hidden
]

prediction = layer2.forward(hidden)[0]

print(
    f"\n10 -> {prediction.data:.4f}"
)
