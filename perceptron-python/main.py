import numpy as np

from perceptron import Perceptron


def perceptron_or():
    # OR example
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])

    print("Training Perceptron to learn OR function!")
    train_and_evaluate(X, y)


def perceptron_and():
    # AND example
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])

    print("Training Perceptron to learn AND function!")
    train_and_evaluate(X, y)


def train_and_evaluate(X, y):
    model = Perceptron(learning_rate=0.1, epochs=20)
    model.fit(X, y)

    print("Trained weights:", model.weights)
    print("Trained bias:", model.bias)

    for (x, target) in zip(X, y):
        pred = model.predict(x)
        print(f"Input = {x}\t Ground-truth = {target[0]}\t Predicted = {pred}")


if __name__ == "__main__":
    perceptron_or()
    print("\n")
    perceptron_and()
