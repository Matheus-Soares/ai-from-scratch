import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            for (x, target) in zip(X, y):
                pred = self._step_function(np.dot(x, self.weights) + self.bias)

                if pred != target:
                    error = target - pred
                    self.weights += self.lr * error * x
                    self.bias += self.lr * error

    def predict(self, X):
        result = np.dot(X, self.weights) + self.bias
        return self._step_function(result)

    def _step_function(self, x):
        return 1 if x >= 0 else 0
