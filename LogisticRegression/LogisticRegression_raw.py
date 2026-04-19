import numpy as np


# In this file, i add bias and X -> X_aug = [b | X]
class LogisticRegressionRaw:
    def __init__(self, lr, epochs):
        self.weight = None
        self.lr = lr
        self.epochs = epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _add_bias(self, X_raw):
        return np.column_stack((np.ones(X_raw.shape[0]), X_raw))

    def fit(self, X_raw, y):
        X_aug = self._add_bias(X_raw)  # X_aug (m x (1+n))
        m, n_b = X_aug.shape

        self.weight = np.zeros(n_b)

        for i in range(self.epochs):
            y_hat = self._sigmoid(X_aug @ self.weight)
            e = y_hat - y

            grad = (1 / m) * (X_aug.T @ e)
            self.weight -= self.lr * grad

    def predict_prob(self, X_raw):
        X_aug = self._add_bias(X_raw)

        y_hat = X_aug @ self.weight  # type: ignore
        y_hat_prob = self._sigmoid(y_hat)
        return y_hat_prob

    def predict(self, X_raw):
        prob = self.predict_prob(X_raw)
        return int(prob > 0.5)
