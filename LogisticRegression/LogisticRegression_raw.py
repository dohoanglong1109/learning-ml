import numpy as np


# In this file, i add bias and X -> X_aug = [b | X]
class LogisticRegressionRaw:
    def __init__(self, lr, epochs):
        self.weight = None
        self.lr = lr
        self.epochs = epochs
        self.history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _add_bias(self, X_raw):
        return np.column_stack((np.ones(X_raw.shape[0]), X_raw))

    def fit(self, X_raw, y):
        epsilon = 1e-15
        y = y.ravel()
        X_aug = self._add_bias(X_raw)  # X_aug (m x (1+n))
        m, n_b = X_aug.shape
        self.weight = np.zeros(n_b)

        for i in range(self.epochs):
            y_hat = self._sigmoid(X_aug @ self.weight)
            y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon)

            term1 = y.T @ np.log(y_hat_clipped)
            term2 = (1 - y).T @ np.log(1 - y_hat_clipped)
            log_loss = (-1 / m) * (term1 + term2)
            self.history.append(log_loss)

            grad = (1 / m) * (X_aug.T @ (y_hat - y))
            self.weight -= self.lr * grad

    def predict_prob(self, X_raw):
        X_aug = self._add_bias(X_raw)

        y_hat = X_aug @ self.weight  # type: ignore
        y_hat_prob = self._sigmoid(y_hat)
        return y_hat_prob

    def predict(self, X_raw):
        prob = self.predict_prob(X_raw)
        return int(prob > 0.5)
