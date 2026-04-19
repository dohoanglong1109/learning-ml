import numpy as np


# Theo thuần Linear Algebra: gộp cột bias vào X thành X = [b | xn]
class LinearRegressionFromScratch:
    def __init__(self, learning_rate, epochs):
        self.weights = None
        # self.bias = None
        self.lr = learning_rate
        self.epochs = epochs
        self.lost_history = []

    def _add_intercept(self, X_raw):
        return np.column_stack((np.ones(X_raw.shape[0]), X_raw))

    def fit(self, X_raw, y):  # X_raw (m,n)
        X_aug = self._add_intercept(X_raw)  # X_aug (m, n+1) = (m, n_b)
        m, n_b = X_aug.shape

        self.weights = np.zeros(n_b)  # (n_b, 1)

        for i in range(self.epochs):
            y_hat = X_aug @ self.weights  # y_predicted
            e = y_hat - y
            grad = (1 / m) * (X_aug.T @ e)
            self.weights -= self.lr * grad

    def predict(self, X_raw):
        X_aug = self._add_intercept(X_raw)

        y_predicted = X_aug @ self.weights  # type: ignore
        return y_predicted
