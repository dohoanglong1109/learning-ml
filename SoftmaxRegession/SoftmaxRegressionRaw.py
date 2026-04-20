import numpy as np

# Thực tế, nó ko khác gì Logistic Regression, chỉ là giờ W và Y có K cột thay vì 1 như trước


class SoftmaxRegressionRaw:
    def __init__(self, lr, epochs, k_classes):
        self.lr = lr
        self.epochs = epochs
        self.k_classes = k_classes
        self.weights = None
        self.history = []

    def _softmax(self, Z):
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_stable)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def _add_intercept(self, X_raw):
        return np.column_stack((np.ones(X_raw.shape[0]), X_raw))

    def _oneHot(self, y, K):
        m = len(y)
        y_ohe = np.zeros((m, K))
        y_ohe[np.arange(m), y] = 1
        return y_ohe

    def fit(self, X_raw, y):
        X_aug = self._add_intercept(X_raw)
        m, n_b = X_aug.shape  # (m x n_b)
        self.weights = np.zeros((n_b, self.k_classes))  # (n_b x k)
        y_ohe = self._oneHot(y, self.k_classes)  # (m x k)

        for i in range(self.epochs):
            z = X_aug @ self.weights
            y_hat = self._softmax(z)
            e = y_hat - y_ohe
            grad = (1 / m) * (X_aug.T @ e)
            self.weights -= self.lr * grad

            loss = -np.mean(np.sum(y_ohe * np.log(y_hat + 1e-15), axis=1))
            self.history.append(loss)

    def predict_prob(self, X_raw):
        X_aug = self._add_intercept(X_raw)
        z = X_aug @ self.weights  # type: ignore
        return self._softmax(z)

    def predict(self, X_raw):
        prob = self.predict_prob(X_raw)
        return np.argmax(prob, axis=1)
