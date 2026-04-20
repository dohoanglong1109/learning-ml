import numpy as np


class KNNRaw:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.Y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y

    def predict(self, X_new):
        X_new_expanded = X_new[:, None, :]
        X_train_expanded = self.X_train[None, :, :]  # type: ignore

        # X_new_expanded (ax1xn)
        # X_train_expanded (1xmxn)
        # dist (axm) (raw is axmxn but axis=2 make it axm)
        dist = np.linalg.norm(X_new_expanded - X_train_expanded, axis=2)
