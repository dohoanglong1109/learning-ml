import numpy as np


class KNNRaw:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.Y_train = None
        self.n_classes = None

    def fit(self, X, y):
        self.X_train = X  # m x n
        self.Y_train = y.ravel()  # m x 1
        self.n_classes = len(np.unique(self.Y_train))

    def countVote(self, y_filtered):  # Expected: (a x K) -> (a x n_classes)
        a = np.shape(y_filtered)[0]
        votes = np.zeros((a, self.n_classes), dtype=int)

        for i in range(a):
            votes[i] = np.bincount(y_filtered[i], minlength=self.n_classes)

        return votes

    def predict(self, X_new):
        X_new_expanded = X_new[:, None, :]  #
        X_train_expanded = self.X_train[None, :, :]  # type: ignore

        # X_new_expanded (a x 1 x n)
        # X_train_expanded (1 x m x n)
        # dist (a x m) (raw is axmxn but axis=2 make it a x m)
        dist = np.linalg.norm(X_new_expanded - X_train_expanded, axis=2)

        top_k_idx = np.argpartition(dist, kth=self.k, axis=1)[:, : self.k]  # (a x K)
        # Although y is (m,1) and top_k_idx is (a,K) but numpy will apply boardcasting index, make y_filtered be (a,K)
        y_filtered = self.Y_train[top_k_idx]  # type: ignore
