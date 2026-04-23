import numpy as np

# Nội dung thuật toán K-Means:
# - Chọn ngẫu nhiên k điểm làm tâm cụm:
#   + Chọn số lượng k theo cảm tính/elbow method/...
#   + Chọn tâm cụm bằng k điểm random hoặc lấy random k điểm dataset
#
# - Tính khoảng cách từ mỗi điểm trong m điểm dataset tới k cụm
# - Gán điểm data point đó vào nhóm tâm cụm gần nhất
# - Cập nhật tâm cụm: Sau khi gán m điểm đó vào tâm cụm, tâm cụm mới là trung bình cộng các điểm trong cụm
# - Lặp lại các bước: Tính khoảng cách - Gán cụm - Cập nhật cụm
# - Lặp cho đến 1 mức set trước hoặc khi nhận ra độ xê dịch của các tâm cụm ko còn quá đáng kể


class KMeansRaw:
    def __init__(self, n_clusters=8, max_iters=300, init="random", random_state=None):
        self.k = n_clusters
        self.max_iters = max_iters
        self.init = init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def _initialize_centers(self, X):
        m = X.shape[0]
        rng = np.random.default_rng(
            self.random_state
        )  # is like by a manchine for rolling random number

        random_idx = rng.choice(m, size=self.k, replace=False)

        return X[random_idx]

    def _compute_dist(self, X):
        X_expanded = X[:, None, :]
        clusters_expanded = self.cluster_center[None, :, :]

        diff = X_expanded - clusters_expanded
        dist = np.linalg.norm(diff, axis=2)

        return dist

    def _assign_clusters(self, X):
        dist = self._compute_dist(X)
        return np.argmin(dist, axis=1)

    def _update_centers(self, X, labels):
        new_centers = np.zeros_like(
            self.cluster_centers_
        )  # We dont update directly on self.cluster_center because we need to compare before and after to see if it's coverge

        for k in range(self.k):
            mask = (
                labels == k
            )  # This is a vector (m,) represent each datapoint belongs to which cluster
            if np.any(
                mask
            ):  # Check if there is any 'True' in mask (means the cluster is empty)
                new_centers[k] = np.mean(
                    X[mask], axis=0
                )  # axis=0 means calculate by vertical axis
            else:
                new_centers[k] = self.cluster_centers_[
                    k
                ]  # If the cluster is empty, dont change anything

        return new_centers

    def fit(self, X):
        cluster_center = self._initialize_centers(X)

        for i in range(self.max_iters):
            pass

    def predict(self, X):
        pass
