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
    def __init__(self, n_clusters, max_iters):
        self.k = n_clusters
        self.random_state = 42
        self.cluster_center = None

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
        pass

    def _update_centers(self, X, labels):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        pass
