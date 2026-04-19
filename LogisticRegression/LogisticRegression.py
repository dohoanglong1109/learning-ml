import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)


# --- BƯỚC 1: GIẢ LẬP DỮ LIỆU (Mô phỏng dữ liệu từ SQL) ---
def load_data():
    np.random.seed(42)
    n_samples = 1000
    data = {
        "income": np.random.normal(50, 15, n_samples),  # Thu nhập (triệu/tháng)
        "credit_score": np.random.normal(600, 100, n_samples),  # Điểm tín dụng
        "age": np.random.randint(20, 70, n_samples),
    }
    df = pd.DataFrame(data)

    # Tạo biến mục tiêu (Target) dựa trên logic thực tế + một chút nhiễu
    # Xác suất đậu thẻ cao nếu thu nhập cao và điểm tín dụng tốt
    z = 0.05 * df["income"] + 0.01 * df["credit_score"] - 8
    prob = 1 / (1 + np.exp(-z))
    df["is_approved"] = (prob > np.random.rand(n_samples)).astype(int)
    return df


# --- BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU (Preprocessing) ---
def preprocess_pipeline(df):
    X = df.drop("is_approved", axis=1)
    y = df["is_approved"]

    # Chia tập dữ liệu: Luôn dùng stratify cho bài toán phân loại để giữ tỉ lệ các lớp
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Senior Tip: Scale dữ liệu là BẮT BUỘC cho Logistic Regression
    scaler = StandardScaler()

    # Chỉ fit trên tập Train để tránh rò rỉ dữ liệu (Data Leakage)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


# --- BƯỚC 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ ---
def train_and_evaluate():
    # 1. Load và Prep
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline(df)

    # 2. Khởi tạo mô hình
    # C=1.0 là mặc định (Regularization Strength), solver 'lbfgs' ổn định cho tập dữ liệu nhỏ/vừa
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)

    # 3. Dự đoán
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 4. In báo cáo chi tiết
    print("--- BÁO CÁO KỸ THUẬT ---")
    print(classification_report(y_test, y_pred))
    print(f"Chỉ số ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # 5. Giải thích hệ số (Interpretability) - Phần quan trọng nhất của Senior
    print("\n--- Tầm quan trọng của các biến (Weights) ---")
    weights = pd.Series(model.coef_[0], index=feature_names)
    print(weights.sort_values(ascending=False))

    return model, X_test, y_test, y_proba


# Thực thi
if __name__ == "__main__":
    model, X_test, y_test, y_proba = train_and_evaluate()
