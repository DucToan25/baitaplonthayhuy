from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import os
import pandas as pd


def train_model(X_train, y_train, preprocessor, model_type='random_forest'):
    """
    Huấn luyện mô hình

    Args:
        X_train: Dữ liệu đặc trưng tập huấn luyện
        y_train: Nhãn tập huấn luyện
        preprocessor: Bộ tiền xử lý dữ liệu
        model_type (str): Loại mô hình muốn huấn luyện

    Returns:
        Pipeline: Pipeline đã huấn luyện (bao gồm cả tiền xử lý và mô hình)
    """
    # Khởi tạo mô hình dựa trên loại được chọn
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1)
    elif model_type == 'svr':
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    else:  # default: random_forest
        model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)

    # Tạo pipeline kết hợp tiền xử lý và mô hình
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)

    print(f"Đã huấn luyện mô hình {model_type} thành công")

    return pipeline


def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình

    Args:
        model: Mô hình đã huấn luyện
        X_test: Dữ liệu đặc trưng tập kiểm tra
        y_test: Nhãn tập kiểm tra

    Returns:
        dict: Các chỉ số đánh giá mô hình
    """
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Tính các chỉ số đánh giá
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # In kết quả
    print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred,
        'y_test': y_test
    }


def save_model(model, model_type, output_dir):
    """
    Lưu mô hình đã huấn luyện

    Args:
        model: Mô hình đã huấn luyện
        model_type (str): Loại mô hình
        output_dir (str): Thư mục đầu ra

    Returns:
        str: Đường dẫn đến file mô hình
    """
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Đường dẫn lưu mô hình
    model_path = os.path.join(output_dir, f"{model_type}_model.joblib")

    # Lưu mô hình
    joblib.dump(model, model_path)
    print(f"Đã lưu mô hình tại {model_path}")

    return model_path


def load_model(model_path):
    """
    Tải mô hình đã lưu

    Args:
        model_path (str): Đường dẫn đến file mô hình

    Returns:
        Model: Mô hình đã tải
    """
    try:
        model = joblib.load(model_path)
        print(f"Đã tải mô hình từ {model_path}")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None


def compare_models(X_train, y_train, X_test, y_test, preprocessor):
    """
    So sánh hiệu suất của các mô hình khác nhau

    Args:
        X_train, y_train: Dữ liệu huấn luyện
        X_test, y_test: Dữ liệu kiểm tra
        preprocessor: Bộ tiền xử lý dữ liệu

    Returns:
        pd.DataFrame: Bảng so sánh hiệu suất các mô hình
    """
    # Danh sách các mô hình cần so sánh
    models = {
        'Linear Regression': 'linear',
        'Ridge Regression': 'ridge',
        'Lasso Regression': 'lasso',
        'SVR': 'svr',
        'Random Forest': 'random_forest',
        'Gradient Boosting': 'gradient_boosting'
    }

    # Lưu kết quả đánh giá
    results = []

    # Huấn luyện và đánh giá từng mô hình
    for model_name, model_type in models.items():
        print(f"\n--- Đang đánh giá mô hình: {model_name} ---")

        # Huấn luyện mô hình
        model = train_model(X_train, y_train, preprocessor, model_type)

        # Đánh giá mô hình
        metrics = evaluate_model(model, X_test, y_test)

        # Lưu kết quả
        results.append({
            'Model': model_name,
            'MSE': metrics['mse'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R2': metrics['r2']
        })

    # Tạo DataFrame kết quả
    results_df = pd.DataFrame(results)

    return results_df