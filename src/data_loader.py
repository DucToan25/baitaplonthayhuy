import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV

    Args:
        file_path (str): Đường dẫn đến file CSV

    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu sinh viên
    """
    try:
        df = pd.read_csv(file_path)  # Sửa từ pd.read_excel() thành pd.read_csv()
        print(f"Đã đọc dữ liệu thành công từ {file_path}")
        print(f"Kích thước dữ liệu: {df.shape}")
        return df
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None


def explore_data(df):
    """
    Khám phá dữ liệu

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu

    Returns:
        dict: Thông tin cơ bản về dữ liệu
    """
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes,
        "missing_values": df.isnull().sum(),
        "summary": df.describe()
    }

    print("=== THÔNG TIN DỮ LIỆU ===")
    print(f"Số hàng, số cột: {info['shape']}")
    print("\nCác cột trong dữ liệu:")
    for col in info['columns']:
        print(f"- {col}")

    print("\nKiểu dữ liệu:")
    print(info['dtypes'])

    print("\nGiá trị thiếu:")
    print(info['missing_values'])

    print("\nThống kê mô tả:")
    print(info['summary'])

    return info


def preprocess_data(df, target_column):
    """
    Tiền xử lý dữ liệu

    Args:
        df (pd.DataFrame): DataFrame gốc
        target_column (str): Tên cột mục tiêu

    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    # Xác định các cột categorical và numerical
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Loại bỏ cột target khỏi features
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)

    # Tạo pipeline tiền xử lý
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Tách features và target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Chia tập train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Kích thước tập train: {X_train.shape}")
    print(f"Kích thước tập test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, preprocessor