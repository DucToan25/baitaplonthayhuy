import pandas as pd
import numpy as np


def detect_outliers(df, column, method='iqr'):
    """
    Phát hiện các outliers trong một cột

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        column (str): Tên cột cần kiểm tra
        method (str): Phương pháp phát hiện ('iqr' hoặc 'zscore')

    Returns:
        pd.Series: Chỉ mục của các outliers
    """
    if method == 'iqr':
        # Phương pháp IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index

    elif method == 'zscore':
        # Phương pháp Z-score
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outliers = df[abs(z_scores) > 3].index

    else:
        raise ValueError("Phương pháp không hợp lệ. Sử dụng 'iqr' hoặc 'zscore'")

    return outliers


def convert_categorical_to_numeric(df, categorical_columns):
    """
    Chuyển đổi biến categorical sang numeric

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        categorical_columns (list): Danh sách các cột categorical

    Returns:
        pd.DataFrame: DataFrame với các cột đã được chuyển đổi
    """
    df_encoded = df.copy()

    for col in categorical_columns:
        # Sử dụng pandas get_dummies để one-hot encoding
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        df_encoded.drop(col, axis=1, inplace=True)

    return df_encoded


def handle_missing_values(df, strategy='mean'):
    """
    Xử lý giá trị thiếu

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        strategy (str): Chiến lược xử lý ('mean', 'median', 'mode', 'drop')

    Returns:
        pd.DataFrame: DataFrame đã xử lý giá trị thiếu
    """
    df_clean = df.copy()

    # Đếm số lượng giá trị thiếu
    missing_count = df.isnull().sum()
    columns_with_missing = missing_count[missing_count > 0].index.tolist()

    if not columns_with_missing:
        print("Không có giá trị thiếu trong dữ liệu")
        return df_clean

    print(f"Các cột có giá trị thiếu: {columns_with_missing}")

    # Xử lý theo chiến lược được chọn
    if strategy == 'drop':
        # Xóa các hàng có giá trị thiếu
        df_clean = df_clean.dropna()
        print(f"Đã xóa {len(df) - len(df_clean)} hàng có giá trị thiếu")

    else:
        for col in columns_with_missing:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Với cột số
                if strategy == 'mean':
                    fill_value = df[col].mean()
                    print(f"Điền giá trị thiếu trong cột {col} bằng giá trị trung bình: {fill_value:.2f}")
                elif strategy == 'median':
                    fill_value = df[col].median()
                    print(f"Điền giá trị thiếu trong cột {col} bằng giá trị trung vị: {fill_value:.2f}")
                else:  # mode
                    fill_value = df[col].mode()[0]
                    print(f"Điền giá trị thiếu trong cột {col} bằng giá trị phổ biến nhất: {fill_value:.2f}")
            else:
                # Với cột categorical
                fill_value = df[col].mode()[0]
                print(f"Điền giá trị thiếu trong cột {col} bằng giá trị phổ biến nhất: {fill_value}")

            df_clean[col] = df_clean[col].fillna(fill_value)

    return df_clean


def print_classification_report_vn(y_true, y_pred, labels=None):
    """
    In báo cáo phân loại với tiếng Việt

    Args:
        y_true: Nhãn thực tế
        y_pred: Nhãn dự đoán
        labels: Danh sách các nhãn
    """
    from sklearn.metrics import classification_report, confusion_matrix

    # Tạo báo cáo phân loại
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)

    # In báo cáo với tiếng Việt
    print("=== BÁO CÁO PHÂN LOẠI ===")
    print("             | Độ chính xác | Độ phủ | F1-score | Số lượng")
    print("-------------|--------------|--------|----------|--------")

    # In thông tin cho từng lớp
    for label, metrics in report.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue

        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1-score']
        support = metrics['support']

        print(f"{label:12} | {precision:12.2f} | {recall:6.2f} | {f1:8.2f} | {support:6}")

    # In thông tin tổng hợp
    print("\n=== THÔNG TIN TỔNG HỢP ===")
    print(f"Độ chính xác tổng thể: {report['accuracy']:.4f}")

    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report:
            print(f"\n{avg_type}:")
            print(f"Độ chính xác: {report[avg_type]['precision']:.4f}")
            print(f"Độ phủ: {report[avg_type]['recall']:.4f}")
            print(f"F1-score: {report[avg_type]['f1-score']:.4f}")

    # In ma trận nhầm lẫn
    print("\n=== MA TRẬN NHẦM LẪN ===")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)