import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_score_distribution(df, target_column, output_dir):
    """
    Vẽ biểu đồ phân phối điểm số

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        target_column (str): Tên cột mục tiêu
        output_dir (str): Thư mục để lưu biểu đồ
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_column], kde=True)
    plt.title(f'Phân phối điểm số của {target_column}')
    plt.xlabel(target_column)
    plt.ylabel('Số lượng')

    output_path = os.path.join(output_dir, f'{target_column.replace("/", "_")}_distribution.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Đã lưu biểu đồ phân phối điểm tại {output_path}")

def plot_correlation_heatmap(df, output_dir):
    """
    Vẽ heatmap tương quan giữa các cột số

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        output_dir (str): Thư mục để lưu biểu đồ
    """
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap Tương Quan')

    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Đã lưu biểu đồ tương quan tại {output_path}")

def plot_categorical_vs_target(df, categorical_columns, target_column, output_dir):
    """
    Vẽ biểu đồ boxplot thể hiện mối quan hệ giữa biến categorical và biến mục tiêu

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        categorical_columns (list): Danh sách các cột categorical
        target_column (str): Tên cột mục tiêu
        output_dir (str): Thư mục để lưu biểu đồ
    """
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y=target_column, data=df)
        plt.title(f'{col.replace("/", "_")} vs {target_column}')
        plt.xlabel(col)
        plt.ylabel(target_column)
        plt.xticks(rotation=45)

        # Tạo đường dẫn file và thay ký tự / bằng _
        output_path = os.path.join(output_dir, f'{col.replace("/", "_")}_vs_{target_column}.png')
        os.makedirs(output_dir, exist_ok=True)  # Đảm bảo thư mục tồn tại
        plt.savefig(output_path)
        plt.close()

        print(f"Đã lưu biểu đồ {col.replace('/', '_')} vs {target_column} tại {output_path}")

def plot_prediction_vs_actual(y_test, y_pred, output_dir):
    """
    Vẽ biểu đồ so sánh giá trị dự đoán và thực tế

    Args:
        y_test (array): Giá trị thực tế
        y_pred (array): Giá trị dự đoán
        output_dir (str): Thư mục để lưu biểu đồ
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('So sánh Giá trị Thực tế và Dự đoán')

    output_path = os.path.join(output_dir, 'prediction_vs_actual.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Đã lưu biểu đồ so sánh tại {output_path}")

def plot_residuals(y_test, y_pred, output_dir):
    """
    Vẽ biểu đồ phần dư

    Args:
        y_test (array): Giá trị thực tế
        y_pred (array): Giá trị dự đoán
        output_dir (str): Thư mục để lưu biểu đồ
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Giá trị dự đoán')
    plt.ylabel('Phần dư')
    plt.title('Biểu đồ Phần dư')

    output_path = os.path.join(output_dir, 'residuals_plot.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Đã lưu biểu đồ phần dư tại {output_path}")

def plot_feature_importance(model, feature_names, output_dir):
    """
    Vẽ biểu đồ tầm quan trọng của các đặc trưng

    Args:
        model: Mô hình đã huấn luyện
        feature_names (list): Danh sách tên đặc trưng
        output_dir (str): Thư mục để lưu biểu đồ
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Tên đặc trưng')
        plt.ylabel('Tầm quan trọng')
        plt.title('Tầm quan trọng của các đặc trưng')

        output_path = os.path.join(output_dir, 'feature_importance.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"Đã lưu biểu đồ tầm quan trọng tại {output_path}")