import os
import pandas as pd
import numpy as np
from src.data_loader import load_data, explore_data, preprocess_data
from src.data_visualization import (plot_score_distribution, plot_correlation_heatmap,
                                    plot_feature_importance, plot_prediction_vs_actual,
                                    plot_residuals, plot_categorical_vs_target)
from src.model import train_model, evaluate_model, save_model, compare_models
from src.utils import handle_missing_values
import argparse

def main():
    # Cấu hình đường dẫn thư mục
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    output_dir = os.path.join(current_dir, 'output')
    figures_dir = os.path.join(output_dir, 'figures')
    models_dir = os.path.join(output_dir, 'models')

    # Tạo thư mục nếu chưa tồn tại, với exist_ok=True để tránh lỗi nếu thư mục đã tồn tại
    for dir_path in [data_dir, output_dir, figures_dir, models_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Đọc dữ liệu
    data_path = os.path.join(data_dir, 'StudentsPerformance.csv')
    df = load_data(data_path)

    if df is None:
        print("Không thể đọc dữ liệu. Kết thúc chương trình.")
        return

    # Khám phá dữ liệu
    explore_data(df)

    # Xử lý giá trị thiếu nếu có
    df = handle_missing_values(df, strategy='mean')

    # Xác định cột mục tiêu (điểm thi)
    target_column = None

    # Tìm cột điểm thi
    score_columns = [col for col in df.columns if 'score' in col.lower() or 'mark' in col.lower()
                     or 'grade' in col.lower() or 'điểm' in col.lower()]

    if score_columns:
        print("\nCác cột điểm thi tìm thấy:")
        for i, col in enumerate(score_columns):
            print(f"{i + 1}. {col}")

        choice = int(input("\nChọn cột điểm thi để dự báo (nhập số): ")) - 1
        target_column = score_columns[choice]
    else:
        # Nếu không tìm thấy cột điểm thi, yêu cầu người dùng nhập tên cột
        print("\nKhông tìm thấy cột điểm thi. Vui lòng nhập tên cột điểm thi:")
        column_list = list(df.columns)
        for i, col in enumerate(column_list):
            print(f"{i + 1}. {col}")

        choice = int(input("\nChọn cột điểm thi để dự báo (nhập số): ")) - 1
        target_column = column_list[choice]

    print(f"\nĐã chọn cột mục tiêu: {target_column}")

    # Trực quan hóa phân phối điểm
    plot_score_distribution(df, target_column, figures_dir)

    # Vẽ heatmap tương quan
    plot_correlation_heatmap(df, figures_dir)

    # Xác định các cột categorical
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Vẽ biểu đồ thể hiện mối quan hệ giữa biến categorical và biến mục tiêu
    if categorical_columns:
        plot_categorical_vs_target(df, categorical_columns, target_column, figures_dir)

    # Tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target_column)

    # So sánh các mô hình
    print("\n=== SO SÁNH CÁC MÔ HÌNH ===")
    results_df = compare_models(X_train, y_train, X_test, y_test, preprocessor)
    print("\nKết quả so sánh các mô hình:")
    print(results_df)

    # Lưu kết quả so sánh
    results_path = os.path.join(output_dir, 'model_comparison.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Đã lưu kết quả so sánh mô hình tại {results_path}")

    # Chọn mô hình tốt nhất dựa trên R2
    best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
    best_model_type = best_model_name.lower().replace(' ', '_')
    print(f"\nMô hình tốt nhất: {best_model_name}")

    # Huấn luyện mô hình tốt nhất
    print(f"\n=== HUẤN LUYỆN MÔ HÌNH TỐT NHẤT: {best_model_name} ===")
    best_model = train_model(X_train, y_train, preprocessor, best_model_type)

    # Đánh giá mô hình
    print("\n=== ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT ===")
    evaluation = evaluate_model(best_model, X_test, y_test)

    # Lưu mô hình
    model_path = save_model(best_model, best_model_type, models_dir)

    # Trực quan hóa kết quả
    # Vẽ biểu đồ so sánh giá trị dự đoán và thực tế
    plot_prediction_vs_actual(evaluation['y_test'], evaluation['y_pred'], figures_dir)

    # Vẽ biểu đồ phần dư
    plot_residuals(evaluation['y_test'], evaluation['y_pred'], figures_dir)

    # Vẽ biểu đồ tầm quan trọng của các đặc trưng (nếu mô hình hỗ trợ)
    try:
        # Lấy tên đặc trưng sau khi tiền xử lý
        if hasattr(best_model.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        else:
            # Fallback nếu không có phương thức get_feature_names_out
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            feature_names = numeric_cols + [f"{col}_{val}" for col in categorical_cols
                                            for val in df[col].unique()]

        plot_feature_importance(best_model.named_steps['model'], feature_names, figures_dir)
    except Exception as e:
        print(f"Không thể vẽ biểu đồ tầm quan trọng đặc trưng: {e}")

    print("\n=== HOÀN THÀNH ===")
    print(f"Tất cả biểu đồ đã được lưu trong thư mục: {figures_dir}")
    print(f"Mô hình đã được lưu tại: {model_path}")
    print(f"Kết quả so sánh các mô hình đã được lưu tại: {results_path}")

if __name__ == "__main__":
    main()