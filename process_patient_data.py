import pandas as pd
import os
import sys

# 定义基础的特征列及其顺序 (值列)
BASE_FEATURES = sorted([
    "Arterial_diastolic_pressure",
    "Arterial_systolic_pressure",
    "ETCO2",
    "Mean_arterial_pressure",
    "No_revenue_generating_compression",
    "Non_invasive_diastolic_blood_pressure",
    "PULSE",
    "ST-I",
    "SpO2",
    "breathe",
    "heart_rate"
])

def process_file(input_file_path, output_file_path):
    """
    读取Excel文件，处理特征值，创建特征存在掩码，并将其另存为CSV文件。
    """
    try:
        df = pd.read_excel(input_file_path, engine='openpyxl')
        
        # 清理并记录原始列名
        original_columns_in_file = {str(col).strip() for col in df.columns}
        df.columns = [str(col).strip() for col in df.columns] # Ensure df.columns are also stripped for immediate use

        if "TIMESTAMP" not in original_columns_in_file:
            print(f"警告: 文件 {input_file_path} 中未找到 TIMESTAMP 列。跳过此文件。", file=sys.stderr)
            return

        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        df = df.set_index("TIMESTAMP")

        # --- 1. 处理特征值 ---
        # 选择当前文件中实际存在的、且属于BASE_FEATURES的列
        current_value_features_present = [col for col in df.columns if col in BASE_FEATURES]
        df_values = df[current_value_features_present]
        
        # 重新索引以确保所有BASE_FEATURES列都存在 (为原始文件中没有的特征列引入NaN)
        df_values_reindexed = df_values.reindex(columns=BASE_FEATURES)
        
        # 对特征值进行插值和填充
        df_values_filled = df_values_reindexed.interpolate(method='time', axis=0).ffill().bfill()

        # 显式处理仍然为NaN的特征列
        # df_values_filled 应该包含所有 BASE_FEATURES 列。
        # 在这些特定列中将任何 NaN 替换为 0。
        # Convert to numeric first, coercing errors, then fillna. This is robust.
        for col in BASE_FEATURES:
            if col in df_values_filled.columns:
                df_values_filled[col] = pd.to_numeric(df_values_filled[col], errors='coerce')
            else:
                 # This case means reindex didn't add the column, or it was dropped.
                 # Add it back as all zeros. This is a safeguard.
                 df_values_filled[col] = 0
        df_values_filled[BASE_FEATURES] = df_values_filled[BASE_FEATURES].fillna(0)

        # --- 2. 创建特征掩码 ---
        mask_data_dict = {}
        for feature_name in BASE_FEATURES:
            mask_column_name = f"{feature_name}_mask"
            # 如果原始文件包含该特征列，掩码为1，否则为0
            mask_data_dict[mask_column_name] = 1 if feature_name in original_columns_in_file else 0
        
        # 创建掩码DataFrame，其索引与特征值DataFrame的索引一致
        df_masks = pd.DataFrame(mask_data_dict, index=df_values_filled.index)
        
        # --- 3. 合并特征值和掩码 ---
        df_combined = pd.concat([df_values_filled, df_masks], axis=1)
        
        # 确保输出列的顺序：所有值列（按BASE_FEATURES顺序），然后所有掩码列（按字母顺序）
        ordered_value_cols = [f for f in BASE_FEATURES if f in df_combined.columns]
        ordered_mask_cols = sorted([f for f in df_combined.columns if f.endswith("_mask") and f not in ordered_value_cols])
        final_ordered_cols = ordered_value_cols + ordered_mask_cols
        df_to_save = df_combined[final_ordered_cols]

        # DEBUG: Check for NaNs before saving CSV
        if input_file_path.endswith("0011.xlsx"): # Print for one specific file for brevity
            nan_check_values = df_to_save[ordered_value_cols].isnull().sum().sum()
            print(f"DEBUG process_patient_data: File {input_file_path}, NaNs in value columns before save: {nan_check_values}", file=sys.stderr)
            if nan_check_values > 0:
                # This print might be too verbose if many rows have NaNs, but useful for a single file check
                print(f"DEBUG process_patient_data: Example NaNs in {input_file_path}:\n{df_to_save[ordered_value_cols][df_to_save[ordered_value_cols].isnull().any(axis=1)].head()}", file=sys.stderr)

        # 将索引（时间戳）转换回列，并确保其名为TIMESTAMP
        df_to_save.reset_index(inplace=True)
        # 如果 reset_index 创建的列名不是 'TIMESTAMP'（例如，它是 'index' 或原始索引名），则重命名它
        # df_timestamps.name 应该是 'TIMESTAMP'，所以 reset_index 应该自动使用它
        # 但作为保险，我们可以明确检查并重命名
        if df_to_save.columns[0] != 'TIMESTAMP' and df_timestamps.name == 'TIMESTAMP':
             df_to_save.rename(columns={df_to_save.columns[0]: 'TIMESTAMP'}, inplace=True)
        elif 'level_0' in df_to_save.columns and df_timestamps.name is None: # common if original index was unnamed then reset
            df_to_save.rename(columns={'level_0': 'TIMESTAMP'}, inplace=True)
        elif df_to_save.columns[0] != 'TIMESTAMP' and df_timestamps.name is not None and df_timestamps.name != 'TIMESTAMP': # Index had a different name
            df_to_save.rename(columns={df_timestamps.name : 'TIMESTAMP'}, inplace=True)
        # Add an assertion or check
        if 'TIMESTAMP' not in df_to_save.columns:
            print(f"CRITICAL ERROR: TIMESTAMP column not found or correctly named in {input_file_path} before saving! Columns: {df_to_save.columns}", file=sys.stderr)
            # Potentially raise an error here or handle

        df_to_save.to_csv(output_file_path, index=False)
        # print(f"已处理 {input_file_path} 并添加掩码，保存至 {output_file_path}")

    except Exception as e:
        print(f"处理文件 {input_file_path} (带掩码) 时出错: {e}", file=sys.stderr)

def process_all_data(input_dir, output_dir):
    """
    处理输入目录中的所有Excel文件，并将结果保存到输出目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录 {input_dir} 未找到。", file=sys.stderr)
        return

    processed_count = 0
    file_list = [f for f in os.listdir(input_dir) if f.endswith(".xlsx")]
    total_files = len(file_list)

    for filename in file_list:
        input_file_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".csv"
        output_file_path = os.path.join(output_dir, output_filename)
        process_file(input_file_path, output_file_path)
        processed_count +=1
        # print(f"已处理 {processed_count}/{total_files} : {filename}")
    
    print(f"已处理 {input_dir} 中的所有 {processed_count} 个Excel文件。输出位于 {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_directory = sys.argv[1]
        output_directory = sys.argv[2]
        print(f"开始处理数据，输入目录: '{input_directory}', 输出目录: '{output_directory}'")
        print(f"将为每个基础特征生成值列和对应的存在掩码列 (_mask 后缀)。")
        process_all_data(input_directory, output_directory)
    else:
        print("用法: python process_patient_data.py <input_directory> <output_directory>", file=sys.stderr)
        print("示例: python process_patient_data.py clean_dataset processed_dataset_with_masks", file=sys.stderr)
        print("注意：建议使用新的输出目录 (例如 'processed_dataset_with_masks') 以避免覆盖之前的结果。", file=sys.stderr) 