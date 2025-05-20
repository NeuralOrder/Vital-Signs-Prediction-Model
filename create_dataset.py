import os
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import List # Required for type hinting

# --- Constants ---
SEQ_LEN = 6
LABEL_LEN = 3  # Typically SEQ_LEN // 2 for Informer
PRED_LEN = 1
TARGET_FEATURES = ['heart_rate', 'PULSE', 'SpO2']

# ALL_BASE_FEATURES are the 11 value features from process_patient_data.py
ALL_BASE_FEATURES = sorted([
    "Arterial_diastolic_pressure", "Arterial_systolic_pressure", "ETCO2",
    "Mean_arterial_pressure", "No_revenue_generating_compression",
    "Non_invasive_diastolic_blood_pressure", "PULSE", "ST-I", "SpO2",
    "breathe", "heart_rate"
])
ALL_MASK_FEATURES = [f"{f}_mask" for f in ALL_BASE_FEATURES]
ENCODER_INPUT_FEATURES = ALL_BASE_FEATURES + ALL_MASK_FEATURES # 22 features for seq_x
DECODER_TARGET_FEATURES = TARGET_FEATURES # 3 features for seq_y (targets)

INPUT_DATA_DIR = 'processed_dataset_with_masks'
OUTPUT_DIR = 'informer_training_data'
OUTPUT_FILENAME = 'patient_data_informer.npz'

# --- Time Feature Extraction (copied and adapted from utils.timefeatures.py) ---
class TimeFeature:
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass
    def __repr__(self):
        return self.__class__.__name__ + "()"

class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

# Simplified: We know we need 't' frequency features.
# The original time_features_from_frequency_str is more general.
def get_minutely_time_features_classes() -> List[TimeFeature]:
    return [MinuteOfHour(), HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]

def extract_time_marks(datetime_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Extracts time features for the Informer model.
    For 't' frequency (minutely), as per Informer's time_features.py (timeenc=1).
    """
    feature_calculators = get_minutely_time_features_classes()
    time_marks = np.vstack([feat(datetime_index) for feat in feature_calculators]).transpose(1, 0)
    return time_marks

# --- Main Data Processing Logic ---
def create_dataset():
    all_seq_x = []
    all_seq_y = []
    all_seq_x_mark = []
    all_seq_y_mark = []

    if not os.path.exists(INPUT_DATA_DIR):
        print(f"输入目录 {INPUT_DATA_DIR} 不存在。请先运行 process_patient_data.py。")
        return

    patient_files = [f for f in os.listdir(INPUT_DATA_DIR) if f.endswith('.csv')]
    if not patient_files:
        print(f"目录 {INPUT_DATA_DIR} 中没有找到CSV文件。")
        return
        
    print(f"找到 {len(patient_files)} 个病人数据文件进行处理。")

    for i, filename in enumerate(patient_files):
        print(f"处理文件 ({i+1}/{len(patient_files)}): {filename}")
        file_path = os.path.join(INPUT_DATA_DIR, filename)
        try:
            df = pd.read_csv(file_path)
            if 'TIMESTAMP' not in df.columns:
                print(f"警告: 文件 {filename} 缺少 'TIMESTAMP' 列，已跳过。")
                continue
            
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
            df = df.set_index('TIMESTAMP')
            df = df.sort_index() # Ensure data is sorted by time

            if len(df) < SEQ_LEN + PRED_LEN:
                # print(f"文件 {filename} 数据点不足 ({len(df)})，无法创建任何序列，已跳过。")
                continue

            # Extract time features for the entire patient's timeline
            time_marks_all = extract_time_marks(df.index)

            # Prepare feature columns
            data_x_values = df[ENCODER_INPUT_FEATURES].values
            data_y_targets = df[DECODER_TARGET_FEATURES].values
            
            num_possible_sequences = len(df) - SEQ_LEN - PRED_LEN + 1
            for k in range(num_possible_sequences):
                s_begin = k
                s_end = s_begin + SEQ_LEN
                
                # Decoder input part 1 (label) and part 2 (prediction target) combined directly from historical data
                # As per Informer's Dataset_ETT_hour, seq_y contains actual values for label_len + pred_len
                r_begin = s_end - LABEL_LEN 
                r_end = r_begin + LABEL_LEN + PRED_LEN

                seq_x = data_x_values[s_begin:s_end]
                seq_y = data_y_targets[r_begin:r_end] # Actual values for decoder input (token) and target
                
                seq_x_mark = time_marks_all[s_begin:s_end]
                seq_y_mark = time_marks_all[r_begin:r_end]
                
                all_seq_x.append(seq_x)
                all_seq_y.append(seq_y)
                all_seq_x_mark.append(seq_x_mark)
                all_seq_y_mark.append(seq_y_mark)
        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")


    if not all_seq_x:
        print("未能从任何文件生成序列数据。")
        return

    # Convert lists to numpy arrays
    data_x_np = np.array(all_seq_x)
    data_y_np = np.array(all_seq_y)
    data_x_mark_np = np.array(all_seq_x_mark)
    data_y_mark_np = np.array(all_seq_y_mark)

    print(f"生成的序列形状:")
    print(f"  data_x: {data_x_np.shape}")
    print(f"  data_y: {data_y_np.shape}")
    print(f"  data_x_mark: {data_x_mark_np.shape}")
    print(f"  data_y_mark: {data_y_mark_np.shape}")

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    np.savez(output_path, 
             data_x=data_x_np, 
             data_y=data_y_np, 
             data_x_mark=data_x_mark_np, 
             data_y_mark=data_y_mark_np)
    print(f"已将处理好的数据保存至: {output_path}")

if __name__ == '__main__':
    create_dataset() 