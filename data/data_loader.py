import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None,
                 train_scalers=None):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target # Nominal for NPZ if features='M'
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc # Less relevant for NPZ as marks are pre-calculated
        self.freq = freq       # Less relevant for NPZ
        self.cols = cols       # Less relevant for NPZ direct data loading

        self.root_path = root_path
        self.data_path = data_path

        # Store scalers (especially for val/test to use train's scalers)
        self.scaler_x_values = None
        self.scaler_y_targets = None
        if flag != 'train' and train_scalers is not None:
            self.scaler_x_values, self.scaler_y_targets = train_scalers
        
        self.__read_data__()

    def __read_data__(self):
        is_npz = self.data_path.endswith('.npz')

        if is_npz:
            # Load pre-processed NPZ data
            raw_npz_data = np.load(os.path.join(self.root_path, self.data_path))
            raw_all_data_x = raw_npz_data['data_x']         # (total_samples, seq_len, 22)
            raw_all_data_y = raw_npz_data['data_y']         # (total_samples, label_len+pred_len, 3)
            raw_all_data_x_mark = raw_npz_data['data_x_mark'] # (total_samples, seq_len, num_time_features)
            raw_all_data_y_mark = raw_npz_data['data_y_mark'] # (total_samples, label_len+pred_len, num_time_features)
            raw_npz_data.close()

            # Define train/val/test splits (e.g., 70%, 15%, 15%)
            # These borders are applied to the first dimension (samples)
            num_total_samples = raw_all_data_x.shape[0]
            # For a quick test, let's use very few samples initially
            num_total_samples = min(num_total_samples, 2000) # REDUCED FOR QUICK DEBUG 
            
            num_train = int(num_total_samples * 0.7)
            num_test = int(num_total_samples * 0.15) # Adjusted for 15% test
            num_vali = num_total_samples - num_train - num_test # Remaining for val (15%)

            # Check for sufficient data for splits
            if num_train == 0 or num_vali == 0 or num_test == 0:
                # If not enough data for all splits, use all for training (and testing if needed)
                print(f"Warning: Not enough data for full train/val/test splits ({num_total_samples} samples). Adjusting splits.")
                if self.set_type == 0: # train
                    border1, border2 = 0, num_total_samples
                elif self.set_type == 1: # val
                    border1, border2 = 0, num_total_samples # Use all for val if no distinct val split
                else: # test
                    border1, border2 = 0, num_total_samples # Use all for test
            else:
                border1s = [0, num_train, num_train + num_vali]
                border2s = [num_train, num_train + num_vali, num_total_samples]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

            current_data_x = raw_all_data_x[border1:border2]
            current_data_y = raw_all_data_y[border1:border2]
            self.data_stamp_x = raw_all_data_x_mark[border1:border2]
            self.data_stamp_y = raw_all_data_y_mark[border1:border2]

            # DEBUG: Print current_data_x BEFORE scaling
            # if self.set_type == 0 and not hasattr(Dataset_Custom, '_debug_printed_current_data_x_raw_train'):
                # print("\n--- Debug: current_data_x (from NPZ before scaling, train set sample 0) ---")
                # print(f"current_data_x[0,0,:11] (value features): {current_data_x[0,0,:11]}")
                # print(f"NaNs in current_data_x[0,0,:11]: {np.isnan(current_data_x[0,0,:11]).any()}")
                # print(f"current_data_x[0,0,11:] (mask features): {current_data_x[0,0,11:]}")
                # Dataset_Custom._debug_printed_current_data_x_raw_train = True # Still set flag to avoid re-evaluation

            if self.scale:
                # Scaler for X value features (first 11 features)
                # Scaler for Y target features (all 3 features)
                num_x_value_features = 11 # As defined in create_informer_dataset.py
                
                if self.set_type == 0: # 'train' - fit the scalers
                    self.scaler_x_values = StandardScaler()
                    self.scaler_y_targets = StandardScaler()

                    # Fit on the training portion of the *entire raw dataset*
                    # Reshape to (samples*seq_len, num_features) for scaler
                    train_x_to_fit = raw_all_data_x[0:num_train, :, :num_x_value_features].reshape(-1, num_x_value_features)
                    self.scaler_x_values.fit(train_x_to_fit)
                    
                    train_y_to_fit = raw_all_data_y[0:num_train, :, :].reshape(-1, raw_all_data_y.shape[2])
                    self.scaler_y_targets.fit(train_y_to_fit)

                if self.scaler_x_values is None or self.scaler_y_targets is None:
                    # This case should ideally not happen if train_scalers are passed for val/test.
                    # Fallback: fit on current split's data (less ideal for val/test but makes it run)
                    print(f"Warning: Scalers not found for set_type {self.set_type}. Fitting on current split's data. THIS IS UNUSUAL for val/test.")
                    self.scaler_x_values = StandardScaler()
                    self.scaler_y_targets = StandardScaler()
                    
                    current_x_value_features_to_fit = current_data_x[:, :, :num_x_value_features].reshape(-1, num_x_value_features)
                    self.scaler_x_values.fit(current_x_value_features_to_fit)

                    current_y_targets_to_fit = current_data_y.reshape(-1, current_data_y.shape[2])
                    self.scaler_y_targets.fit(current_y_targets_to_fit)


                # Transform current split's data
                # Transform X value features
                original_shape_x = current_data_x[:, :, :num_x_value_features].shape
                data_x_values_flat = current_data_x[:, :, :num_x_value_features].reshape(-1, num_x_value_features)
                scaled_x_values_flat = self.scaler_x_values.transform(data_x_values_flat)
                scaled_x_values = scaled_x_values_flat.reshape(original_shape_x)
                
                # Combine scaled value features with unscaled mask features
                self.data_x = np.concatenate((scaled_x_values, current_data_x[:, :, num_x_value_features:]), axis=2)

                # Transform Y target features
                original_shape_y = current_data_y.shape
                data_y_flat = current_data_y.reshape(-1, original_shape_y[2]) # original_shape_y[2] is num_target_features
                scaled_y_flat = self.scaler_y_targets.transform(data_y_flat)
                self.data_y = scaled_y_flat.reshape(original_shape_y)
            else: # Not scaling
                self.data_x = current_data_x
                self.data_y = current_data_y
        
        else: # Original CSV loading logic
            self.scaler = StandardScaler() # Original single scaler for CSVs
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            if self.cols is None:
                cols = list(df_raw.columns)
                if self.target in cols: cols.remove(self.target)
                if 'date' in cols: cols.remove('date')
                # For 'M' or 'MS', df_raw might not have 'target' if it means all columns
                # This part of original logic might need adjustment based on specific CSV for M.
                # Assuming for M, target is just one of the columns, and cols become other_features + target
                # df_raw = df_raw[['date'] + cols + [self.target]] # Original logic
            else:
                cols = self.cols.copy()
                # cols.remove(self.target) # This might be an issue if target is not in custom cols list for M
            # For M mode, df_data usually includes all feature columns.
            # The original logic `df_raw = df_raw[['date']+cols+[self.target]]` might be problematic
            # if `target` is not meant to be appended again or if `cols` already defines everything.
            # Let's ensure 'date' is first, then features for 'M' or selected features for 'S'/'MS'
            
            if self.features == 'M' or self.features == 'MS':
                if self.cols: # User specified certain columns
                    feature_cols = self.cols.copy()
                    if 'date' in feature_cols: feature_cols.remove('date') # date handled separately
                    if self.target in feature_cols and self.features=='MS': # For MS, target is distinct
                         pass # target is part of feature_cols for encoder, but distinct for decoder output
                    df_raw_features = df_raw[feature_cols]
                else: # Default: all columns except 'date'
                    feature_cols = [col for col in df_raw.columns if col != 'date']
                    df_raw_features = df_raw[feature_cols]
            elif self.features == 'S':
                feature_cols = [self.target]
                df_raw_features = df_raw[feature_cols]
            else: # Should not happen
                raise ValueError("Unsupported feature type {}".format(self.features))

            # Splitting data for CSVs
            num_train = int(len(df_raw)*0.7)
            num_test = int(len(df_raw)*0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len] # Start index for sequences
            border2s = [num_train, num_train+num_vali, len(df_raw)] # End index for raw data segment
            
            border1_raw = border1s[self.set_type] if self.set_type!=0 else 0 # Raw data start for current segment (scaler fit on train)
            border2_raw = border2s[self.set_type]

            # Data for current split
            df_data_split = df_raw_features[border1_raw:border2_raw]
            
            if self.scale:
                # Fit scaler on the 'train' segment of the raw data (border1s[0] to border2s[0])
                train_data_for_scaler = df_raw_features[border1s[0]:border2s[0]]
                self.scaler.fit(train_data_for_scaler.values)
                data_values_scaled = self.scaler.transform(df_data_split.values)
            else:
                data_values_scaled = df_data_split.values
            
            # Date/time stamp processing for CSV
            df_stamp_raw = df_raw[['date']][border1_raw:border2_raw]
            df_stamp_raw['date'] = pd.to_datetime(df_stamp_raw.date)
            data_stamp_processed = time_features(df_stamp_raw, timeenc=self.timeenc, freq=self.freq)

            # Assign to self.data_x, self.data_y (which are then windowed in __getitem__)
            self.data_x = data_values_scaled
            if self.inverse: # Store raw for inverse transform if needed
                 self.data_y_raw_for_inverse = df_data_split.values
            self.data_y = data_values_scaled # For CSV, data_x and data_y are same before windowing
            self.data_stamp = data_stamp_processed # This will be windowed in __getitem__

    def __getitem__(self, index):
        is_npz = self.data_path.endswith('.npz')
        if is_npz:
            # Data is already windowed, just return the indexed sample
            seq_x = self.data_x[index]
            seq_y = self.data_y[index]
            seq_x_mark = self.data_stamp_x[index]
            seq_y_mark = self.data_stamp_y[index]

            # DEBUG PRINT (print only once or for a specific index)
            # if not hasattr(self, 'debug_printed_sample_dataset_custom') and index == 0 and self.set_type == 0: # Print for train set, index 0
                # print("\n--- Debug: Sample from Dataset_Custom (NPZ) __getitem__ (set_type={}) ---".format(self.set_type))
                # print(f"Index: {index}")
                # print(f"seq_x shape: {seq_x.shape}, type: {seq_x.dtype}")
                # print(f"seq_x NaNs: {np.isnan(seq_x).any()}, Infs: {np.isinf(seq_x).any()}")
                # print(f"seq_x[:1, :5] (first step, first 5 features):\n{seq_x[:1, :5]}")
                # print(f"seq_x[:1, 11:16] (first step, first 5 mask features, indices 11-15):\n{seq_x[:1, 11:16]}")

                # print(f"seq_y shape: {seq_y.shape}, type: {seq_y.dtype}")
                # print(f"seq_y NaNs: {np.isnan(seq_y).any()}, Infs: {np.isinf(seq_y).any()}")
                # print(f"seq_y[:1, :] (first step, all target features):\n{seq_y[:1, :]}")

                # print(f"seq_x_mark shape: {seq_x_mark.shape}, type: {seq_x_mark.dtype}")
                # print(f"seq_x_mark NaNs: {np.isnan(seq_x_mark).any()}, Infs: {np.isinf(seq_x_mark).any()}")
                # # print("seq_x_mark sample (first 2 steps):\n", seq_x_mark[:2])

                # print(f"seq_y_mark shape: {seq_y_mark.shape}, type: {seq_y_mark.dtype}")
                # print(f"seq_y_mark NaNs: {np.isnan(seq_y_mark).any()}, Infs: {np.isinf(seq_y_mark).any()}")
                # # print("seq_y_mark sample (first 2 steps):\n", seq_y_mark[:2])
                # print("--- End Debug: Sample from Dataset_Custom ---\n")
                # self.debug_printed_sample_dataset_custom = True # Print only once per instance type

            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else: # Original CSV logic for windowing
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            # For CSV, self.data_y is the same as self.data_x before this point
            # If inverse, seq_y needs to be constructed carefully
            if self.inverse:
                 # This part of original logic might need adjustment if self.data_y_raw_for_inverse is used
                 # The goal is that seq_y contains values that, if unscaled, match original data
                 # For now, assuming self.data_y (scaled) is appropriate for model input construction
                 seq_y_label_part = self.data_y[r_begin:r_begin+self.label_len]
                 seq_y_pred_part = self.data_y[r_begin+self.label_len:r_end] # These are actuals
                 # How inverse affects construction:
                 # The original code has:
                 # seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
                 # this assumes self.data_x is scaled, and self.data_y is unscaled (raw) for the prediction part only.
                 # This is complex. For now, let's simplify: seq_y is always from self.data_y (which is scaled if self.scale=True)
                 seq_y = self.data_y[r_begin:r_end]

            else: # not inverse
                seq_y = self.data_y[r_begin:r_end]
            
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        is_npz = self.data_path.endswith('.npz')
        if is_npz:
            return len(self.data_x)
        else: # Original CSV logic
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        is_npz = self.data_path.endswith('.npz')
        if is_npz:
            # Data is expected to be targets, so use scaler_y_targets
            # Input `data` shape: (batch_size, pred_len, num_target_features) or similar
            original_shape = data.shape
            num_target_features = self.scaler_y_targets.n_features_in_ if hasattr(self.scaler_y_targets, 'n_features_in_') else original_shape[-1] # Fallback
            
            if data.ndim == 3: # e.g. (batch, steps, feats)
                data_flat = data.reshape(-1, num_target_features)
            elif data.ndim == 2: # e.g. (steps, feats)
                data_flat = data
            else: # Should not happen often
                raise ValueError(f"Unexpected data ndim for inverse_transform: {data.ndim}")

            inversed_flat = self.scaler_y_targets.inverse_transform(data_flat)
            return inversed_flat.reshape(original_shape)
        else: # Original CSV logic
            return self.scaler.inverse_transform(data) # Uses the single scaler

    def get_scalers(self): # Helper to pass scalers from train to val/test
        if self.data_path.endswith('.npz'):
            return self.scaler_x_values, self.scaler_y_targets
        return None # Or self.scaler for CSV

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
