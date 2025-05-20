from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        print("--- Exp_Informer __init__ successfully completed ---")
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'PATIENT_CUSTOM': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        train_scalers_for_custom_npz = None
        if flag == 'train':
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols
            )
            if self.args.data == 'PATIENT_CUSTOM' and hasattr(data_set, 'get_scalers'):
                self.train_scalers_for_custom_npz = data_set.get_scalers()
        elif flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
            current_train_scalers = self.train_scalers_for_custom_npz if hasattr(self, 'train_scalers_for_custom_npz') else None
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols,
                train_scalers=current_train_scalers if self.args.data == 'PATIENT_CUSTOM' else None
            )
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data_pred = Dataset_Pred
            current_train_scalers = self.train_scalers_for_custom_npz if hasattr(self, 'train_scalers_for_custom_npz') else None
            data_set = Data_pred(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols,
            )
        else:
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
            current_train_scalers = self.train_scalers_for_custom_npz if hasattr(self, 'train_scalers_for_custom_npz') else None
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols,
                train_scalers=current_train_scalers if self.args.data == 'PATIENT_CUSTOM' else None
            )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        print(f"--- Exp_Informer train method started for setting: {setting} ---")
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y_original_from_loader = batch_y.float() # Keep original batch_y before modifications
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # --- BEGIN DEBUG PRINTS for _process_one_batch inputs ---
        print_this_batch_debug = False
        # if not hasattr(Exp_Informer, '_debug_printed_process_one_batch_globally_v3') or not Exp_Informer._debug_printed_process_one_batch_globally_v3:
            # Exp_Informer._debug_printed_process_one_batch_globally_v3 = True # Enable printing for the first batch
            # print_this_batch_debug = True
        
        if print_this_batch_debug: # This whole block will be skipped
            # This will print for the first batch processed by any instance of Exp_Informer in this run
            print("\n--- Debug: Data in _process_one_batch (start) ---")
            print(f"batch_x shape: {batch_x.shape}, type: {batch_x.dtype}, device: {batch_x.device}")
            print(f"batch_x NaNs: {torch.isnan(batch_x).any().item()}, Infs: {torch.isinf(batch_x).any().item()}")
            if batch_x.numel() > 0:
                print(f"batch_x[0, 0, :5]:\n{batch_x[0, 0, :5]}")
                print(f"batch_x[0, 0, 11:16] (masks for first 5 value features):\n{batch_x[0, 0, 11:16]}") 

            print(f"batch_y_original_from_loader shape: {batch_y_original_from_loader.shape}, type: {batch_y_original_from_loader.dtype}")
            print(f"batch_y_original_from_loader NaNs: {torch.isnan(batch_y_original_from_loader).any().item()}, Infs: {torch.isinf(batch_y_original_from_loader).any().item()}")
            if batch_y_original_from_loader.numel() > 0:
                print(f"batch_y_original_from_loader[0, 0, :]:\n{batch_y_original_from_loader[0, 0, :]}")

            print(f"batch_x_mark shape: {batch_x_mark.shape}, type: {batch_x_mark.dtype}")
            print(f"batch_x_mark NaNs: {torch.isnan(batch_x_mark).any().item()}, Infs: {torch.isinf(batch_x_mark).any().item()}")
            print(f"batch_y_mark shape: {batch_y_mark.shape}, type: {batch_y_mark.dtype}")
            print(f"batch_y_mark NaNs: {torch.isnan(batch_y_mark).any().item()}, Infs: {torch.isinf(batch_y_mark).any().item()}")
        # --- END DEBUG PRINTS ---

        # decoder input
        if self.args.padding==0:
            dec_inp_padding = torch.zeros([batch_y_original_from_loader.shape[0], self.args.pred_len, batch_y_original_from_loader.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp_padding = torch.ones([batch_y_original_from_loader.shape[0], self.args.pred_len, batch_y_original_from_loader.shape[-1]]).float()
        
        dec_inp = torch.cat([batch_y_original_from_loader[:,:self.args.label_len,:], dec_inp_padding], dim=1).float().to(self.device)

        if print_this_batch_debug:
            print(f"dec_inp shape: {dec_inp.shape}, type: {dec_inp.dtype}, device: {dec_inp.device}")
            print(f"dec_inp NaNs: {torch.isnan(dec_inp).any().item()}, Infs: {torch.isinf(dec_inp).any().item()}")
            print(f"dec_inp[0,0,:] (label part):\n{dec_inp[0,0,:]}") 
            print(f"dec_inp[0,self.args.label_len,:] (padding part):\n{dec_inp[0,self.args.label_len,:]}")

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        if print_this_batch_debug:
            print(f"outputs (model prediction) shape: {outputs.shape}, type: {outputs.dtype}")
            print(f"outputs NaNs: {torch.isnan(outputs).any().item()}, Infs: {torch.isinf(outputs).any().item()}")
            print(f"outputs[0,0,:] (first sample, first pred step, all targets):\n{outputs[0,0,:]}")

        if self.args.inverse:
            outputs_before_inverse = outputs.clone()
            outputs = dataset_object.inverse_transform(outputs)
            if print_this_batch_debug:
                print(f"outputs (after inverse_transform) NaNs: {torch.isnan(outputs).any().item()}, Infs: {torch.isinf(outputs).any().item()}")
                if torch.isnan(outputs).any().item() and not torch.isnan(outputs_before_inverse).any().item():
                     print("NaNs introduced by inverse_transform!")
       
        true_y = batch_y_original_from_loader[:,-self.args.pred_len:,:].to(self.device)

        if print_this_batch_debug:
            print(f"true_y (for loss) shape: {true_y.shape}, type: {true_y.dtype}")
            print(f"true_y NaNs: {torch.isnan(true_y).any().item()}, Infs: {torch.isinf(true_y).any().item()}")
            print(f"true_y[0,0,:] (first sample, first actual pred step, all targets):\n{true_y[0,0,:]}")
            print("--- End Debug: Data in _process_one_batch (end) ---")
            # Exp_Informer._debug_printed_process_one_batch_globally_v3 = True # This line is redundant if set above
        
        return outputs, true_y
