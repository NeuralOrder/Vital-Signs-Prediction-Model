import numpy as np
import torch

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self, epsilon=1e-8):
        self.mean = 0.
        self.std = 1.
        self.epsilon = epsilon
    
    def fit(self, data):
        # data is expected to be a 2D numpy array (samples, features)
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data

        self.mean = np.nanmean(data_np, axis=0) # Calculate mean ignoring NaNs
        self.std = np.nanstd(data_np, axis=0)   # Calculate std ignoring NaNs
        
        # Replace NaN means (all-NaN column) with 0 and NaN/zero stds with 1
        self.mean[np.isnan(self.mean)] = 0
        self.std[np.isnan(self.std)] = 1
        self.std[self.std < self.epsilon] = 1 # Avoid division by zero or very small std

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / (std + self.epsilon)

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        
        mean_selected = mean
        std_selected = std
        if data.shape[-1] < mean.shape[-1]: 
             mean_selected = mean[-data.shape[-1]:] 
             std_selected = std[-data.shape[-1]:]   
        elif data.shape[-1] > mean.shape[-1] and mean.shape[-1]==1:
            mean_selected = mean.repeat(data.shape[-1])
            std_selected = std.repeat(data.shape[-1])

        return (data * (std_selected + self.epsilon)) + mean_selected