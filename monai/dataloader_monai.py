import os
import torch

import torchio as tio
import numpy as np

from glob import glob
from tqdm import tqdm
from random import randint

### Height, Width, Dimension Config ###
H = 256
W = 256
D = 176


class CustomDataset():
    def __init__(self, base_path, transform=None, test=False):
        self.labels_path = os.path.join(base_path, 'labelsTr') 
        self.images_path = os.path.join(base_path, 'imagesTr')

        self.transform = transform
        self.test = test

        self.imagesList = sorted(glob('{}/*.nii.gz'.format(self.images_path))) 
        self.labelsList = sorted(glob('{}/*.nii.gz'.format(self.labels_path))) 

        if test == True:
            self.imagesList = self.imagesList[:10] 
            self.labelsList = self.labelsList[:10] 

        # self.images_ts_path = os.path.join(base_path, 'imagesTs')
        # self.imagesTsList = sorted(glob('{}/*.nii.gz'.format(self.images_ts_path))) 

        self.data_dicts = [{"image":image_name, "label":label_name} for image_name, label_name in zip(self.imagesList, self.labelsList)]

    def __len__(self):
        return len(self.imagesList)

    def __getitem__(self, idx):
        transformed_dict = self.transform(self.data_dicts[idx]) if self.transform is not None else self.data_dicts[idx]
        return transformed_dict


class EarlyStopping:
    def __init__(self, patience=5, path='checkpoint.pt', verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_best(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_checkpoint(val_loss, model) # no
        else:
            self.best_score = score
            self.save_best(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save({"model": "UNetR_3d_monai",
                    "model_state_dict": model.state_dict()}, self.path.replace('checkpoint', 'checkpoint_latest'))
                
        self.val_loss_min = val_loss


    def save_best(self, val_loss, model):
        torch.save({"model": "UNetR_3d_monai",
                    "model_state_dict": model.state_dict()}, self.path.replace('checkpoint', 'checkpoint_best'))
        
        self.val_loss_min = val_loss
