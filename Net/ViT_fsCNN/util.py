### DataLoader, LRscheduler, EarlyStopper ###
import os
import torch
import math
import random

import torchio as tio
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler
from glob import glob
from tqdm import tqdm


class torchCustomDataset:
    def __init__(self, base_path, num_cases, transform=None, test=False, shuffle=True, axis=0):
        self.FLAG = 0
        self.AXIS = axis

        self.images = os.path.join(base_path, 'labelsTr')
        self.labels = os.path.join(base_path, 'imagesTr')

        self.transform = transform

        self.imagesList = glob('{}/*.nii.gz'.format(self.images))
        self.labelsList = []

        self.imagesList = self.imagesList[:10] if test == True else self.imagesList # 10 cases for a simple test
        
        if shuffle == True: random.shuffle(self.imagesList)
        else: sorted(self.imagesList)
        self.imagesList = self.imagesList[:num_cases]

        for imgs in self.imagesList: self.labelsList.append(imgs.replace('images', 'labels'))


    def __len__(self):
        assert len(self.imagesList) == len(self.labelsList), "IMG/LABEL dataset lengths are not the same."
        return len(self.imagesList)


    def __getitem__(self, idx):
        SUBJECTS = []

        for (images, labels) in tqdm(zip(self.imagesList[idx], self.labelsList[idx])):
            if self.FLAG == 0:
                self.SPACE_REF = tio.LabelMap(labels)
                self.FLAG = 1

            self.transform_init = tio.Compose([tio.Resample(self.SPACE_REF)]) # unifying space info 
            self.transform = transform if transform != None else None

            subject = tio.Subject(IMG = tio.ScalarImage(images),LABEL = tio.LabelMap(labels),)
            subject_init = self.transform_init(subject)
            subject_tf = self.transform(subject_init)

            if self.AXIS != 0: # default sagittal
                target_axis = 3 if self.AXIS == 1 else 2
                # transpose with AXIS <-> target_axis - 1, 3(axial) or 1, 2(coronal)
                subject_tf.IMG.data = subject_tf.IMG.data.transpose(1, target_axis)
                subject_tf.LABEL.data = subject_tf.LABEL.data.transpose(1, target_axis)

            SUBJECTS.append(subject_tf)

        return SUBJECTS


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int): raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int): raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int): raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch) # _LRScheduler 상속
    

    def get_lr(self):
        if self.T_cur == -1: return self.base_lrs
        elif self.T_cur < self.T_up: return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else: return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2 for base_lr in self.base_lrs]


    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()): param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, net_name='ViT_FS', model_path='chkpoint_latest.pt', delta=0):
        self.counter = 0                # accumlated value
        self.delta = delta              # decay
        self.net_name = net_name        # network name
        self.model_path = model_path    # model(weight) path

        if os.path.exists(path):
            self.model_loaded = True
            self.model_weights = torch.load(model_path)
            self.best_score = self.model_weights["model_loss"]
        
        else:
            self.model_loaded = False
            self.best_score = np.Inf


    def __call__(self, val_loss, model):
        ### prev score > best score
        if val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f"[VALID LOSS] Best score({self.best_score:.4f}) is better than Current one({val_loss:.4f}), Accumulated {self.counter}")
            
            torch.save({"model": self.net_name.replace('_best', '_latest'), 
                        "model_state_dict": model.state_dict(),
                        "model_loss":val_loss}, self.path.replace('_best', '_latest'))

        ### best score > prev score
        else:
            print(f'[VALID LOSS] Best score({self.best_score:.4f}), Current({val_loss:.4f})')
            
            torch.save({"model": self.net_name.replace('_latest', '_best'),
                        "model_state_dict": model.state_dict(),
                        "model_loss":val_loss}, self.path.replace('_latest', '_best'))

            self.best_score = val_loss        
            self.counter = 0

