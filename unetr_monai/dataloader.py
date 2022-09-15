import os
import torch

import torchio as tio
import numpy as np

from glob import glob
from tqdm import tqdm

### Height, Width, Dimension Config ###
H = 256
W = 256
D = 176


class nnUNet_CustomDataset():
    def __init__(self, base_path, mode, NUM_OF_DATA):
        MODE_DICT = {"train":"Tr", "test":"Ts"}
        self.FLAG = 0
        self.NUM_OF_DATA = NUM_OF_DATA

        self.labels_path = os.path.join(base_path, 'labels{}'.format(MODE_DICT[mode])) 
        self.images_path = os.path.join(base_path, 'images{}'.format(MODE_DICT[mode]))

        self.labelsList = sorted(glob('{}/*.nii.gz'.format(self.labels_path)))[1000:1000+self.NUM_OF_DATA]
        self.imagesList = sorted(glob('{}/*.nii.gz'.format(self.images_path)))[1000:1000+self.NUM_OF_DATA]

    def __len__(self):
        # assert len(labelsList) == len(imagesList)
        return len(self.imagesList)

    def __getitem__(self, idx) :
        SUBJECTS = []
        for (img_path, lab_path) in tqdm(zip(self.imagesList[idx], self.labelsList[idx])):
            if self.FLAG == 0:
                self.SPACE_REF = tio.LabelMap(lab_path)
                self.FLAG = 1

            subject = tio.Subject(
                IMG = tio.ScalarImage(img_path),
                LABEL = tio.LabelMap(lab_path),
            )

            transform = tio.Compose([tio.ToCanonical(), 
                                    tio.Resample(self.SPACE_REF),
                                    tio.CropOrPad((H,W,D), mask_name='LABEL'),
                                    tio.RescaleIntensity(out_min_max=(0, 1)),
                                    tio.ZNormalization(masking_method=lambda x: x > x.mean()),
                                    #  tio.EnsureShapeMultiple(16),
                                    tio.RandomFlip(axes=('LRAPIS'), flip_probability=0.5),
                                    tio.RandomGamma(log_gamma=(-0.2, 0.2)),
                                    ])

            # if subject.IMG.data.shape[1] < subject.IMG.data.shape[3]: 
            #     subject.IMG.data = torch.transpose(subject.IMG.data, 1, 3)
            subject_transformed = transform(subject)

            label_target = subject_transformed.LABEL.data

            ### change this part when you want to making a multi-label based label ###
            bg = torch.zeros((label_target.shape)).squeeze() 
            gm = torch.zeros((label_target.shape)).squeeze()
            wm = torch.zeros((label_target.shape)).squeeze()
            csf = torch.zeros((label_target.shape)).squeeze()
            fat = torch.zeros((label_target.shape)).squeeze()

            y1, x1, z1 = torch.where((label_target.squeeze() == 1)) # gm
            y2, x2, z2 = torch.where((label_target.squeeze() == 2)) # wm
            y3, x3, z3 = torch.where((label_target.squeeze() == 3)) # csf
            y4, x4, z4 = torch.where((label_target.squeeze() == 4)) # fat

            gm[y1, x1, z1] = 1
            wm[y2, x2, z2] = 2
            csf[y3, x3, z3] = 3
            fat[y4, x4, z4] = 4

            subject_transformed.LABEL.data = torch.stack((bg, gm, wm, csf, fat))
            ### change this part when you want to making a multi-label based label ###

            SUBJECTS.append(subject_transformed)

        return SUBJECTS


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
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save({"model": "UNetR3d",
                    "model_state_dict": model.state_dict()}, self.path)
        
        # torch.save(model.state_dict(), self.path)
        
        self.val_loss_min = val_loss
