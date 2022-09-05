import os
import torch

import torchio as tio
import numpy as np

from glob import glob
from tqdm import tqdm

def nnUNet_CustomDataset(base_path, mode, NUM_OF_DATA):
    MODE_DICT = {"train":"Tr", "test":"Ts"}
    SUBJECTS = []
    FLAG = 0

    labels_path = os.path.join(base_path, 'labels{}'.format(MODE_DICT[mode]))
    images_path = os.path.join(base_path, 'images{}'.format(MODE_DICT[mode]))

    labelsList = sorted(glob('{}/*.nii.gz'.format(labels_path)))[:NUM_OF_DATA]
    imagesList = sorted(glob('{}/*.nii.gz'.format(images_path)))[:NUM_OF_DATA]

    assert len(labelsList) == len(imagesList)
        
    for (img_path, lab_path) in tqdm(zip(imagesList, labelsList), desc=f"[Torchio DataLoader: {NUM_OF_DATA}]"):
        # print(img_path, lab_path)
        if FLAG == 0:
            # print('space reference has been created')
            img_refer = tio.ScalarImage(img_path)
            FLAG = 1

        subject = tio.Subject(
            SPACE_REFER = img_refer,
            IMG = tio.ScalarImage(img_path),
            LABEL = tio.LabelMap(lab_path),
        )

        transform = tio.Compose([tio.ToCanonical(), 
                                 tio.Resample('SPACE_REFER'),
                                 tio.Resize((128,128,128), image_interpolation='linear', label_interpolation='nearest'),
                                #  tio.CropOrPad((128,128,128), mask_name='LABEL'), # 128, 128, 128
                                ])
        subject_transformed = transform(subject)
        # print(subject_transformed.LABEL.count_labels())

        label_target = subject_transformed.LABEL.data
        # print(subject_transformed.LABEL.data.shape)

        ### example: this script for dataset which include 4 classes + Background
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

        # torch.save(model.state_dict(), self.path)
        torch.save({"model": "UNetR3d",
                    "model_state_dict": model.state_dict()}, self.path)
        
        self.val_loss_min = val_loss
