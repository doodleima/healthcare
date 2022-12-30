import os
import torch
import random

import torchio as tio
import numpy as np

from glob import glob
from tqdm import tqdm
# from random import randint


class torchCustomDataset():
    def __init__(self, base_path, num_of_cases, transform=None, isTest=False, shuffle=True):
        MODE_DICT = {"train":"Tr", "test":"Ts"}
        self.FLAG = 0

        self.labels_path = os.path.join(base_path, 'labelsTr') 
        self.images_path = os.path.join(base_path, 'imagesTr')
        
        self.transform = transform
    
        self.imagesList = []
        self.labelsList = glob('{}/*.nii.gz'.format(self.labels_path))[:num_of_cases]

        if isTest == True:
            print("Testing mode: 10 cases")
            self.labelsList = self.labelsList[:10]

        if shuffle == True: random.shuffle(self.labelsList)
        else: self.labelsList = sorted(self.labelsList)        

        for labels in self.labelsList: self.imagesList.append(labels.replace('labels', 'images')) # imagesList

    def __len__(self):
        return len(self.imagesList)

    def __getitem__(self, idx) :
        SUBJECTS = []
        for (img_path, lab_path) in tqdm(zip(self.imagesList[idx], self.labelsList[idx])):
            # print(img_path, lab_path) # print debugging
            if self.FLAG == 0:
                self.SPACE_REF = tio.LabelMap(lab_path)
                self.FLAG = 1

            subject = tio.Subject(
                IMG = tio.ScalarImage(img_path),
                LABEL = tio.LabelMap(lab_path),
            )
            
            # if subject.IMG.data.shape[1] < subject.IMG.data.shape[3]: subject.IMG.data = torch.transpose(subject.IMG.data, 1, 3)
            self.transform_init = tio.Compose([tio.ToCanonical(), tio.Resample(self.SPACE_REF)])

            subject_init = self.transform_init(subject)
            subject_transformed = self.transform(subject_init)
            label_target = subject_transformed.LABEL.data
            label_shape = label_target.squeeze()
           
            ### white, grey matter, CSF and head fat to detect 
            bg = torch.zeros((label_target.shape)).squeeze()
            # wb = torch.zeros((label_target.shape)).squeeze()
            # for i in [1,2,3,5]: wb[torch.where((label_shape == i))] = 1

            gm = torch.zeros((label_target.shape)).squeeze()
            wm = torch.zeros((label_target.shape)).squeeze()
            csf = torch.zeros((label_target.shape)).squeeze()
            stem = torch.zeros((label_target.shape)).squeeze()
            # # fat = torch.zeros((label_target.shape)).squeeze()

            # # ukbiobank >> reversed(gm, wm, csf)
            y1, x1, z1 = torch.where((label_shape == 1)) # gm
            y2, x2, z2 = torch.where((label_shape == 2)) # wm
            y3, x3, z3 = torch.where((label_shape == 3)) # csf
            y5, x5, z5 = torch.where((label_shape == 5)) # brain stem
            # # y4, x4, z4 = torch.where((label_shape == 5)) # fat

            gm[y1, x1, z1] = 1
            wm[y2, x2, z2] = 2
            csf[y3, x3, z3] = 3
            stem[y5, x5, z5] = 4
            # # fat[y4, x4, z4] = 5

            # subject_transformed.LABEL.data = torch.stack((bg, wb))
            # subject_transformed.LABEL.data = torch.stack((bg, gm, wm, csf, fat, stem))
            subject_transformed.LABEL.data = torch.stack((bg, gm, wm, csf, stem))
            SUBJECTS.append(subject_transformed)

        return SUBJECTS


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, model_name='swinUNETR', path='checkpoint_latest.pt', delta=0):
        self.counter = 0                # accumlated value
        self.delta = delta              # decay
        self.path = path                # target path (model)
        self.model_name = model_name

        if os.path.exists(path):
            self.model_loaded = True
            self.model_weights = torch.load(path)
            self.best_score = self.model_weights["model_loss"]
        
        else:
            self.model_loaded = False
            self.best_score = np.Inf


    def __call__(self, val_loss, model):        
        ### prev score > best score
        if val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f"[VALID LOSS] Best score({self.best_score:.4f}) is better than Current one({val_loss:.4f}), Accumulated {self.counter}")
            torch.save({"model": self.model_name.replace('_best', '_latest'), 
                        "model_state_dict": model.state_dict(),
                        "model_loss":val_loss}, self.path.replace('_best', '_latest'))

        ### best score > prev score
        else:
            print(f'[VALID LOSS] Best score({self.best_score:.4f}), Current({val_loss:.4f})')
            torch.save({"model": self.model_name.replace('_latest', '_best'),
                    "model_state_dict": model.state_dict(),
                    "model_loss":val_loss}, self.path.replace('_latest', '_best'))

            self.best_score = val_loss        
            self.counter = 0