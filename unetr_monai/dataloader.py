import os
import torch

import nibabel as nib
import numpy as np
import skimage.transform as skTrans

from glob import glob


class nnUNet_CustomDataset():
    def __init__(self, base_path, mode, transform=None, target_transform=None):
        self.mode_dict = {"train":"Tr", "test":"Ts"}

        self.labels_path = os.path.join(base_path, 'labels{}'.format(self.mode_dict[mode]))
        self.images_path = os.path.join(base_path, 'images{}'.format(self.mode_dict[mode]))

        self.labelsList = sorted(glob('{}/*.nii.gz'.format(self.labels_path)))
        self.imagesList = sorted(glob('{}/*.nii.gz'.format(self.images_path)))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        assert len(self.labelsList) == len(self.imagesList)
        return len(self.labelsList) 

    def __getitem__(self, idx) :
        label_path = os.path.join(self.labels_path, self.labelsList[idx])
        image_path = os.path.join(self.images_path, self.imagesList[idx]) # {}_0000.nii.gz

        # nifti
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        for niis in [image, label]:
            if niis.shape[0] > niis.shape[2]: niis = np.swapaxes(niis, 0, 2) # swap axis if it reversed

        
        if self.transform: 
            image_norm = ((image - np.min(image)) / (np.max(image) - np.min(image)))
            image = skTrans.resize(image_norm, (128,128,128))
            image = self.transform(image)
                
        # ### normalize ###
        # img_resized -= img_resized.min(1, keepdim=True)[0]
        # img_resized /= img_resized.max(1, keepdim=True)[0]

        if self.target_transform: 
            label = skTrans.resize(label, (128,128,128))
            label = label.astype(np.int8)

            # wb
            bg = np.zeros((label.shape))
            wb = np.zeros((label.shape))
            fat = np.zeros((label.shape))

            y1, x1, z1 = np.where((label == 1)) # gm
            y2, x2, z2 = np.where((label == 2)) # wm

            wb[y1, x1, z1] = 1
            fat[y2, x2, z2] = 2 

            bg = self.target_transform(bg)
            wb = self.target_transform(wb)
            fat = self.target_transform(fat)

            # label = self.target_transform(label)
        
        img_resized = torch.unsqueeze(image, dim=0)
        # img_resized = torch.vstack((img_resized, img_resized, img_resized, img_resized))
       
        # lab_resized = torch.unsqueeze(label, dim=0)
        for j in [bg, wb, fat]: j = torch.unsqueeze(j, dim=0)
        lab_resized = torch.stack((bg, wb, fat))


        return img_resized, lab_resized
