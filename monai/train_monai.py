import os
import sys
import warnings
import torch
import torchinfo

import torchio as tio
import numpy as np

from tqdm import tqdm
from torchio import DATA
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import UNETR_MONAI, NNUNET_MONAI
from dataloader_monai import CustomDataset, EarlyStopping

# from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract

from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    RandAffined
)

import torchsummary
import multiprocessing
# import torch.nn as nn
# import torch.nn.functional as F

#########################################################################################################################
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available() == True: print(f"GPU [{torch.cuda.get_device_name(0)}]: {torch.cuda.device_count()}")
#########################################################################################################################
H = 128
W = 128
D = 128

def stack_segments(label):
    bg = torch.zeros((label.shape)).squeeze()
    gm = torch.zeros((label.shape)).squeeze()
    wm = torch.zeros((label.shape)).squeeze()
    csf = torch.zeros((label.shape)).squeeze()
    fat = torch.zeros((label.shape)).squeeze()

    y1, x1, z1 = torch.where((label.squeeze() == 1)) # gm
    y2, x2, z2 = torch.where((label.squeeze() == 2)) # wm
    y3, x3, z3 = torch.where((label.squeeze() == 3)) # csf
    y4, x4, z4 = torch.where((label.squeeze() == 4)) # fat

    gm[y1, x1, z1] = 1
    wm[y2, x2, z2] = 2
    csf[y3, x3, z3] = 3
    fat[y4, x4, z4] = 4
    
    label_stacked = torch.stack((bg, gm, wm, csf, fat))

    return label_stacked.unsqueeze(dim=0)


### model train
def train(dataloader, model, optimizer, loss_function):
    model.train()
    train_loss = 0.0
    step = 0

    for batch in tqdm(dataloader):
        step += 1
        img, label = batch["image"], stack_segments(batch["label"])        
        img, label = (img.to(device), label.to(device))

        optimizer.zero_grad()
        try:
            output = model(img)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        except:
            print(img.shape)

    train_loss = train_loss / len(dataloader)
    print(f"Train Loss : {round(train_loss, 4)}")


### model eval
def eval(dataloader, model, loss_function):
    model.eval()
    val_loss = 0.0

    ### valid model
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img, label = batch["image"], stack_segments(batch["label"])        
            img, label = (img.to(device), label.to(device))
            # img, label = (batch["image"].to(device), batch["label"].to(device))

            output = model(img)
            loss = loss_function(output, label)

            val_loss += loss.item()

        val_loss = val_loss / len(dataloader)
        print(f"Valid Loss : {round(val_loss, 4)}")

        # must deactivate this if use earlystopping method
        torch.save(model.state_dict(), os.path.join(MODEL_STORE, 'nnunet_chkpoint.pt'))
        print("saved new best metric model")     


if __name__ == "__main__":
    BASE = '/home/pmx/src/pytorch/data_kku'
    MODEL_STORE = '/home/pmx/model/trained' # model store path

    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.2
    BATCH_SIZE = 1
    EPOCH_SIZE = 500
    PATIENCE = 20

    transform_monai = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        # RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(64, 64, 64), pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
        RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.5, spatial_size=(176, 176, 176), rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1))
    ])

    subject = CustomDataset(BASE, transform_monai)
    total_dataset = (len(subject))


    TRAIN_SIZE = int(TRAIN_RATIO * total_dataset)
    VALID_SIZE = total_dataset - TRAIN_SIZE

    # print(TRAIN_SIZE, VALID_SIZE)
    train_data = subject[:TRAIN_SIZE]
    valid_data = subject[-VALID_SIZE:]


    train_dataset = Dataset(data=train_data)#, transform=transform_monai)
    valid_dataset = Dataset(data=valid_data)#, transform=transform_monai)

    TRAIN_LOADER = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=0,#multiprocessing.cpu_count(), # import multiprocessing
        pin_memory=False
    )

    VALID_LOADER = DataLoader(
        valid_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=0,#multiprocessing.cpu_count(), # import multiprocessing
        pin_memory=False
    )

    model = UNETR_MONAI() # nnunet official(monai)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True).to(device) # DiceCE(Cross Entropy) loss monai

    # summary
    # print(torchsummary.summary(model, (1, 128, 128, 128)))
    # print(torchinfo.summary(model, input_size=(BATCH_SIZE,1,128,128,128))) 
    
    for epochs in range(EPOCH_SIZE):
        print(f"Epoch: {epochs+1}")
        train(TRAIN_LOADER, model, optimizer, criterion)
        eval(VALID_LOADER, model, criterion)

    print("Train Finished")
