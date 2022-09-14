import os
import warnings
import torch
import torchinfo

import torchio as tio

from tqdm import tqdm
from torchio import DATA
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

from model import UNETR_MONAI
from dataloader import nnUNet_CustomDataset, EarlyStopping


# import multiprocessing
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
H = 256
W = 256
D = 176

# ### save chkpoint weight if some conditions are 'True'
# def save_chkpoint(epoch_num, model, optimizer, cost, model_path):
#     print("EPOCH [{}], MODEL SAVE: {}".format(epochs+1, os.path.join(model_path, 'unetr_chkpoint.pt')))
#     torch.save({"model": "UNetR3d",
#                 "epoch": epoch_num,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "cost": cost,
#                 "description": "checkpoint_{}".format(epoch_num),}, os.path.join(model_path, 'unetr_chkpoint.pt'))
# else: pass


### model train
def train(dataloader, model, optimizer, loss_function, epoch):
    model.train()
    train_loss = 0.0

    ### train model
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'[TorchIO] TRAIN [{epoch}]')):
        train_data = batch['IMG'][DATA].to(device, dtype=torch.float) 
        train_label = batch['LABEL'][DATA].to(device, dtype=torch.int8)

        optimizer.zero_grad()
        output_train = model(train_data).to(dtype=torch.float)

        t_loss = loss_function(output_train, train_label)

        # backprop
        t_loss.backward()
        optimizer.step()

        train_loss += t_loss.item()

    train_loss = train_loss / len(dataloader)
    # print(f"Train Loss : {train_loss}")

    return train_loss


### model eval
def eval(dataloader, model, loss_function, epoch):
    model.eval()
    val_loss = 0.0

    ### valid model
    with torch.no_grad():

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'[TorchIO] VALID [{epoch}]')):
            valid_data = batch['IMG'][DATA].to(device, dtype=torch.float)
            valid_label = batch['LABEL'][DATA].to(device, dtype=torch.int8)
            
            output_val = model(valid_data)
            # valid_label = batch['LABEL'][DATA].to("cpu", dtype=torch.int8)
            v_loss = loss_function(output_val, valid_label)

            val_loss += v_loss.item()
        
    val_loss = val_loss / len(dataloader)
    # print(f"Valid Loss : {val_loss}\n")

    return val_loss


if __name__ == "__main__":
    BASE = '/home/pmx/src/pytorch/data'
    MODEL_STORE = '/home/pmx/model/trained' # model store path

    TOTAL_DATASET = 220 #280 # number of dataset for use
    TRAIN_RATIO = 0.85
    VALID_RATIO = 0.15
    BATCH_SIZE = 1
    EPOCH_SIZE = 200
    PATIENCE = 20

    subject = nnUNet_CustomDataset(BASE, 'train', TOTAL_DATASET) # train_data
    earlyStopping = EarlyStopping(PATIENCE, verbose = True, path = os.path.join(MODEL_STORE, 'unetr_chkpoint.pt')) # init earlystopping: _2 is SGD optim

    total_dataset = (len(subject))
    TRAIN_SIZE = int(TRAIN_RATIO * total_dataset)
    # VALID_SIZE = int(VALID_RATIO * total_dataset)

    train_subjects = subject[:TRAIN_SIZE]
    valid_subjects = subject[TRAIN_SIZE:]
    # valid_subjects = subject[TRAIN_SIZE:TRAIN_SIZE+VALID_SIZE]

    train_dataset = tio.SubjectsDataset(train_subjects)
    valid_dataset = tio.SubjectsDataset(valid_subjects)

    TRAIN_LOADER = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=0#multiprocessing.cpu_count(), # import multiprocessing
    )

    VALID_LOADER = DataLoader(
        valid_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=0#multiprocessing.cpu_count(), # import multiprocessing
    )

    model = UNETR_MONAI() # unetr official(monai)
    # print(torchinfo.summary(model, input_size=(BATCH_SIZE,1,H,W,D))) # summary

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-7)
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True).to(device) # DiceCE(Cross Entropy) loss monai
    
    for epochs in range(EPOCH_SIZE):
        train_loss = train(TRAIN_LOADER, model, optimizer, criterion, epochs)
        valid_loss = eval(VALID_LOADER, model, criterion, epochs)

        # if (epochs+1) % 5 == 0: save_chkpoint(epochs, model, criterion, train_loss, MODEL_STORE)
        earlyStopping(valid_loss, model)
        if earlyStopping.early_stop or valid_loss < 0: ### prevent overfitting
            print("Activate: Early Stopping")
            break

        print(f"RESULT: Train_{round(train_loss, 4)} | Valid_{round(valid_loss, 4)}")

    print("Train Finished")
