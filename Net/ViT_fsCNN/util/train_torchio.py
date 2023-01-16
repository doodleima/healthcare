import os
import sys
import warnings
import argparse
import multiprocessing
import random
import torch

import numpy as np
import torchio as tio

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from torchio import DATA
from monai.losses import DiceCELoss

p = os.path.abspath('..')
sys.path.insert(1, p)

from torchsummary import summary
from model import *
from dataloader_torchio import torchCustomDataset, EarlyStopping

import gc
gc.collect()
torch.cuda.empty_cache()

import wandb
wandb.init(project="fsCNN")

### torch.nn.parallel (multi GPU support)

#########################################################################################################################
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.autograd.set_detect_anomaly(True)
if torch.cuda.is_available() == True: print(f"GPU [{torch.cuda.get_device_name(0)}]: {torch.cuda.device_count()}")

# multi class labels: for wandb
class_labels = {0: "background",
                1: "Grey Matter",
                2: "White Matter",
                3: "CSF",
                4: "Brain Stem",
                }
#########################################################################################################################


### not finished yet ###
### model train - 2d, 배치 n에도 호환되도록 변경해야함
def train_2d(dataloader, model, optimizer, loss_function, epoch):
    model.train()
    train_loss = 0.0
    best_loss = np.Inf
    train_samples={}   

    input_slice, input_label_slice, output_label_slice = None, None, None

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'[TorchIO] TRAIN [{epoch}]')):
        
        train_data = batch['IMG'][DATA].to(device, dtype=torch.float) 
        train_label = batch['LABEL'][DATA].to(device, dtype=torch.int8)
        train_label = torch.argmax(train_label, dim=1)

        train_data_all = []
        t_data = None
        for i in range(len(train_data)):
            image_slice = train_data[i].squeeze(0) # image
            label_slice = train_label[i].squeeze(0) # label

            for j in range(len(image_slice)): # loss calculation with model output & GT
                t_output = model(image_slice[j].unsqueeze(0).unsqueeze(0)) # dummy dim (4D)
                t_output = torch.argmax(t_output, dim=1).to(device, dtype=torch.float) # make 3D shape
                t_loss = loss_function(t_output.squeeze(0), label_slice[j])

                if t_loss <= best_loss:
                    best_loss = t_loss
                    input_slice = image_slice[j]#.detach().numpy()
                    input_label_slice = label_slice[j]#.detach().numpy()
                    output_label_slice = t_output#.detach().numpy()

                train_data_all.append(t_output.squeeze(0))
            t_data = torch.stack(train_data_all).to(device, dtype=torch.float)

        tt_loss = loss_function(t_data.unsqueeze(0), train_label)
        train_loss += tt_loss.item()

        # backprop
        optimizer.zero_grad()
        tt_loss.requires_grad_(True) # no loss decrease > apply gradcam & check the output datas
        tt_loss.backward()
        optimizer.step() 

    train_samples["prediction"] = output_label_slice.squeeze(0).detach().cpu().numpy()
    train_samples["input"] = input_slice.detach().cpu().numpy()
    train_samples["gt"] = input_label_slice.detach().cpu().numpy()

    train_loss = train_loss / len(dataloader)
    wandb.log({"Train_Loss": train_loss, "Learning_Rates(T)": optimizer.param_groups[0]['lr'],
              "Train_Patches": wandb.Image(train_samples["input"], masks = {"ground_truth": {"mask_data": train_samples["gt"], "class_labels": class_labels},
                                                                            "predictions": {"mask_data": train_samples["prediction"], "class_labels": class_labels}})})

    return train_loss


### model eval - 2d
def eval_2d(dataloader, model, loss_function, epoch):
    model.eval()
    valid_loss = 0.0
    best_loss = np.Inf
    valid_samples={}   

    input_slice, input_label_slice, output_label_slice = None, None, None

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'[TorchIO] VALID [{epoch}]')):

            valid_data = batch['IMG'][DATA].to(device, dtype=torch.float) 
            valid_label = batch['LABEL'][DATA].to(device, dtype=torch.int8)
            valid_label = torch.argmax(valid_label, dim=1)

            valid_data_all = []
            v_data = None
            for i in range(len(valid_data)):
                image_slice = valid_data[i].squeeze(0) # image
                label_slice = valid_label[i].squeeze(0) # label

                for j in range(len(image_slice)): # loss calculation with model output & GT
                    v_output = model(image_slice[j].unsqueeze(0).unsqueeze(0)) # check if slice[j].to(device) == CUDA
                    v_output = torch.argmax(v_output, dim=1).to(device, dtype=torch.float) # make 3D shape
                    v_loss = loss_function(v_output.squeeze(0), label_slice[j])

                    if v_loss <= best_loss:
                        best_loss = v_loss
                        input_slice = image_slice[j]#.detach().numpy()
                        input_label_slice = label_slice[j]#.detach().numpy()
                        output_label_slice = v_output#.detach().numpy()
                    
                    valid_data_all.append(v_output.squeeze(0))
                v_data = torch.stack(valid_data_all).to(device, dtype=torch.float)

            vv_loss = loss_function(v_data.unsqueeze(0), valid_label)
            valid_loss += vv_loss.item()


    valid_samples["prediction"] = output_label_slice.squeeze(0).detach().cpu().numpy()
    valid_samples["input"] = input_slice.detach().cpu().numpy()
    valid_samples["gt"] = input_label_slice.detach().cpu().numpy()

    valid_loss = valid_loss / len(dataloader)
    wandb.log({"Valid_Loss": valid_loss, "Learning_Rates(T)": optimizer.param_groups[0]['lr'],
              "Valid_Patches": wandb.Image(valid_samples["input"], masks = {"ground_truth": {"mask_data": valid_samples["gt"], "class_labels": class_labels},
                                                                            "predictions": {"mask_data": valid_samples["prediction"], "class_labels": class_labels}})})

    return train_loss
### not finished yet ###


### model train
def train(dataloader, model, optimizer, loss_function, epoch):
    model.train()
    train_loss = 0.0
    train_samples={}

    input_train_patch, input_label_train_patch, output_train_patch = None, None, None

    ### train model
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'[TorchIO] TRAIN [{epoch}]')):

        train_data = batch['IMG'][DATA].to(device, dtype=torch.float) 
        train_label = batch['LABEL'][DATA].to(device, dtype=torch.int8)

        output_train = model(train_data)        
        t_loss = loss_function(output_train, train_label)
        train_loss += t_loss.item()

        # backprop
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()

        # patch sample for wandb
        input_train_patch = train_data.cpu()
        input_label_train_patch = train_label.cpu()
        output_train_patch = output_train.cpu()

    train_label_patch = torch.argmax(output_train_patch[0], 0)
    train_label_gt_patch = torch.argmax(input_label_train_patch[0], 0)

    train_image_patch = input_train_patch[0].squeeze().squeeze() # batch 2 이상이면 input_train_patch 뒤에 [0 ~ n]
    
    max = int(len(train_image_patch) - 50)
    midval = random.randint(49, max)

    train_samples["prediction"] = train_label_patch[midval].detach().numpy()
    train_samples["input"] = train_image_patch[midval].detach().numpy()
    train_samples["gt"] = train_label_gt_patch[midval].detach().numpy()

    train_loss = train_loss / len(dataloader)
    wandb.log({"Train_Loss": train_loss, "Learning_Rates(T)": optimizer.param_groups[0]['lr'],
              "Train_Patches": wandb.Image(train_samples["input"], masks = {"ground_truth": {"mask_data": train_samples["gt"], "class_labels": class_labels},
                                                                            "predictions": {"mask_data": train_samples["prediction"], "class_labels": class_labels}})})

    return train_loss


### model eval
def eval(dataloader, model, loss_function, epoch):
    model.eval()
    val_loss = 0.0
    val_samples={}

    input_val_patch, input_label_val_patch, output_val_patch = None, None, None

    ### valid model
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'[TorchIO] VALID [{epoch}]')):

            valid_data = batch['IMG'][DATA].to(device, dtype=torch.float)
            valid_label = batch['LABEL'][DATA].to(device, dtype=torch.int8)
            
            output_val = model(valid_data)

            v_loss = loss_function(output_val, valid_label)
            val_loss += v_loss.item()

            # patch sample for wandb
            input_val_patch = valid_data.cpu()
            input_label_val_patch = valid_label.cpu()
            output_val_patch = output_val.cpu()

    valid_label_patch = torch.argmax(output_val_patch[0], 0)
    valid_label_gt_patch = torch.argmax(input_label_val_patch[0], 0)

    valid_image_patch = input_val_patch[0].squeeze().squeeze()
    
    max = int(len(valid_image_patch) - 50)
    midval = random.randint(49, max)

    val_samples["prediction"] = valid_label_patch[midval].detach().numpy()
    val_samples["input"] = valid_image_patch[midval].detach().numpy()
    val_samples["gt"] = valid_label_gt_patch[midval].detach().numpy()
    
    val_loss = val_loss / len(dataloader)
    wandb.log({"Valid_Loss": val_loss, #"Learning_Rates(E)": optimizer.param_groups[0]['lr'], 
               "Valid_Patches": wandb.Image(val_samples["input"], masks = {"ground_truth": {"mask_data": val_samples["gt"], "class_labels": class_labels},
                                                                           "predictions": {"mask_data": val_samples["prediction"], "class_labels": class_labels}})})

    return val_loss


def transform_tio():
    transform = tio.Compose([tio.CropOrPad((256), mask_name='LABEL'), # (N, N, N) >> resize
                             tio.Resize((160), label_interpolation='label_gaussian'), 
                             tio.ToCanonical(),
                             tio.RescaleIntensity(out_min_max=(0, 1)),
                            #  tio.EnsureShapeMultiple(16),
                            #  tio.RandomBiasField(coefficients=(-0.1, 0.1), order=3),
                            #  tio.RandomGamma(log_gamma=(-0.1, 0.1)),
                            #  tio.RandomAnisotropy(axes=(0,1,2), downsampling=(1.5,5), image_interpolation='linear'),
                            #  tio.RandomNoise(mean=0, std=(0, 0.1)),
                            #  tio.RandomGhosting(num_ghosts=(0,2), axes=(0,1,2), intensity=(0.1, 0.2), restore=0.02),
                            #  tio.ZNormalization(masking_method=lambda x: x > x.mean()),
                            ])

    return transform



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--test", help="Test pipeline with 10 cases", action="store_true")
    parser.add_argument("-s", "--shuffle", help="Dataloader: shuffle", action="store_true")
    parser.add_argument("-i", "--input_dir", type=str, help="input data path", required=True)
    parser.add_argument("-o", "--model_output", type=str, help="model weight path", required=True)
    parser.add_argument("-n", "--model_name", type=str, help="model weight name(surfix must be ended by _latest.pt)", default="model_latest.pt")
    parser.add_argument("-batch", "--batch_size", type=int, help="number of batches", default=1)
    parser.add_argument("-epochs", "--epochs", help="number of epochs", default=500)
    parser.add_argument("-dim", "--dimension", type=int, help="Dimension Size: 2(D) or 3(D), Default is 3", default=3)    
    parser.add_argument("-a", "--axis_convert", type=int, help="transpose target axis, ex) 1: transpose(1, 3)", default=0)    
    
    args = parser.parse_args()

    # wandb.config.update(args)
    
    ### argparser config ###
    IS_TEST = args.test if args.test else False
    IS_SHUFFLE = args.shuffle if args.shuffle else False
    BASE = args.input_dir
    MODEL_STORE = args.model_output
    BATCH_SIZE = args.batch_size
    EPOCH_SIZE = args.epochs
    WEIGHT_NAME = args.model_name.replace('_latest', '_best') if os.path.exists(os.path.join(args.model_output, args.model_name.replace('_latest', '_best'))) else args.model_name    
    DIM = args.dimension
    AXIS = args.axis_convert
    ### argparser config ###

    NUM_OF_CASES = 120
    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.2
    PATIENCE = 5 # not use now

    transform = transform_tio()

    ### select model & load weight
    params = {'dimension': DIM, 
              'act_function': 'mish', 
              'num_channels': 1, 
              'num_filters': 32, 
              'kernel_height': 3, 'kernel_width': 3, 
              'stride_c': 1, 'stride_p': 2,
              'kernel_c': 1, 'kernel_p': 2, 
              'output_classes': 5}
    if DIM == 3: params['kernel_depth'] = int(params['kernel_height']) # append kernel depth if DIM==3

    model = fsCNN(params)
    summary(model, (1, 128, 128, 128)) # print summary(params size), 2d/3d applicable (change)

    if os.path.exists(os.path.join(MODEL_STORE, WEIGHT_NAME)):
        prev_weight = torch.load(os.path.join(MODEL_STORE, WEIGHT_NAME))
        model.load_state_dict(prev_weight["model_state_dict"])
        print(f"Previous Weight loaded: '{WEIGHT_NAME}'")

    else: print(f"New Weight: '{WEIGHT_NAME}'")

    subject = torchCustomDataset(BASE, NUM_OF_CASES, transform, IS_TEST, IS_SHUFFLE, AXIS) # IS_TEST == 'False': setting up to train mode(no test)
    earlyStopping = EarlyStopping(PATIENCE, verbose = True, model_name = WEIGHT_NAME.replace('.pt', ''), path = os.path.join(MODEL_STORE, WEIGHT_NAME))

    total_dataset = (len(subject))
    TRAIN_SIZE = int(TRAIN_RATIO * total_dataset)
    VALID_SIZE = total_dataset - TRAIN_SIZE

    print(f'\nInput Data path: {BASE}')
    print(f'Output Weights path: {MODEL_STORE}')
    print(f'\n### Torchio DataLoader_Train: [{TRAIN_SIZE}] ###')
    train_subjects = subject[:TRAIN_SIZE]

    print(f'\n### Torchio DataLoader_Valid: [{VALID_SIZE}] ###')
    valid_subjects = subject[TRAIN_SIZE:]

    train_dataset = tio.SubjectsDataset(train_subjects)
    valid_dataset = tio.SubjectsDataset(valid_subjects)

    TRAIN_LOADER = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=0   # multiprocessing.cpu_count()
    )

    VALID_LOADER = DataLoader(
        valid_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=0   # multiprocessing.cpu_count()
    )   

    optimizer = AdamW(model.parameters(), lr=1e-07) # , weight_decay=1e-05
    scheduler = CosineAnnealingWarmUpRestarts(optimizer,  T_0=50, T_mult=1, eta_max=1e-03,  T_up=10, gamma=0.5) # bag of tricks (AWS) - lr scheduler
    # scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-05) # set lr 1e-01(~ 1e-05)
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True).to(device)  # DiceCE(Cross Entropy) loss monai

    for epochs in range(EPOCH_SIZE):
        if DIM == 3:
            train_loss = train(TRAIN_LOADER, model, optimizer, criterion, epochs)
            valid_loss = eval(VALID_LOADER, model, criterion, epochs)
        else:
            train_loss = train_2d(TRAIN_LOADER, model, optimizer, criterion, epochs)
            valid_loss = eval_2d(VALID_LOADER, model, criterion, epochs)

        scheduler.step()
        earlyStopping(valid_loss, model) # save best or latest

        ## earlystopping method not use now
        # if earlyStopping.early_stop: ### prevent overfitting
        #     print("Kill")
        #     break

        print(f"RESULT: Train_{round(train_loss, 5)} | Valid_{round(valid_loss, 5)}")
    print("Train Finished")