import os
import sys
import warnings
import argparse
import multiprocessing
import torch
import torchinfo # torchsummary

import torchio as tio
import numpy as np

from tqdm import tqdm
from torchio import DATA
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

p = os.path.abspath('..')
sys.path.insert(1, p)

from model import UNETR_MONAI, SwinUNETR_MONAI
from dataloader_torchio import UNetR_CustomDataset, EarlyStopping

import gc
gc.collect()
torch.cuda.empty_cache()

import wandb
wandb.init(project="TorchIO_swinUNETR_160")

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
                5: "Head Fat"}
#########################################################################################################################

### model train
def train(dataloader, model, optimizer, loss_function, epoch):
    model.train()
    train_loss = 0.0
    loss_div = len(dataloader)
    train_samples={}

    input_train_patch, input_label_train_patch, output_train_patch = None, None, None

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

        # patch sample (latest one): for wandb
        input_train_patch = train_data.cpu()
        input_label_train_patch = train_label.cpu()
        output_train_patch = output_train.cpu()

    train_label_patch = torch.argmax(output_train_patch[0], 0)
    train_label_gt_patch = torch.argmax(input_label_train_patch[0], 0)

    train_image_patch = input_train_patch.squeeze().squeeze()

    midval = int(len(train_image_patch)/2 - 1)

    train_samples["prediction"] = train_label_patch[midval].detach().numpy()
    train_samples["input"] = train_image_patch[midval].detach().numpy()
    train_samples["gt"] = train_label_gt_patch[midval].detach().numpy()

    train_loss = train_loss / len(dataloader)
    wandb.log({"Train_Loss": train_loss, 
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
            # valid_label = batch['LABEL'][DATA].to("cpu", dtype=torch.int8)
            v_loss = loss_function(output_val, valid_label)

            val_loss += v_loss.item()

            # patch sample (latest one): for wandb
            input_val_patch = valid_data.cpu()
            input_label_val_patch = valid_label.cpu()
            output_val_patch = output_val.cpu()

    valid_label_patch = torch.argmax(output_val_patch[0], 0)
    valid_label_gt_patch = torch.argmax(input_label_val_patch[0], 0)

    valid_image_patch = input_val_patch.squeeze().squeeze()

    midval = int(len(valid_image_patch)/2 - 1)

    val_samples["prediction"] = valid_label_patch[midval].detach().numpy()
    val_samples["input"] = valid_image_patch[midval].detach().numpy()
    val_samples["gt"] = valid_label_gt_patch[midval].detach().numpy()
    
    val_loss = val_loss / len(dataloader)
    wandb.log({"Valid_Loss": val_loss, "epoch": epoch, 
               "Valid_Patches": wandb.Image(val_samples["input"], masks = {"ground_truth": {"mask_data": val_samples["gt"], "class_labels": class_labels},
                                                                           "predictions": {"mask_data": val_samples["prediction"], "class_labels": class_labels}})})

    return val_loss


def transform_methods():
    transform = tio.Compose([#  tio.ToCanonical(),
                             tio.RescaleIntensity(out_min_max=(0, 1)),
                             tio.RandomFlip(axes=('LRAPIS'), flip_probability=0.25),
                             tio.CropOrPad((256,256,256), mask_name='LABEL'),
                            #  tio.EnsureShapeMultiple(16),
                            #  tio.RandomBiasField(coefficients=(-0.5, 0.5), order=3),
                            #  tio.RandomGamma(log_gamma=(-0.2, 0.2)),
                            #  tio.RandomAnisotropy(axes=(0,1,2), downsampling=(1.5,5), image_interpolation='linear'),
                            #  tio.RandomNoise(mean=0, std=(0, 0.25)),
                            #  tio.RandomGhosting(num_gohsts=(1,4), axes=(0,1,2), intensity=(0.2, 1), restore=0.02),
                            #  tio.ZNormalization(masking_method=lambda x: x > x.mean()),
                            ])

    return transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--test", help="Test pipeline with 10 cases", action="store_true")
    parser.add_argument("-s", "--shuffle", help="Dataloader: shuffle", action="store_true")
    parser.add_argument("-i", "--input_dir", type=str, help="input data path", default="absolute/input/path")
    parser.add_argument("-o", "--model_output", type=str, help="model weight path", default="absolute/input/path")
    parser.add_argument("-n", "--model_name", type=str, help="model weight name", default="swinunetr_latest.pt")
    parser.add_argument("-batch", "--batch_size", help="number of batches", default=1)
    parser.add_argument("-epochs", "--epochs", help="number of epochs", default=500)
    
    args = parser.parse_args()
    wandb.config.update(args)

    ### if best >> change
    
    IS_TEST = args.test if args.test else False
    IS_SHUFFLE = args.shuffle if args.shuffle else False
    BASE = args.input_dir
    MODEL_STORE = args.model_output
    BATCH_SIZE = args.batch_size
    EPOCH_SIZE = args.epochs
    WEIGHT_NAME = args.model_name.replace('_latest', '_best') if os.path.exists(os.path.join(args.model_output, args.model_name.replace('_latest', '_best'))) else args.model_name    
    
    ### 100 * 3
    NUM_OF_CASES = 90 # cases used for training 
    PATCH_SIZE = 160   # (equals with H, W, D)
    TRAIN_RATIO = 0.85
    VALID_RATIO = 0.15
    PATIENCE = 5
    QUEUE_LEN = 200
    VOL_SAMPLES = 2

    transform = transform_methods()

    ### select model & load weight
    model = SwinUNETR_MONAI() # UNETR_MONAI()
    
    if os.path.exists(os.path.join(MODEL_STORE, WEIGHT_NAME)):
        prev_weight = torch.load(os.path.join(MODEL_STORE, WEIGHT_NAME))
        model.load_state_dict(prev_weight["model_state_dict"])
        print(f"Previous Weight loaded: '{WEIGHT_NAME}'")

    else: print(f"New Weight: '{WEIGHT_NAME}'")

    subject = UNetR_CustomDataset(BASE, NUM_OF_CASES, transform, IS_TEST, IS_SHUFFLE) # IS_TEST == 'False': setting up to train mode(no test)
    earlyStopping = EarlyStopping(PATIENCE, verbose = True, path = os.path.join(MODEL_STORE, WEIGHT_NAME))

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

    sampler = tio.data.UniformSampler(PATCH_SIZE) # generate random space patch ((patch_size, patch_size, patch_size))

    # patch-wised
    train_patches = tio.Queue(train_dataset, QUEUE_LEN, VOL_SAMPLES, sampler, num_workers=0, shuffle_subjects=True, shuffle_patches=True)
    valid_patches = tio.Queue(valid_dataset, QUEUE_LEN, VOL_SAMPLES, sampler, num_workers=0, shuffle_subjects=True, shuffle_patches=True)

    TRAIN_LOADER = DataLoader(
        # train_dataset,
        train_patches,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=0   # multiprocessing.cpu_count()
    )

    VALID_LOADER = DataLoader(
        # valid_dataset,
        valid_patches,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=0   # multiprocessing.cpu_count()
    )

    # try: 
    #     print(torchinfo.summary(model, input_size=(BATCH_SIZE,1,PATCH_SIZE,PATCH_SIZE,PATCH_SIZE))) # summary - torchinfo
    # except: print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-06)
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True).to(device)  # DiceCE(Cross Entropy) loss monai

    for epochs in range(EPOCH_SIZE):
        train_loss = train(TRAIN_LOADER, model, optimizer, criterion, epochs)
        valid_loss = eval(VALID_LOADER, model, criterion, epochs)
        earlyStopping(valid_loss, model) # save best or latest

        ### earlystopping method
        # if earlyStopping.early_stop or valid_loss < 0: ### prevent overfitting
        #     print("Activate: Early Stopping")
        #     break

        print(f"RESULT: Train_{round(train_loss, 4)} | Valid_{round(valid_loss, 4)}")
    print("Train Finished")
