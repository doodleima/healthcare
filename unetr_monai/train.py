import os
import warnings
import torch

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms 
from torch.utils.data import DataLoader, random_split

from model import unetModel, unetRModel, UNETR_MONAI
from dataloader import nnUNet_CustomDataset


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.autograd.set_detect_anomaly(True)

## gimmick ^_^
if torch.cuda.is_available() == True: 
    print(f"GPU [{torch.cuda.get_device_name(0)}]: {torch.cuda.device_count()}")
                                                    

### dice loss function
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):        
        inputs = F.sigmoid(inputs) 
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice 


### save chkpoint weight if some conditions are 'True'
def save_chkpoint(epoch_num, model, optimizer, cost, model_path):
    print("EPOCH [{}], MODEL SAVE".format(epochs+1))
    torch.save({"model": "UNetR3d",
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cost": cost,
                "description": "checkpoint_{}".format(epoch_num),}, os.path.join(model_path, 'unetr_checkpoint.pt'))
# else: pass


### model train
def train(dataloader, model, optimizer, loss_function):
    model.train()
    train_loss = 0.0

    ### train model
    label_values = []
    for train_data, train_label in tqdm(dataloader, desc='TRAIN'):
        for vals in torch.unique(train_label).numpy():
            if vals not in label_values: label_values.append(vals)

        # print(label_values)
        # print(train_label.size())

        train_data, train_label = train_data.to(device, dtype=torch.float), train_label.to(device, dtype=torch.int8)

        output_train = model(train_data)
        t_loss = loss_function(output_train, train_label)

        # backpropagation
        optimizer.zero_grad()
        t_loss.sum().backward()
        optimizer.step()

        train_loss += t_loss.item()

    train_loss = train_loss / len(dataloader)
    # print(f"Train Loss : {train_loss}")

    return label_values, train_loss


### model eval
def eval(dataloader, model, loss_function):
    label_values = []
    with torch.no_grad():
        model.eval()
        val_loss = 0.0

        for valid_data, valid_label in tqdm(dataloader, desc='VALID'):
            for vals in torch.unique(valid_label).numpy():
                if vals not in label_values: label_values.append(vals)
            valid_data, valid_label = valid_data.to(device, dtype=torch.float), valid_label.to(device, dtype=torch.int8)
            
            output_val = model(valid_data)
            v_loss = loss_function(output_val, valid_label)

            val_loss += v_loss.item()
        
    val_loss = val_loss / len(dataloader)
    # print(f"Valid Loss : {val_loss}\n")

    return label_values, val_loss


if __name__ == "__main__":
    TARGET = '101_wholebrain' # or 102_tissue >> use argparse to select target dir 
    # TARGET = '100_original' # or 102_tissue >> use argparse to select target dir 
    BASE = '/home/pmx/data/adni/nnUNet_raw_data_base/nnUNet_raw_data/Task{}'.format(TARGET)
    MODEL_STORE = '/home/pmx/model/trained' # model store path

    train_data = nnUNet_CustomDataset(
        base_path = BASE,
        mode = 'train',
        transform = transforms.ToTensor(),
        target_transform = transforms.ToTensor()
    )

    ## split dataset to train / valid / test
    DATA_SIZE = len(train_data)
    TRAIN_SIZE = int(DATA_SIZE * 0.07) # 0.7
    VALID_SIZE = int(DATA_SIZE * 0.02) # 0.2
    TEST_SIZE =  int(DATA_SIZE * 0.01) # 0.1
    REST_SIZE = int(DATA_SIZE * 0.9)
    BATCH_SIZE = 4
    EPOCH_SIZE = 100

    train_set, valid_set, test_set, temp_set = random_split(train_data, [TRAIN_SIZE, VALID_SIZE, TEST_SIZE, REST_SIZE])
    # for i in [train_set, valid_set, test_set]: print(len(i))

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)
    # test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True) # for train

    # model = unetModel() # unet unofficial
    # model = unetRModel() # unetr unofficial
    model = UNETR_MONAI() # unetr official(monai)

    criterion = DiceLoss().to(device) # dice loss
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-04, eps = 1e-08) # learning rate 0.00001
    TOTAL_TRAIN, TOTAL_VALID = int(TRAIN_SIZE/BATCH_SIZE), int(VALID_SIZE/BATCH_SIZE)

    print(f"DATASET SIZE [TRAIN]: {TOTAL_TRAIN}")
    print(f"DATASET SIZE [VALID]: {TOTAL_VALID}")

    for epochs in range(EPOCH_SIZE):
        print(f"\nEPOCH [{epochs+1}]")
        train_labels, train_loss = train(train_dataloader, model, optimizer, criterion)
        if (epochs+1) % 5 == 0: save_chkpoint(epochs, model, optimizer, train_loss, MODEL_STORE)
        valid_labels, valid_loss = eval(valid_dataloader, model, criterion)
        
        print(f"LABEL: Train_{train_labels} | Valid_{valid_labels}")
        print(f"RESULT: Train_{round(train_loss, 4)} | Valid_{round(valid_loss, 4)}")
