import os
import sys
import warnings
import argparse

import numpy as np
import torchio as tio

import torch
import multiprocessing

import nibabel as nib
import SimpleITK as sitk

from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter

p = os.path.abspath('..')
sys.path.insert(1, p)

from model import *

#########################################################################################################################
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.autograd.set_detect_anomaly(True)
if torch.cuda.is_available() == True: print(f"GPU [{torch.cuda.get_device_name(0)}]: {torch.cuda.device_count()}")
#########################################################################################################################


def initInfo(nii_path): # space normalization: reference 
    nii_space = tio.ScalarImage(nii_path)

    nii_aff = nib.load(nii_path)
    nii_shape = nii_aff.get_fdata().shape
    nii_aff = nii_aff.affine

    return nii_space, nii_shape, nii_aff


### if rawShape 256,256,256 > 256, ex) 512, 512, 537 > 512
def dataLoader(nii_path, nii_space, axis=0, rawShape=256, reShape=128): # transpose axis(source)
    SUBJECTS = []
    subject = tio.Subject(IMG = tio.ScalarImage(nii_path))

    transform = tio.Compose([tio.CropOrPad((rawShape)),
                             tio.Resize((reShape), label_interpolation='label_gaussian'),
                             tio.ToCanonical(),
                             tio.RescaleIntensity(out_min_max=(0, 1)),
                            ])

    subject_transformed = transform(subject)

    if axis != 0: 
        if axis == 3: subject_transformed.IMG.data = subject_transformed.IMG.data.transpose(1, 2)    
        else: subject_transformed.IMG.data = subject_transformed.IMG.data.transpose(axis, 3) 

    SUBJECTS.append(subject_transformed)
    subject_final = tio.SubjectsDataset(SUBJECTS)

    return subject_final


### append raw Shape(non-resized)
def reshape2Raw(nii_tensor, rawShape, rawSize=256):
    transform_raw = tio.Compose([tio.Resize((rawSize), image_interpolation='label_gaussian'),
                                 tio.CropOrPad(rawShape)
                                #  tio.CropOrPad((int(rawShape[0]), int(rawShape[1]), int(rawShape[2])))
                                ])

    nii_final = transform_raw(nii_tensor.unsqueeze(0).cpu())

    return nii_final.squeeze()


def modelLoader(dim, classes=5):
    params = {'dimension': dim, 'silu': False, 'num_channels': 1, 'num_filters': 32, 'kernel_height': 3, 'kernel_width': 3, 'kernel_depth': 3, 'stride_c': 1, 'kernel_c': 1, 'stride_p': 2, 'kernel_p': 2, 'output_classes': classes}
    
    return fsCNN(params)


def weightLoader(weight_dir, weight_name, axis=0):
    pt_name = f"{weight_name}{axis+1}_best.pt"
    # print(pt_name)

    return torch.load(os.path.join(weight_dir, pt_name))


# for one subject
def eval(dataset, model, nii_aff):
    model.eval()

    with torch.no_grad():
        for batch in dataset:
            eval_img = batch['IMG']['data'].to(device)#, dtype=torch.float)
            eval_output = torch.argmax(model(eval_img).squeeze(), 0).to(dtype=torch.int16) # hard voting
            # eval_output = model(eval_img) # soft voting

    return eval_output


def save_nii(output_dir, result, nii_aff, is_cvt=False, axis_num=0):
    # result = result.cpu().detach().numpy()
    result = result.detach().numpy()
    
    if is_cvt == True:
        if axis_num != 0: result = np.swapaxes(result, axis_num-1, 2)
        print('axis converted') # a simple debugging method ;)

    result_nii = nib.Nifti1Image(result, affine=nii_aff)
    nib.save(result_nii, output_dir)


### bias correction(SimpleITK): compare with nipype's(ANTs package wrapped)
### N2C: bias field correction with reshaped array(tensor > array, correction > tensor. in eval())
def biasCorrection(input_dir):
    input_img = sitk.ReadImage(input_dir)
    mask_img = sitk.OtsuThreshold(input_img, 0, 1, 200)

    input_img = sitk.Cast(input_img, sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_img = corrector.Execute(input_img, mask_img)

    log_bias_field = sitk.Exp(corrector.GetLogBiasFieldAsImage(input_img)) # estimated bias field
    corrected_img = input_img/log_bias_field
                                      
    sitk.WriteImage(log_bias_field, input_dir.replace('.nii.gz', '_bias.nii.gz'))  # bias field
    sitk.WriteImage(corrected_img, input_dir) # corrected img
    print(f'Saved: {input_dir}')


if __name__ == "__main__":
    '''
    AXIS NUMBER(Example)
    {
        0: [NO TRANSPOSE, _3d{1}_best.pt]
        1: [TRANSPOSE 1&3, _3d{2}_best.pt]
        2: [TRANSPOSE 2&3, _3d{3}_best.pt]
    }
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dir", type=str, help="input data path (Absolute)", required=True)        # /{path}/{name.nii.gz}
    parser.add_argument("-o", "--output_dir", type=str, help="input data path (Absolute)", required=True)        # /{path}/{name.nii.gz}
    parser.add_argument("-m", "--model_weight", type=str, help="model weight path (Absolute)", required=True)   # /{path}
    parser.add_argument("-n", "--weight_name", type=str, help="weight name (ex: dataset_3d)", required=True)   # {dataset}_3d(dimension)
    parser.add_argument("-a", "--axis", type=int, help="Transpose Axis, 0: RAW, 1: 1>3, 2: 2>3", default=0)
    parser.add_argument("-b", "--bias", type=bool, help="is input need bias-field correction: True or False", default=False)
    parser.add_argument("-c", "--reshape", type=int, help="size to crop", default=256)
    parser.add_argument("-r", "--resize", type=int, help="size to resizeing", default=128)
    parser.add_argument("-batch", "--batch_size", type=int, help="number of batches", default=1)
    parser.add_argument("-dim", "--dimension", type=int, help="Dimension Size: 2(D) or 3(D), Default is 3", default=3)    

    args = parser.parse_args()

    FLAG = 0
    AXIS_NUM = args.axis
    DATA_DIR = args.input_dir
    RESULT_DIR = args.output_dir
    WEIGHT_DIR = args.model_weight
    WEIGHT_NAME = args.weight_name
    CROP_SIZE = args.reshape
    RE_SIZE = args.resize
    IS_CORR = args.bias
    
    CONCAT_LIST = []

    if IS_CORR == True: biasCorrection(DATA_DIR)

    # initial def: referencecable spatial information(shape, affine & space info)
    if FLAG == 0:
        NII_SPACE, NII_SHAPE, NII_AFF = initInfo(DATA_DIR)
        FLAG = 1

    # model & weight load
    model = modelLoader(args.dimension, 5) # tissue = 5 

    for axis in range(0, 3):
        weight = weightLoader(WEIGHT_DIR, WEIGHT_NAME, axis)

        model.load_state_dict(weight['model_state_dict'])
        print(f"Weight / Loss: {weight['model']}, {weight['model_loss']:.4f}")
        
        # data load
        subjects = dataLoader(DATA_DIR, NII_SPACE, axis, CROP_SIZE, RE_SIZE) # transpose axis(source), if loop, remove axis_num
        
        TEST_LOADER = DataLoader(
            subjects,
            batch_size=args.batch_size,
            num_workers=0 # multiprocessing.cpu.count()
        )

        result = eval(TEST_LOADER, model, NII_AFF)
        if axis != 0: 
            result = torch.transpose(result, 0, 1) if axis == 3 else torch.transpose(result, axis-1, 2)
            # result = torch.transpose(result, axis-1, 2) # hard voting
        # if axis != 0: result = torch.transpose(result, axis+1, 4) # soft voteing

        CONCAT_LIST.append(result.squeeze())

    ### Soft Voting ###
    # concat_result_new = torch.zeros((CONCAT_LIST[0].shape))
    # print(CONCAT_LIST[0].shape)

    # for c in range(CONCAT_LIST[0].size(dim=0)): # 5
    #     for d in tqdm(range(CONCAT_LIST[0].size(dim=1))): # 128
    #         for w in range(CONCAT_LIST[0].size(dim=2)): # 128
    #             for h in range(CONCAT_LIST[0].size(dim=3)): # 128
    #                 sum = 0 # init
    #                 for val in range(0, 3): sum += CONCAT_LIST[val][c][d][w][h]
    #                 concat_result_new[c][d][w][h] = sum / 3                        

    ### print debugging line & check the results from each directions ### 
    # print(concat_result.shape, torch.unique(concat_result))
    # save_nii(RESULT_DIR, concat_result, NII_AFF) # save the result to nii type each

    ### Hard Voting(Counter) ###
    concat_result = torch.stack((CONCAT_LIST[0], CONCAT_LIST[1], CONCAT_LIST[2]),0)
    concat_result_new = torch.zeros((concat_result.size(dim=1), concat_result.size(dim=2), concat_result.size(dim=3))) 

    for d in tqdm(range(concat_result.size(dim=1))): # 128
        for w in range(concat_result.size(dim=2)): # 128
            for h in range(concat_result.size(dim=3)): # 128
                counter = Counter([concat_result[0][d][w][h].item(), concat_result[1][d][w][h].item(), concat_result[2][d][w][h].item()])                
                concat_result_new[d][w][h] = counter.most_common()[0][0] # TUPLE: if len(counter.most_common()) < 3 else counter.most_common()[1][0]

    ### reshape & store ###
    concat_result_new = reshape2Raw(concat_result_new, NII_SHAPE, CROP_SIZE)
    print(concat_result_new.shape, torch.unique(concat_result_new))

    save_nii(RESULT_DIR, concat_result_new, NII_AFF)
