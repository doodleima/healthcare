import numpy as np
import torchio as tio

import torch
import multiprocessing

from tqdm import tqdm
from torchio import DATA
from torch.utils.data import DataLoader
from model import UNETR_MONAI

import nibabel as nib

import torchinfo
# import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H = 256
W = 256
D = 176


# def test_data(nii_path):
#     nii_raw_affine = nib.load(nii_path)
#     nii_raw = nib.load(nii_path).get_fdata()
#     nii_shape = nii_raw.shape

#     print(nii_raw.shape)
#     if nii_raw.shape[0] < nii_raw.shape[2]: nii_raw = np.swapaxes(nii_raw, 0, 2)
#     print(nii_raw.shape)
    

def make_refer(nii_path): # space normalization: make a reference 
    SPACE_REF = tio.ScalarImage(nii_path)

    return SPACE_REF

def test_data(nii_path, SPACE_REF):
    SUBJECTS = []

    subject = tio.Subject(
            IMG = tio.ScalarImage(nii_path),
        )

    transform = tio.Compose([tio.ToCanonical(), 
                             tio.Resample(SPACE_REF),
                             tio.Resize((H,W,D)),
                             # tio.CropOrPad((D,H,W)),
                             tio.RescaleIntensity(out_min_max=(0, 1)),
                             tio.ZNormalization(masking_method=lambda x: x > x.mean()),
                             tio.CopyAffine('IMG')
                            ])


    subject_transformed = transform(subject)
    if subject_transformed.IMG.data.shape[1] < subject_transformed.IMG.data.shape[3]: 
        subject_transformed.IMG.data = torch.transpose(subject_transformed.IMG.data, 1, 3)
        print(subject_transformed.IMG.data.shape)

    SUBJECTS.append(subject_transformed)

    return SUBJECTS


def model_load(model_path):
    model = UNETR_MONAI()
    chkpoint = torch.load(model_path)

    return model, chkpoint


def eval(dataloader, model, nii_aff):
    model.eval()
    with torch.no_grad():    
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='[TorchIO] TEST')):
            #nib.save(nib.Nifti1Image(batch['IMG'][DATA].numpy(), affine=nii_aff), '/home/pmx/src/test/T1w_Origin.nii.gz')# nifti save
            test_data = batch['IMG'][DATA].to(device, dtype=torch.float)
            output_test = model(test_data)
            
    return output_test


def load_affine(nii_path):
    nii_aff = nib.load(nii_path)

    return nii_aff.get_fdata().shape, nii_aff.affine


if __name__ == "__main__":
    BATCH_SIZE = 1
    FLAG = 0

    test_path = 'absolute/path/filename.nii.gz'
    model_path = 'absolute/path/model_chkpoint.pt'

    if FLAG == 0:
        SPACE_REF = make_refer(test_path)
        nii_shape, nii_aff = load_affine(test_path)
        FLAG = 1

    subject = test_data(test_path, SPACE_REF) # , nii_ext 
    model, weight = model_load(model_path)

    model.load_state_dict(weight['model_state_dict'])

    # print(torchinfo.summary(model, input_size=(1,1,H,W,D))) # summary
    test_dataset = tio.SubjectsDataset(subject)

    TEST_LOADER = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=multiprocessing.cpu_count(), # import multiprocessing
    )

    nii_output = eval(TEST_LOADER, model, nii_aff) # tensor (B, C, 128, 128, 128)

    # nii_output = F.softmax(nii_output, 1)
    nii_output = torch.argmax(nii_output, 1)
    nii_output = nii_output.to(dtype=torch.int8) # type cast

    # if nii_output.shape[1] > nii_output.shape[3]: 
    #     nii_output.data = torch.transpose(nii_output.data, 1, 3)

    transform_raw = tio.Resize(target_shape=nii_shape, image_interpolation='nearest')
    nii_transformed = transform_raw(nii_output.cpu())
    nii_transformed = nii_transformed.squeeze() # (C, n, n, n)

    # nii_output = nii_output.squeeze() # (C, n, n, n)
    # nii_output = nii_output.cpu().detach().numpy()
    nii_output = nii_transformed.detach().numpy()

    nii_img = nib.Nifti1Image(nii_output, affine=nii_aff) # nii_aff
    nib.save(nii_img, test_path.replace("_0000", "_Unetr"))
