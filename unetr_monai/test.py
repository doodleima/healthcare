import numpy as np
import torchio as tio

import torch
import multiprocessing

from tqdm import tqdm
from torchio import DATA
from torch.utils.data import DataLoader
from model import unetRModel, UNETR_MONAI

import nibabel as nib

import torchinfo
# import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H = 256
W = 256
D = 176


def make_refer(nii_path): # space normalization: make a reference 
    SPACE_REF = tio.ScalarImage(nii_path)

    return SPACE_REF

def test_data(nii_path, SPACE_REF):
    SUBJECTS = []
    # AFFINE = []

    subject = tio.Subject(
            IMG = tio.ScalarImage(nii_path)
        )

    transform = tio.Compose([tio.ToCanonical(), 
                                tio.Resample(SPACE_REF),
                                tio.CropOrPad((256,256,176)),
                            #  tio.Resize((208,208,208), image_interpolation='linear', label_interpolation='nearest'), # 128, 128, 128
                                tio.RescaleIntensity(out_min_max=(0, 1)),
                                tio.ZNormalization(masking_method=lambda x: x > x.mean()),
                                tio.CopyAffine('IMG')
                            ])

    subject_transformed = transform(subject)
    # AFFINE.append(subject_transformed.shape)
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

    return nii_aff.affine


if __name__ == "__main__":
    BATCH_SIZE = 1
    FLAG = 0

    test_path = '/home/pmx/src/test/T1w_0000.nii.gz'
    model_path = '/home/pmx/model/trained/unetr_chkpoint.pt'

    if FLAG == 0:
        SPACE_REF = make_refer(test_path)
        nii_aff = load_affine(test_path)
        FLAG = 1

    subject = test_data(test_path, SPACE_REF) # , nii_ext 
    model, weight = model_load(model_path)

    model.load_state_dict(weight['model_state_dict'])

    print(torchinfo.summary(model, input_size=(1,1,H,W,D))) # summary
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
    nii_output = nii_output.squeeze() # (C, n, n, n)
    
    print(torch.unique(nii_output))
    print(nii_output.shape)

    nii_output = nii_output.cpu().detach().numpy()

    nii_img = nib.Nifti1Image(nii_output, affine=nii_aff) # nii_aff
    nib.save(nii_img, test_path.replace("_0000", "_Unetr"))
