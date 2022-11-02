import numpy as np
import torchio as tio

import torch
import multiprocessing

from tqdm import tqdm
from torchio import DATA
from torch.utils.data import DataLoader
from model import UNETR_MONAI, SwinUNETR_MONAI

import nibabel as nib

import torchinfo
# import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
PATCH_SIZE = 64
QUEUE_LEN = 500
VOL_SAMPLES = 16


def make_refer(nii_path): # space normalization: make a reference 
    SPACE_REF = tio.ScalarImage(nii_path)

    return SPACE_REF

def test_data(nii_path, SPACE_REF):
    SUBJECTS = []

    subject = tio.Subject(
            IMG = tio.ScalarImage(nii_path),
        )

    # affine_copied = tio.CopyAffine('IMG')

    transform = tio.Compose([tio.ToCanonical(),
                             tio.RescaleIntensity(out_min_max=(0, 1)),
                             tio.EnsureShapeMultiple(16),
                             tio.RandomFlip(axes=('LRAPIS'), flip_probability=0.5),
                             tio.RandomBiasField(coefficients=(-0.5, 0.5), order=3),
                             tio.RandomGamma(log_gamma=(-0.2, 0.2)),
                             tio.CropOrPad((256,256,256), mask_name='LABEL'),
                            ])


    subject_transformed = transform(subject)
    if subject_transformed.IMG.data.shape[1] < subject_transformed.IMG.data.shape[3]: 
        subject_transformed.IMG.data = torch.transpose(subject_transformed.IMG.data, 1, 3)
        print(subject_transformed.IMG.data.shape)

    SUBJECTS.append(subject_transformed)

    return SUBJECTS


def model_load(model_path):
    model = SwinUNETR_MONAI()
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
    BATCH_SIZE = 4
    FLAG = 0

    test_path = '/home/pmx/src/test/T1w_0000.nii.gz'
    model_path = '/home/pmx/model/trained/unetr_chkpoint_latest.pt'

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

    print(nii_output.shape)
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
