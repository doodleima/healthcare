import os
import sys
import numpy as np
import torchio as tio

import torch
import multiprocessing

from tqdm import tqdm
from torchio import DATA
from torch.utils.data import DataLoader

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
p = os.path.abspath('..')
sys.path.insert(1, p)

from model import UNETR_MONAI, SwinUNETR_MONAI

import nibabel as nib
import torchinfo
# import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
PATCH_SIZE = 192 #160

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
                             tio.CropOrPad((256,256,256)),
                            #  tio.EnsureShapeMultiple(16),
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


def eval(dataloader, aggregator, model, nii_aff):
    model.eval()
    with torch.no_grad():    
        for batch in tqdm(dataloader, desc='[TorchIO] TEST'):
            #nib.save(nib.Nifti1Image(batch['IMG'][DATA].numpy(), affine=nii_aff), '/home/pmx/src/test/T1w_Origin.nii.gz')# nifti save
            test_data = batch['IMG'][DATA].to(device, dtype=torch.float)
            locations = batch[tio.LOCATION]
            output_test = model(test_data)

            aggregator.add_batch(output_test, locations)

    # return output_test


def load_affine(nii_path):
    nii_aff = nib.load(nii_path)

    return nii_aff.get_fdata().shape, nii_aff.affine


if __name__ == "__main__":
    BATCH_SIZE = 4
    FLAG = 0

    test_path = '/absolute/path/sample.nii.gz'
    model_path = '/absolute/path/swinunetr_best.pt' # patches

    if FLAG == 0:
        SPACE_REF = make_refer(test_path)
        nii_shape, nii_aff = load_affine(test_path)
        FLAG = 1

    subjects = test_data(test_path, SPACE_REF) # , nii_ext 
    model, weight = model_load(model_path)

    model.load_state_dict(weight['model_state_dict'])
    print(weight['model'], weight['model_loss'])

    # print(torchinfo.summary(model, input_size=(1,1,H,W,D))) # summary
    test_dataset = tio.SubjectsDataset(subjects)
    # print(dir(test_dataset))
    
    for idx in range(len(test_dataset)):
        _subject = test_dataset[idx]
        sampler = tio.GridSampler(subject=_subject, patch_size=PATCH_SIZE)

        TEST_LOADER = DataLoader(
            sampler,
            batch_size=BATCH_SIZE,
            num_workers=0 # multiprocessing.cpu_count(), # import multiprocessing
        )
    
        aggregator = tio.inference.GridAggregator(sampler)

        eval(TEST_LOADER, aggregator, model, nii_aff) # tensor (B, C, 128, 128, 128)
        nii_output = aggregator.get_output_tensor() # (C, n, n, n)
        nii_output = torch.argmax(nii_output, 0)
        # nii_output = nii_output.to(dtype=torch.int8) # type cast
 
        # if nii_output.shape[0] > nii_output.shape[2]: 
            # nii_output.data = torch.transpose(nii_output.data, 0, 2)

        # nii_output = nii_output.unsqueeze(0) # (C, n, n, n)

        # transform_raw = tio.CropOrPad((nii_shape)) # if croporpad
        # nii_transformed = transform_raw(nii_output.cpu())
        # nii_transformed = torch.argmax(nii_transformed, 0)
        # nii_transformed = nii_transformed.squeeze() # (C, n, n, n)

        # nii_output = nii_transformed.detach().numpy()        
        nii_output = nii_output.detach().numpy()        
        print(nii_output.shape, np.unique(nii_output))

        nii_img = nib.Nifti1Image(nii_output, affine=np.eye(4))#nii_aff) # nii_aff
        nib.save(nii_img, test_path.replace("sample", "result_swinunetr"))
        print(f'file saved: {test_path.replace("sample", "result_swinunetr")}')
