import numpy as np
import torchio as tio

import torch
import multiprocessing

from tqdm import tqdm
from torchio import DATA
from torch.utils.data import DataLoader
from model import unetRModel, UNETR_MONAI

import nibabel as nib

# import torchinfo
# import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_refer(nii_path): # for space normalization(resample)
    nii_ref = tio.ScalarImage(nii_path)

    return nii_ref

def test_data(nii_path, space_ref):
    SUBJECTS = []

    subject = tio.Subject(
            SPACE_REF = space_ref,
            IMG = tio.ScalarImage(nii_path)
        )

    transform = tio.Compose([tio.ToCanonical(), 
                                 tio.Resample('SPACE_REF'),
                                 tio.Resize((128,128,128), image_interpolation='linear'),
                                #  tio.CropOrPad((128,128,128)),
                            ])

    subject_transformed = transform(subject)
    SUBJECTS.append(subject_transformed)

    return SUBJECTS


def model_load(model_path):
    model = UNETR_MONAI()
    chkpoint = torch.load(model_path)

    return model, chkpoint


def eval(dataloader, model):
    model.eval()
    with torch.no_grad():    
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Torchio TEST')):
            test_data =batch['IMG'][DATA].to(device, dtype=torch.float)
            
            output_test = model(test_data)

    return output_test


def load_affine(nii_path):
    nii_aff = nib.load(nii_path)

    return nii_aff.affine


if __name__ == "__main__":
    BATCH_SIZE = 4
    FLAG = 0

    test_path = 'ABSOLUTE PATH(Nii.gz)'
    model_path = 'ABSOLUTE PATH(checkpoint.pt)'

    if FLAG == 0:
        nii_ref = make_refer(test_path)
        nii_aff = load_affine(test_path)
        FLAG = 1

    subject = test_data(test_path, nii_ref) # , nii_ext 
    model, weight = model_load(model_path)

    model.load_state_dict(weight['model_state_dict'])

    # print(torchinfo.summary(model, input_size=(4,1,128,128,128))) # model summary
    test_dataset = tio.SubjectsDataset(subject)

    TEST_LOADER = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=multiprocessing.cpu_count(), # import multiprocessing
    )

    nii_output = eval(TEST_LOADER, model) # tensor (B, C, 128, 128, 128)
    
    # nii_output = F.softmax(nii_output, 1)
    nii_output = torch.argmax(nii_output, 1)
    nii_output = nii_output.to(dtype=torch.int8) # type casting
    nii_output = nii_output.squeeze() # (C, 128, 128 ,128)
    
    print(torch.unique(nii_output))
    print(nii_output.shape)

    nii_output = nii_output.cpu().detach().numpy()

    # # print(np.unique(nii_output))
    # # print(nii_output.shape)

    nii_img = nib.Nifti1Image(nii_output, affine=np.eye(4)) # or 'affine=nii_aff'
    nib.save(nii_img, test_path.replace('REMOVE', 'NEW'))
