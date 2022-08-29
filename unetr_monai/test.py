import skimage.transform as skTrans
import nibabel as nib
import numpy as np

import torch

from model import unetRModel, UNETR_MONAI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_data(nii_path):
    nii_raw_affine = nib.load(nii_path)
    nii_raw = nib.load(nii_path).get_fdata()
    nii_shape = nii_raw.shape

    if nii_raw.shape[0] > nii_raw.shape[2]: nii_raw = np.swapaxes(nii_raw, 0, 2)


    nii = skTrans.resize(nii_raw, (128,128,128))
    nii = torch.from_numpy(nii)

    nii_resized = torch.unsqueeze(nii, dim=0)

    return nii_raw_affine, nii_shape, nii_resized


def model_load(model_path):
    # model = unetRModel()
    model = UNETR_MONAI()
    chkpoint = torch.load(model_path)

    return model, chkpoint


def eval(target_nii, model):
    with torch.no_grad():
        model.eval()
        # total_cost = 0.0
    
        eval_data = target_nii.to(device, dtype=torch.float)
        output = model(eval_data)

        return output


if __name__ == "__main__":
    test_path = '/home/pmx/src/test/T1w_0000.nii.gz'
    model_path = '/home/pmx/model/trained/unetr_checkpoint.pt'

    nii_aff, nii_shape, nii= test_data(test_path) # , nii_ext 

    model, chkpoint = model_load(model_path)

    model.load_state_dict(chkpoint['model_state_dict'])

    nii = nii.unsqueeze(0)

    nii_result = eval(nii, model)
    nii_result = nii_result.squeeze() # (C, H W D)
    nii_result = nii_result.cpu().detach().numpy()

    nii_edited = skTrans.resize(nii_result, nii_shape)

    nii_img = nib.Nifti1Image(nii_edited, affine=nii_aff.affine)
    nib.save(nii_img, test_path.replace("_0000", "_Unetr"))

