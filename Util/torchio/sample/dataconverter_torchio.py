import os

import torchio as tio
import numpy as np
import nibabel as nib

from glob import glob
from tqdm import tqdm


def getPathOrPid(base, target=None):
    if target is not None: 
        return os.path.join(base, target) 

    else:
        return base.split('/')[-1] # extract sub-XXX.nii.gz (type: str)
        

def openNifti(nii_path):
    nii_raw = nib.load(nii_path)
    nii_data = nii_raw.get_fdata()
    nii_aff = nii_raw.affine

    return nii_data, nii_aff


def makeLabel(img_list):
    for img in tqdm(img_list):
        subname = img.split('/')[-1] # extract sub-XXX.nii.gz (type: str)
        
        raw_label_path = getPathOrPid(BASE+'/labelsPREV', subname)
        fat_label_path = getPathOrPid(BASE+'/labelsPREV_TARGET_TO_CONCAT', subname)

        arr_raw, aff_raw = openNifti(raw_label_path)
        arr_fat, aff_fat = openNifti(fat_label_path)

        y4, x4, z4 = np.where((arr_fat == 2))
        arr_raw[y4, x4, z4] = 4 # store to labelsRAW

        store_path = getPathOrPid(BASE+'/labelsRAW', subname)
        nib.save(nib.Nifti1Image(arr_raw, affine=aff_raw), store_path)


def restore_tio(subname, transform_tio):
    HOSPITAL = "HOSPITAL"
    IMAGE_PATH = subname
    LABEL_PATH = subname.replace('imagesRAW', 'labelsRAW')
    sub = getPathOrPid(subname)
    PID = sub.replace('.nii.gz', '') # or use re
    
    # print(PID, IMAGE_PATH, LABEL_PATH) simple debug method

    subject = tio.Subject(
        IMG = tio.ScalarImage(IMAGE_PATH),
        LABEL = tio.LabelMap(LABEL_PATH)
    )

    subject_transformed = transform_tio(subject) # apply transform to the subject files

    ### torchIO subject = transformation > insert PID / Hospital > save
    # subject_transformed['PID'] = PID
    # subject_transformed['HOSPITAL'] = HOSPITAL
    # print(subject_transformed)
    
    # affine: use subject_transformed.IMG.affine or none > nib.save()
    # print(IMAGE_PATH.replace('imagesRAW', 'imagesNEW'), LABEL_PATH.replace('labelsRAW', 'labelsNEW'))

    # torch to numpy(+ squeeze), non-affine transformation
    nib.save(nib.Nifti1Image(subject_transformed.IMG.data.numpy().squeeze(), None), IMAGE_PATH.replace('imagesRAW', 'imagesNEW'))
    nib.save(nib.Nifti1Image(subject_transformed.LABEL.data.numpy().squeeze(), None), LABEL_PATH.replace('labelsRAW', 'labelsNEW'))


if __name__ == "__main__":
    BASE = '/home/pmx/src/pytorch/data/kku'
    # SUBJECTS = []

    path_raw = getPathOrPid(BASE, 'imagesRAW')
    image_list = sorted(glob('{}/*.nii.gz'.format(path_raw)))

    if not os.path.exists(image_list[0].replace('imagesRAW', 'labelsRAW')):
        print('Generate a New label dataset')
        makeLabel(image_list) # generate labelset if not exist

    # necessary methods: CropOrPad excluded 
    transform = tio.Compose([
                             tio.ToCanonical(),
                             tio.RescaleIntensity(out_min_max=(0, 1)),
                            ])

    for subpath in tqdm(image_list):
        subject = restore_tio(subpath, transform)
    

