from PIL import Image
import os, glob, natsort
import numpy as np
import matplotlib.pyplot as plt

import time

### path configuration ###
BASE_LOC = '/mnt/nas03/RSNA_2021/Brain_Tumor_Classification/img_converted_new/train' #rsna-miccai-brain-tumor-radiogenomic-classification

files = []
slices = []
# file_dir = []

def align_collect_filepath() :
    patient_ordered = natsort.natsorted(os.listdir(BASE_LOC))
    # print(patient_ordered[:50])
    for patients in patient_ordered :
        # convert dicom files T1wCE type only
        file_dir = natsort.natsorted(glob.glob('{}/{}/T1wCE/*.png'.format(BASE_LOC,patients)))
        numof_files = len(file_dir)

        for filepaths in file_dir :
            img = Image.open(filepaths)
            np_data = np.asarray(img)

            img_shape = list(np_data.shape)
            img_shape.append(numof_files)

            print(img_shape)
            
            # print(np_data)
            # print(np_data.shape)
            # img_size = img.size
            # files.append()
                
            # time.sleep(100)

def combine_info() :
    pass

def convert_axis() :
    pass

def create_new_axis() :
    pass

if __name__ == "__main__" :
    align_collect_filepath()