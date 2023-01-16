import os, glob

from utils import makedir, return_path, convert_dcmtk
from tqdm import tqdm

### path configuration
# BASE_LOC = '/home/phenomx/mason/BrainTumor'
# DEST_LOC = '{}/img_converted'.format(BASE_LOC)
# FILE_LOC = '{}/img_sample'.format(BASE_LOC) # nas02 or nas03

### path configuration for NAS ###
BASE_LOC, TRAIN_TEST_DIR = return_path()
BASE_LOC_LOCAL = '/home/phenomx/mason/BrainTumor/img_converted'
IMG_PATH_DIR = {'FLAIR' : [], 'T1w' : [], 'T1wCE' : [], 'T2w' : []} # init directory

def addfileLoc() :
    for dir in TRAIN_TEST_DIR :
        FILE_LOC = '{}{}'.format(BASE_LOC, dir)
        DEST_LOC = '{}{}'.format(BASE_LOC_LOCAL, dir)
        PATIENT_LIST = os.listdir(FILE_LOC)
        for PATIENTS in PATIENT_LIST :
            # print('[{}]'.format(PATIENTS)) # patient number
            for MRI_TYPE in IMG_PATH_DIR.keys() :
                IMG_PATH_DIR[MRI_TYPE] = glob.glob('{}/{}/{}/*.dcm'.format(FILE_LOC, PATIENTS, MRI_TYPE)) # replacement

            for i in IMG_PATH_DIR.keys() :
                # print(i, ' ', len(IMG_PATH_DIR[i]))
                # print(i, ' ', IMG_PATH_DIR[i])
                print(PATIENTS)
                for files in tqdm(IMG_PATH_DIR[i]) :
                    convert_dcmtk(files, DEST_LOC)

if __name__ == "__main__" :
    addfileLoc()
