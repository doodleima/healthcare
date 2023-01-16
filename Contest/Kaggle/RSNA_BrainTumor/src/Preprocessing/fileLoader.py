import os, glob, natsort
from utils import makedir, move_files, return_path

### path configuration for NAS ###
BASE_LOC, TRAIN_TEST_DIR = return_path()
BASE_LOC = '/'.join(BASE_LOC.split('/')[:-1])
TARGET_LOCAL = '/home/phenomx/mason/BrainTumor/converted_DCMTK'
IMG_PATH_DIR = {'T1wCE': []}
DCMTK_CONVERTED = '/img_converted_new'

# 1) pick the patients which have over 100 images
def pick_patients() :
    for dir in TRAIN_TEST_DIR :
        FILE_LOC = '{}{}{}'.format(BASE_LOC, DCMTK_CONVERTED, dir)
        DEST_LOC = '{}{}'.format(TARGET_LOCAL, dir)
        PATIENT_LIST = os.listdir(FILE_LOC)
        # print(PATIENT_LIST)
        for PATIENTS in PATIENT_LIST : 
            # print('{}/{}/T1wCE/*.png'.format(FILE_LOC, PATIENTS))
            IMG_PATH_DIR['T1wCE'] = natsort.natsorted(glob.glob('{}/{}/T1wCE/*.png'.format(FILE_LOC, PATIENTS)))

            if len(IMG_PATH_DIR['T1wCE']) >= 100 : pick_center_files(IMG_PATH_DIR['T1wCE'], DEST_LOC)

# 2) pick img name which located in the center part from patients
def pick_center_files(PATH_LIST, DEST_LOC) :
    FILE_START = len(PATH_LIST)//2 - 50
    FILE_END = len(PATH_LIST)//2 + 50

    convert_target_list = PATH_LIST[FILE_START:FILE_END]
    PATH_TARGET = ('{}/{}').format(DEST_LOC, convert_target_list[0].split('/')[-3]) # Target path to Paste files
    move_files(convert_target_list, PATH_TARGET)

if __name__ == "__main__" :
    pick_patients()