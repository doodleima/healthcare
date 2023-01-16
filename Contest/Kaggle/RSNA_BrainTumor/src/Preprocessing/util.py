import os
import pandas as pd 

from tqdm import tqdm

def makedir(dir_path) :
    try :
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except OSError:
        print ('Error: Creating directory ' +  dir_path)

def load_csv(csv_path) :
    df = pd.read_csv(csv_path)
    return df

def move_files(FILE_LIST, PATH_TARGET) : 
    makedir(PATH_TARGET)
    for files in tqdm(FILE_LIST) : 
        # FILE_NAME = files.split('/')[-1]
        move_cmd = 'sudo cp {} {}'.format(files, PATH_TARGET)
        os.system(move_cmd)

def return_path(is_npz=False) :
    BASE_LOC = '/mnt/nas03/RSNA_2021/Brain_Tumor_Classification/rsna-miccai-brain-tumor-radiogenomic-classification'
    
    if is_npz == True :
        BASE_LOC = '/home/phenomx/mason/BrainTumor/converted_DCMTK'
    
    TRAIN_TEST_DIR = {'/train', '/test'}

    return BASE_LOC, TRAIN_TEST_DIR

def convert_dcmtk(FILE_LOC, DEST_LOC) :
    SPLIT_FILEPATH = FILE_LOC.split('/')
    PATIENT_NUM = SPLIT_FILEPATH[-3]
    PATIENT_MRI_TYPE = SPLIT_FILEPATH[-2]
    PATIENT_FILE_NAME = SPLIT_FILEPATH[-1]

    FULL_DEST_PATH = '{}/{}/{}'.format(DEST_LOC, PATIENT_NUM, PATIENT_MRI_TYPE)
    makedir(FULL_DEST_PATH)

    # log level 1(print message if fatal prob occured)
    convert_cmd = 'sudo dcmj2pnm -q {TARGET_PATH} {DEST_PATH}/{IMG_NAME} +Wm +Sxv 512 +Syv 512'.format(
                    TARGET_PATH = FILE_LOC, # in TOTAL_FILES_LIST
                    DEST_PATH = FULL_DEST_PATH,
                    IMG_NAME = PATIENT_FILE_NAME.replace('.dcm','.png'))
    # print(convert_cmd)
    os.system(convert_cmd)