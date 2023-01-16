import os, gc, natsort, glob, cv2
import numpy as np

from pandas import Series
from utils import makedir, load_csv, return_path
from tqdm import tqdm

BASE_LOC, TRAIN_TEST_DIR = return_path(True)
TARGET_LOC_LOCAL = '/home/phenomx/mason/BrainTumor/converted_DCMTK/npz'

TRAIN_X_DATA = [] # image : 512*512*100 >> 472
TEST_X_DATA = [] # image : 512*512*100 >> 72
TRAIN_Y_DATA = [] # label : 512

# load dataframe
cls_df = load_csv('/home/phenomx/mason/BrainTumor/src/train_labels.csv')

def load_img() :
    for dir in TRAIN_TEST_DIR :
        FILE_LOC = '{}{}'.format(BASE_LOC, dir)
        DEST_LOC = '{}{}'.format(TARGET_LOC_LOCAL, dir)
        PATIENT_LIST = natsort.natsorted(os.listdir(FILE_LOC))

        for patients in PATIENT_LIST :
            IMG_PATH_LIST = natsort.natsorted(glob.glob('{}/{}/*.png'.format(FILE_LOC, patients)))

            Train_X = []
            Test_X = []

            for files in IMG_PATH_LIST :
                info_npz(dir, files, patients, Train_X, Test_X)

            if dir == '/train' :
                cls_deter = cls_df.loc[cls_df['BraTS21ID'] == int(patients), ['MGMT_value']]
                tumor_cls = Series.tolist(cls_deter['MGMT_value'])[0]
        
                TRAIN_Y_DATA.append(tumor_cls) # train_Y
                generate_npz(Train_X, dir)

            else :
                generate_npz(Test_X, dir)

def info_npz(dir, img_path, patient, trainX, testX) :
    img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), dsize=(512, 512), interpolation=cv2.INTER_AREA)

    if dir == '/train' :
        trainX.append(img)
    else :
        testX.append(img)

def generate_npz(np_list, dir) :
    base_np = np.zeros((100,512,512), dtype=np.uint8)
    
    for i in range(100) :
        base_np[i] = np_list[i]

    if dir == '/train' : 
        # pass
        TRAIN_X_DATA.append(base_np)
    else : 
        TEST_X_DATA.append(base_np)    
    
    print('train:', len(TRAIN_X_DATA), 'test:', len(TEST_X_DATA))
    
if __name__ == '__main__' :
    load_img()

    np.savez('{}/train.npz'.format(TARGET_LOC_LOCAL), x_data=TRAIN_X_DATA, y_data=TRAIN_Y_DATA)
    np.savez('{}/test.npz'.format(TARGET_LOC_LOCAL), x_data=TEST_X_DATA)

    # print(len(TRAIN_X_DATA), len(TRAIN_Y_DATA))
    # print(len(TEST_X_DATA))