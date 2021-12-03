from flask import Flask, request
from flask_restx import Api, Resource
from markupsafe import escape
from tqdm import tqdm

import pandas as pd
import numpy as np 
import pydicom as pydcm

import os, glob, shutil
import cv2, ast, json

### Flask api ###
app = Flask(__name__)
api = Api(app)

def makedir(dir_path) :
    try :
        if os.path.exists(dir_path):
            print('================ DELETE : PREVIOUS RESULT ================')
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    except OSError:
        print ('Error: Creating directory ' + dir_path)

### USE PYDICOM : Directory
def convert_img(FILE_LOC, DEST_LOC, IS_OKAY=True) :
    makedir(DEST_LOC)

    JSON_CONTENT = []
    FILES_LIST = os.listdir(FILE_LOC)
    CASE_NAME = FILE_LOC.split('/')[-1]

    for FILES in tqdm(FILES_LIST) :
        TARGET_PATH = "{INPUT}/{FNAME}".format(INPUT = FILE_LOC, FNAME = FILES)
        try :
            dicoms = pydcm.read_file(TARGET_PATH)
            INS_NUM = dicoms.InstanceNumber

            img = dicoms.pixel_array - np.min(dicoms.pixel_array)
            if np.amax(img) != 0:
                img = img / np.amax(img)
            img = (img * 255).astype(np.uint8)

            img_name = DEST_LOC + '/' + FILES.replace('.dcm', '.png')
            cv2.imwrite(img_name, img)
            img_info = dict([("Path",TARGET_PATH), ("Instance_Number",INS_NUM)])
            JSON_CONTENT.append(str(img_info))            
            
        except Exception as e :
            print(e)
            IS_OKAY = False

    JSON_CONTENT = ", ".join(JSON_CONTENT)
    total_info = dict([('CASE_{}'.format(CASE_NAME), {"IS_OKAY":IS_OKAY, "RESULT":ast.literal_eval(JSON_CONTENT)})])

    return total_info

### API ###
# example : curl -X POST -F input=[/absolute/file/path] [http://ip.add.ress.num]:[port]/converted
@app.route('/converted', methods=['POST'])
def convert_post():
    pwd = os.getcwd()
    input_path = '{}/temp/input_sample/case01'.format(pwd) if 'input' not in request.form else request.form['input']
    output_path = '{}/temp/output_sample'.format(pwd) if 'output' not in request.form else request.form['output']

    result = convert_img(input_path, output_path)
    result = json.dumps(result, indent='\t')

    # return f"{escape(result)}" # dict 
    return f"{(result)}"

if __name__ == "__main__" :
    # app.run(debug=True, host='[ip.add.ress.num]', port=[port]) # custom port
    app.run(debug=True, host='0.0.0.0', port=33333) # lclhost
