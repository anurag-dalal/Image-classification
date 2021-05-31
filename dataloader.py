'''
Loads the compete data, for high RAM anh good GPU systems
'''
import numpy as np
import cv2
import pandas as pd
import os


DATASET_FOLDER = '../Dataset/celebA/img_align_celeba/img_align_celeba'
SPLIT_FILE = '../Dataset/celebA/list_eval_partition.csv'
KEYPOINT_FILE = '../Dataset/celebA/list_landmarks_align_celeba.csv'

HEIGHT = 218
WIDTH = 178

def load_data():
    split = pd.read_csv(SPLIT_FILE)
    keyp = pd.read_csv(KEYPOINT_FILE)
    
    # scaling
    
    keyp['lefteye_x']= keyp['lefteye_x']/WIDTH
    keyp['righteye_x']= keyp['righteye_x']/WIDTH
    keyp['nose_x']= keyp['nose_x']/WIDTH
    keyp['leftmouth_x']= keyp['leftmouth_x']/WIDTH
    keyp['rightmouth_x']= keyp['rightmouth_x']/WIDTH

    keyp['lefteye_y']= keyp['lefteye_y']/HEIGHT
    keyp['righteye_y']= keyp['righteye_y']/HEIGHT
    keyp['nose_y']= keyp['nose_y']/HEIGHT
    keyp['leftmouth_y']= keyp['leftmouth_y']/HEIGHT
    keyp['rightmouth_y']= keyp['rightmouth_y']/HEIGHT
    
    split = split.drop(['image_id'], axis=1)
    labels = pd.concat([split, keyp], axis=1)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_val = []
    y_val = []
    for ind in labels.index:
        img = cv2.imread(os.path.join(DATASET_FOLDER, labels['image_id'][ind]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img/255.0
        if(ind%1000==0):
            print(ind)
        if(labels['partition'][ind]==0):
            x_train.append(img)
            y_train.append(list(keyp.iloc[ind])[1:])
        elif(labels['partition'][ind]==0):
            x_test.append(img)
            y_test.append(list(keyp.iloc[ind])[1:])
        else:
            x_val.append(img)
            y_val.append(list(keyp.iloc[ind])[1:])
    x_train = np.array(x_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')        
    x_test = np.array(x_test, dtype='float32')        
    y_test = np.array(y_test, dtype='float32')        
    x_val = np.array(x_val, dtype='float32')        
    y_val = np.array(y_val, dtype='float32')        
    
    return x_train, y_train, x_test, y_test, x_val, y_val

