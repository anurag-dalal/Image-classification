import numpy as np
import cv2
import pandas as pd
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class data_load_generator():
    def __init__(self):
        self.DATASET_FOLDER = '../Dataset/celebA/img_align_celeba/img_align_celeba'
        self.SPLIT_FILE = '../Dataset/celebA/list_eval_partition.csv'
        self.KEYPOINT_FILE = '../Dataset/celebA/list_landmarks_align_celeba.csv'
        self.ATTRIBUTE_FILE = '../Dataset/celebA/list_attr_celeba.csv'
        
        self.HEIGHT = 218
        self.WIDTH = 178
        self.NUM_OUTPUTS = 10
        self.input_shape = (self.HEIGHT, self.WIDTH, 1)
        
        
        self.split = pd.read_csv(self.SPLIT_FILE)
        self.keyp = pd.read_csv(self.KEYPOINT_FILE)
        self.split = self.split.drop(['image_id'], axis=1)
        self.attr = pd.read_csv(self.ATTRIBUTE_FILE)
        self.labels = pd.concat([self.split, self.keyp], axis=1)
         
        self.train = self.labels[self.labels['partition'] == 0]
        self.val = self.labels[self.labels['partition'] == 1]
        self.test = self.labels[self.labels['partition'] == 2]
        
        self.fileList = list(self.train['image_id'])
        
        # scaling

        self.keyp['lefteye_x']= self.keyp['lefteye_x']/self.WIDTH
        self.keyp['righteye_x']= self.keyp['righteye_x']/self.WIDTH
        self.keyp['nose_x']= self.keyp['nose_x']/self.WIDTH
        self.keyp['leftmouth_x']= self.keyp['leftmouth_x']/self.WIDTH
        self.keyp['rightmouth_x']= self.keyp['rightmouth_x']/self.WIDTH
        
        self.keyp['lefteye_y']= self.keyp['lefteye_y']/self.HEIGHT
        self.keyp['righteye_y']= self.keyp['righteye_y']/self.HEIGHT
        self.keyp['nose_y']= self.keyp['nose_y']/self.HEIGHT
        self.keyp['leftmouth_y']= self.keyp['leftmouth_y']/self.HEIGHT
        self.keyp['rightmouth_y']= self.keyp['rightmouth_y']/self.HEIGHT
    
    def load_images(self, files):
        X = []
        for file in files:
            img = cv2.imread(os.path.join(self.DATASET_FOLDER,file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
            img = (img/255.0) - 0.5
            img = img[..., np.newaxis]
            X.append(img)
        return np.array(X, dtype='float32')    
    
    def load_targets(self, files):
        Y = []
        for file in files:
            ind = int(file[0:6])-1
            #Y.append(list(keyp.iloc[ind])[1:])
            if self.attr['Male'][ind]==1:
                Y.append(1)
            else:
                Y.append(0)
        return np.array(Y, dtype='float32')    
    
    def get_vaildation_data(self):
        testX = self.load_images(list(self.val['image_id']))
        testY = self.load_targets(list(self.val['image_id']))
        return testX, testY
    def get_file_name(self):
        return self.fileList
        
    
    def imageLoader(self, files, batch_size):
    
        L = len(files)
    
        #this line is just to make the generator infinite, keras needs that    
        while True:
    
            batch_start = 0
            batch_end = batch_size
    
            while batch_start < L:
                limit = min(batch_end, L)
                X = self.load_images(files[batch_start:limit])
                Y = self.load_targets(files[batch_start:limit])
    
                yield (X,Y) #a tuple with two numpy arrays with batch_size samples     
    
                batch_start += batch_size   
                batch_end += batch_size
                
