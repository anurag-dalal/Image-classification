import cv2
import pandas as pd
import os


DATASET_FOLDER = '../Dataset/celebA/img_align_celeba/img_align_celeba'
SPLIT_FILE = '../Dataset/celebA/list_eval_partition.csv'
KEYPOINT_FILE = '../Dataset/celebA/list_landmarks_align_celeba.csv'

HEIGHT = 218
WIDTH = 178

def show(filename = '000001.jpg'):

    img = cv2.imread(os.path.join(DATASET_FOLDER, filename))
    
    keyp = pd.read_csv(KEYPOINT_FILE)
    
    lefteye = (keyp.loc[keyp['image_id'] == filename]['lefteye_x'].get(0), keyp.loc[keyp['image_id'] == filename]['lefteye_y'].get(0))
    righteye = (keyp.loc[keyp['image_id'] == filename]['righteye_x'].get(0), keyp.loc[keyp['image_id'] == filename]['righteye_y'].get(0))
    nose = (keyp.loc[keyp['image_id'] == filename]['nose_x'].get(0), keyp.loc[keyp['image_id'] == filename]['nose_y'].get(0))
    leftmouth = (keyp.loc[keyp['image_id'] == filename]['leftmouth_x'].get(0), keyp.loc[keyp['image_id'] == filename]['leftmouth_y'].get(0))
    rightmouth = (keyp.loc[keyp['image_id'] == filename]['rightmouth_x'].get(0), keyp.loc[keyp['image_id'] == filename]['rightmouth_y'].get(0))
    
    img = cv2.circle(img, lefteye, 4, (255, 0, 0), 2)
    img = cv2.circle(img, righteye, 4, (255, 0, 0), 2)
    img = cv2.circle(img, nose, 4, (255, 0, 0), 2)
    img = cv2.circle(img, leftmouth, 4, (255, 0, 0), 2)
    img = cv2.circle(img, rightmouth, 4, (255, 0, 0), 2)
    
    
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()