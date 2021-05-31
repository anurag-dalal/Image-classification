import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt

print('[INFO] Loading model...')
model = tf.keras.models.load_model('detectv1.h5', custom_objects=None, compile=True, options=None)
print('[INFO] Loading model... Done')

DATASET_FOLDER = '../Dataset/celebA/img_align_celeba/img_align_celeba'
filename = '000432.jpg'
img = cv2.imread(os.path.join(DATASET_FOLDER,filename))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
img = (img/255.0) - 0.5
img = img[..., np.newaxis]
img = img[np.newaxis, ...]
img = img.astype('float32')


pred = model.predict(img) 
if(pred[0,0]>0.5):
    print('Male')
else:
    print('Female')