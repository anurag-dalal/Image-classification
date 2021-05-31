from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
import tensorflow as tf

print('[INFO] Loading model...')
model = tf.keras.models.load_model('detectv1.h5', custom_objects=None, compile=True, options=None)
print('[INFO] Loading model... Done')
app = Flask(__name__)


@app.route('/upload', methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['File']
        print('[INFO] Receiving...',f.filename)
        f.save(secure_filename(f.filename))
        img = cv2.imread(f.filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
        img = (img/255.0) - 0.5
        img = img[..., np.newaxis]
        img = img[np.newaxis, ...]
        img = img.astype('float32')
        print('[REQUEST FROM]',request.remote_addr)
    
        pred = model.predict(img) 
        if(pred[0,0]>0.5):
            s = 'Male'
        else:
            s = 'Female'
        os.remove(f.filename)
        return jsonify({'msg': 'success','sex':s})
     


app.run(host='0.0.0.0', port=5000, debug=True)