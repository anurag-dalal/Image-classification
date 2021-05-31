# Image-classification
This repository contain training pipeline for training images from celebA dataset for attribute based classification. Building an API and dockerizing it.

# Project Details
## Enviornment and installation guides
The project is build using python 3.8.0 using anaconda for virtual enviornment.
Anaconda can be download and installed visiting the website [Anaconda Download](https://www.anaconda.com/products/individual)
To use anaconda create enviornment etc you can follow this [Anaconda Tutorial](https://www.youtube.com/watch?v=beh7GE4FdnM)

* Then you can clone this repository using git clone, and install requirements.txt
```bash
$ git clone https://github.com/anurag-dalal/Image-classification
$ cd Image-classification
$ pip install -r requirements.txt
```
* The folder structure of the will look like this
```
Dataset
│   
└───CelebA
│   │   list_attr_celeba.csv
│   │   list_bbox_celeba.csv
│   │   list_eval_partition.csv
│   │   list_landmarks_align_celeba.csv
|   |
│   └───img_align_celeba
│       │   000001.jpg
│       │   000002.jpg
│       │   ...
│   
Image-classification
|
└───images
|   │   loss_and_accuracy.PNG
|   │   model.PNG
|   app.py
|   convert_to_tflite.py
|   dataloader.py
|   dataloader2.py
|   detectv1.h5
|   Dockerfile
|   EDA1.ipynb
|   predict.py
|   requirements.txt
|   train.py
|   visualization.py
```
