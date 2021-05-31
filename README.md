# Image-classification
This repository contain training pipeline for training images from celebA dataset for attribute based classification. Building an API and dockerizing it.

# Project Details
## Requirements and pre-requisites

* 1\. Python
    * 1.1\. tensorflow
    * 1.2\. opencv
    * 1.2\. flask
    * 1.2\. anaconda or venv
* 2\. Docker
* 3\. Postman

## Dataset
The dataset used is the CelebFaces dataset, this can be found in [Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset)

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

* To change the dataset folder edit dataloader2.py line 11 to 14
```python
        self.DATASET_FOLDER = '../Dataset/celebA/img_align_celeba/img_align_celeba'
        self.SPLIT_FILE = '../Dataset/celebA/list_eval_partition.csv'
        self.KEYPOINT_FILE = '../Dataset/celebA/list_landmarks_align_celeba.csv'
        self.ATTRIBUTE_FILE = '../Dataset/celebA/list_attr_celeba.csv'
```
* The CNN model used for calssification can be summerized as:
![model Image](/images/model.PNG "loss image")
* You can then run trainv3.py to train the model based on Male attribute, this can be changed in line 64 of dataloader2.py
         You can also open another terminal during training and run:
         ```
         tensorboard --logdir logs
         ```
         to view the accuracy and loss in various epochs
        The tensorboard logs are visualized like this: \
        ![Loss Image](/images/loss_and_accuracy.PNG "loss image")
* If you want to use the pretrained model skip the previous step, the pretrained model is contained in the detectv1.h5 file.
* The flask API is in the file app.py.
* The to dockerize it open a terminal, navigate to the proper location which contain Dockerfile, then execute the following command:
```bash
docker build -t flaskytd .
```
This will build the docker image, the process will look like:\
![Docker Build Image](/images/docker-build-image.png "loss image")
* Then we can run the dockerfile by the following command:
```bash
docker run -it -p 5000:5000 flaskytd
```
This will start the API running on localhost in port 5000. This will look like:
![Docker run Image](/images/docker-run-as-container.png "loss image")

* Then we can use postman to check whether the API is working properly:
![Postman Image](/images/postman-api.png "loss image")

