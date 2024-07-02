This repository consists of a classifier that I created as part of my master thesis. 
The classifier aims to identify artefacts from EEG recordings

Preprocessing.py was used to lable the images and sort them into folders for the classifier (Can be ignored for usage)

Classifier.py contains the defined model + the model training

## Prerequisite: 
Change the data directory to the folder where the input data for the classifier is stored. Images need to be seperated into folders to represent classes

## Following versions are required:

Python 3.11

Tensorflow 2.13.1

numpy 1.24.3

## Trained Models:
Model able to classify 5 classes (Brain Activity, Blink, Channel Pop, Horizontal Eye Movement, Muscle) https://www.dropbox.com/scl/fo/4je1irwrx4f4rzh057npt/ADXaGN206EJkp2RJHu1-HvI?rlkey=byrhm8bjlm5pvmq1deau7xaxm&st=s5hqe7zp&dl=0

Model able to classify 2 classes (Brain Activity, Artefacts (see above) https://www.dropbox.com/scl/fo/zfhhm3aop225yjzlq29cr/AIOnyenm-stEXP-LZYFP_KQ?rlkey=qc1khfyb3hmsp6ficjnqyje9w&st=rccru1qk&dl=0

Intended input format of the picture: https://neuraldatascience.io/_images/6ea52e7cc36a10990a9c59b6f4437693dd76ea733502a146ef0bc6d036098677.png

## How to Use 
Open Load_Model and set the directories accordingly.
