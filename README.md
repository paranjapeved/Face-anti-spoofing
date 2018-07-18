# Face-anti-spoofing
This is a project which uses Machine Learning techniques(CNN followed by SVM) to classify an image if it is spoofed or not spoofed.

## Application
With many face recognition systems coming into action, it is crucial to first check the image/video data for anti spoofing. 
This project is a part of a larger project of face verification using Siamese Neural Network.

## Dataset
http://parnec.nuaa.edu.cn/xtan/NUAAImposterDB_download.html

## Model details
The CNN (VGG16 or any other architecture) extracts features from a picture and encodes an image in a 4096 dimensional vector.
These vectors of training images are then passed on to an SVM as input and the output of SVM is binary (0 for non-spoofed and 1 
for spoofed). This simple model gives 97% accuracy on the NUAA dataset.

## Instructions

