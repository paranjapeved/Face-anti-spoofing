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
Run the create_embeddings.py to generate the embeddings for your images. You can change the name of the folder (default "raw").
Two files called ImposterRaw.pkl and ClientRaw.pkl would be formed which would contain the training features. 
Then run model.py which trains an SVM using these two pickle files as inputs and tests the trained model on the testing dataset. The train-test files can be specified as seprate txt files where each line contains the location of the corresponding images.

