from sklearn import svm
import pickle as pkl
import numpy as np
import random
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from sklearn.metrics import confusion_matrix

# Load file
def load_file(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# Load features of the pictures
def load_features(filename,dataset):
    all_features = pkl.load(open(filename, 'rb'))
    # filter features

    features = []
    for j in dataset:
        # for arr,_ in all_features[j]:
        # print(all_features[j][0])
        try:
            features.append(list(all_features[j][0]))
        except:
            continue

    # unzipped = []
    # for i in features:
    #     print(i)
    #
    #     unzipped.append(i)
    # print(unzipped)

    return features

def train_svm(feature_vector,labels):
    model = svm.SVC()
    model.fit(feature_vector,labels)
    return model

def predict_image(svm_model,path,target_size=224):
    model = VGG16()
    # Modify model to remove the last layer
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    image = load_img(path, target_size=(target_size, target_size))
    image = img_to_array(image)

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    img_feature = model.predict(image, verbose=0)
    return svm_model.predict(list(img_feature))




if __name__ == '__main__':
    positive_train = load_file('client_train_raw.txt')
    positive_train = positive_train.split('\n')
    positive_train = positive_train[0:-1]
    negative_train = load_file('imposter_train_raw.txt')
    negative_train = negative_train.split('\n')
    negative_train = negative_train[0:-1]


    positive_features = load_features('ClientRaw.pkl',positive_train)
    negative_features = load_features('ImposterRaw.pkl',negative_train)

    print("training non-spoofed images:", len(positive_features))
    print("training spoofed_images:", len(negative_features))

    positive_labels = list(np.zeros((len(positive_features)),int))
    negative_labels = list(np.ones((len(negative_features)),int))
    for i in range(len(negative_features)):
        positive_features.append(negative_features[i])
        positive_labels.append(negative_labels[i])

    combined = list(zip(positive_features, positive_labels))
    random.shuffle(combined)

    positive_features[:], positive_labels[:] = zip(*combined)
    print(len(positive_features[0]))

    model = train_svm(positive_features,positive_labels)
    print(model)

    positive_test = load_file('client_test_raw.txt')
    positive_test = positive_test.split('\n')
    positive_test = positive_test[0:-1]
    positive_features_test = load_features('ClientRaw.pkl',positive_test)
    negative_test = load_file('imposter_test_raw.txt')
    negative_test = negative_test.split('\n')
    negative_test = negative_test[0:-1]
    negative_features_test = load_features('ImposterRaw.pkl', negative_test)
    # print(positive_features_test[0])
    print("Test images not spoofed:",len(positive_features_test))
    print("Test images spoofed:",len(negative_features_test))
    predictions = []

    positive_labels_test = list(np.zeros((len(positive_features_test)), int))
    negative_labels_test = list(np.ones((len(negative_features_test)), int))
    for i in range(len(negative_features_test)):
        positive_features_test.append(negative_features_test[i])
        positive_labels_test.append(negative_labels_test[i])

    combined = list(zip(positive_features_test, positive_labels_test))
    random.shuffle(combined)

    positive_features_test[:], positive_labels_test[:] = zip(*combined)

    print("Now predicting...")
    for img in positive_features_test:

        predictions.append(model.predict([img])[0])

    tn, fp, fn, tp = confusion_matrix(list(positive_labels_test),predictions).ravel()

    print("TP:",tp)
    print("TN:",tn)
    print("FP:",fp)
    print("FN:",fn)

    print("Accuracy:",(tp+tn)/(tp+tn+fp+fn))


