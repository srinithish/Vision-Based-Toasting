import numpy as np
import imutils
import glob
import cv2
import os
import tensorflow as tf
from keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import pickle

def image_arrays(image_path):
    image_list = [cv2.imread(img).ravel() for img in image_path]
    return image_list

def extract_labels(inpPath):
    dirListing = os.listdir(inpPath)
    labels = []
    for item in dirListing:
        if ".jpg" in item:
            #print(item)
            #print(int(item[-5]))
            labels.append(int(item[-6]))
    print(len(labels))
    print(np.asarray(labels))
    return labels

filename = './Data/rf_model200.pkl'
filename2 = './Data/rf_model167.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model2 = pickle.load(open(filename2, 'rb'))

test_image_path = glob.glob("./images/test/*.jpg")
test_inpPath = "./images/test"
test_image_list = image_arrays(test_image_path)
extract_labels(test_inpPath)

predicted = loaded_model.predict(test_image_list)
predicted2 = loaded_model2.predict(test_image_list)
print(predicted)
print(predicted2)

