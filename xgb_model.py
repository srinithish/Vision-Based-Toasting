import numpy as np
import imutils
import glob
import cv2
import os
import tensorflow as tf
import xgboost
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
    print(np.asarray(image_list).shape)
    return image_list

def extract_labels(inpPath):
    dirListing = os.listdir(inpPath)
    labels = []
    for item in dirListing:
        if ".jpg" in item:
            #print(item)
            #print(int(item[-5]))
            labels.append(int(item[-5]))
    print(len(labels))
    return labels

def build_model(image_list, labels):
    xgb_model = xgboost.XGBClassifier(objective="multi:softprob", random_state=42)
    xgb_model.fit(np.asarray(image_list), np.asarray(labels))
    return model


image_path = glob.glob("./images/train/*.jpg")
inpPath = "./images/train"
image_list = image_arrays(image_path)
labels = extract_labels(inpPath)

# save the model to disk
model = build_model(image_list,labels)
filename = 'xg_boost.pkl'
pickle.dump(model, open(filename, 'wb'))