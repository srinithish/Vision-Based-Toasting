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

'''
# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

# variables to hold the results and names
results = []
names = []
'''

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
    # split the training and testing data
    #(trainDataGlobal, testDataGlobal,
    #trainLabelsGlobal, testLabelsGlobal) = train_test_split(image_list, labels,
    #                                                       test_size=test_size, random_state=seed)
    model = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
    #model = LinearDiscriminantAnalysis()
    #model = KNeighborsClassifier()
    model.fit(image_list,labels)
    return model

num_trees = 100
test_size = 0.10
seed = 9
scoring = "accuracy"

image_path = glob.glob("./images/train2/*.jpg")
inpPath = "./images/train2"
image_list = image_arrays(image_path)
labels = extract_labels(inpPath)

# save the model to disk
model = build_model(image_list,labels)
filename = 'rf_model167.pkl'
pickle.dump(model, open(filename, 'wb'))


'''
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
'''

'''
# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
'''

