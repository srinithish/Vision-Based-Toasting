import numpy as np
import imutils
import glob
import cv2
import os
import tensorflow as tf
from keras.preprocessing.image import img_to_array

import sys

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

import time
import pickle
import collections


'''
def image_arrays(image):
    image_array = []
    image_array.append(image.ravel())
    return image_array
'''
def cropRequiredPart(frame):
    x1 = 170
    x2 = 500
    y1 = 130
    y2 = 400
    frame = frame[y1:y2, x1:x2]
    return frame

def showFrame(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (250, 200)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, str(text),
               bottomLeftCornerOfText,
               font,
               fontScale,
               fontColor,
               lineType)

    cv2.imshow('LiveFeed', img)
    pass

def SwitchOff():
    manager = VeSync("juan.huerta@geappliances.com", "Ge1234")
    manager.login()
    manager.update()
    my_switch = manager.outlets[0]
    # Turn on the first switch
    my_switch.turn_off()
    time.sleep(2)
    # Turn off the first switch
    my_switch.turn_on()
    # Get energy usage data
    # manager.update_energy()
    # Display outlet device information
    # for device in manager.outlets:
    #    device.display()
    pass


def image_arrays(image_path):
    image_list = [cv2.imread(img).ravel() for img in image_path]
    return image_list

def brown_test(brown,skip = int(10)):
    req_brown = brown
    req_perc = 0.8
    camNum = 0
    pool_frames = 5 * skip
    skip_frames = skip
    condition_check_frames = int(20)

    filename = './Data/rf_model200.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    frame_path = glob.glob("./images/live/*.jpg")

    pred_brown_list = []

    vidCap = cv2.VideoCapture(camNum)  ## from camera
    print("CAMERA ON!")

    count = 0

    while True:
        count+=1
        # Capture frame-by-frame
        ret, frame1 = vidCap.read()
        frame = cropRequiredPart(frame1)
        frame = cv2.resize(frame, (590, 510))

        if count % skip_frames == 0:

            if ret is True and count % pool_frames == 0:
                    count = 0
            else:
                cv2.imwrite(os.path.join('./images/live', "frame{:d}.jpg".format(count)), frame)

            if ret is True and count % pool_frames == (pool_frames - skip_frames):
                image_array = image_arrays(frame_path)
                predicted_brown = loaded_model.predict(image_array)
                print(predicted_brown)
                pred_brown_list.extend(predicted_brown)
                #print(pred_brown_list)
                showFrame(frame, predicted_brown)

                if len(pred_brown_list) >= condition_check_frames:

                    if cv2.waitKey(1) == 27:
                        break
                    counter = collections.Counter(pred_brown_list[-condition_check_frames:-1])
                    print(counter)

                    sum = 0

                    for i in range(0,10):

                        if req_brown + i <= 10:
                            sum = sum + pred_brown_list[-condition_check_frames:-1].count(req_brown + i)

                            if sum >= int(req_perc * condition_check_frames):
                                    print("True")
                                    return True

start_time = time.time()
brown_test(7)
print("Time taken: {:d} seconds".format(int(time.time() - start_time)))
