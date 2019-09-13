# -*- coding: utf-8 -*-
"""TrainImageClassifier_GE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oe1ix3coQ9x4-nPDvLqNiEojGRGiQk6z
"""

import tensorflow as tf
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import pickle

import os

import cv2 as cv
import time
from sklearn.model_selection import train_test_split

##ccustom imports
#import getRecipe
#import makeInference


##  True if training and False if inferencing
isTrain = True 
trainFromScratch = True



workDir = "./"

dictXnYPath = workDir+'Data/AllObjectLevelData.pkl'

dictOfXandY = pickle.load(open(dictXnYPath, 'rb'))


nClasses = len(set(dictOfXandY['Y']))

X = tf.placeholder(tf.float32,[None,28,28,3],name = "X")
Y  =  tf.placeholder(tf.float32,[None,nClasses], name = "Y")

### layer configuration

layer1 = { "type":"conv2d",
            "filters":16,
            "kernel_size":[4,4],
            "activation": tf.nn.leaky_relu
         }

layer2 = { "type":"maxpool",
            "pool_size":3,
            "padding":'valid'
         }


layer3 = { "type":"conv2d",
            "filters":32,
            "kernel_size":[4,4],
          "activation": tf.nn.leaky_relu
         }

layer4 = { "type":"maxpool",
            "pool_size":3,
            "padding":'valid'
         }



layer5 = { "type":"fullyConnected",
            'outputUnits': nClasses,
            
            "activation": None
          }

layers = [layer1,layer2,layer3,layer4,layer5]

def get_network_output(input_x,layers):
    '''
    
    inputs = input_x 
    output = latent_vectors
    
    input.shape => (batch_size,28,28)  // We need to reshape to add filters dim
    output.shape => (batch_size,6)  //6 values corresponding to 3 means and 3 sd
    '''
    constructed_network = []
    
    # He initialization
    initializer = tf.keras.initializers.he_normal()
        
    for layer in layers:
      
        if len(constructed_network) == 0: # This is the First layer
            this_input = input_x
        else:
            this_input = constructed_network[-1]
   
      
        if layer["type"] == "conv2d":
            layerOutput = tf.keras.layers.Conv2D( 
                         
                         filters = layer["filters"],
                         kernel_size = layer["kernel_size"],
                         strides = 1,
                         padding = "same",
                         kernel_initializer = initializer,
                         activation = layer["activation"]
                        )(this_input)
        
        elif layer["type"] == "maxpool":
            layerOutput = tf.keras.layers.MaxPool2D(
                         
                          pool_size = layer["pool_size"],
                          strides = layer["pool_size"], # Same as pool size to not consider the same box twice
                          padding='valid')(this_input)
        
        
        elif layer['type'] == 'fullyConnected':
          
            this_input = tf.keras.layers.Flatten()(this_input) ##flatten input
          
            layerOutput = tf.keras.layers.Dense(
                              units = layer['outputUnits'],
                              activation = layer["activation"],
                              kernel_initializer= initializer)(this_input)

            
        
        # Push this layer to network
        constructed_network.append(layerOutput)
      
    return constructed_network[-1]



""" params """


epochs = 200
learning_rate = 0.003
print_step = 1
save_step = 1
batchSize = 100
save_dir = workDir+'saved_models/'


"""### Loss and minimisation"""

networkOutput = get_network_output(X,layers)

gradOptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)








"""
Different predictions
"""
softmaxProbPredictions = tf.nn.softmax(networkOutput)
sigmoidProbPredictions = tf.nn.sigmoid(networkOutput)
maxSigmoidProbs = tf.reduce_max(sigmoidProbPredictions,axis=1)
labelPreds = tf.argmax(networkOutput, 1)


##losses and train step
lossCalcu  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=networkOutput, labels=Y))
#lossCalcu  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=networkOutput, labels=Y))
#lossCalcu = tf.losses.mean_squared_error(predictions=sigmoidProbPredictions, labels=Y)

train = gradOptimizer.minimize(lossCalcu)




correct_pred = tf.equal(tf.argmax(networkOutput, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#############
initialise = tf.global_variables_initializer()
sess = tf.InteractiveSession()

##when traiining initialise variables
#if isTrain == True:
sess.run(initialise)


##do not restore dense layers as their dimensions has changed
varsToRestore = [var for var in tf.global_variables() if not var.name.startswith('dense')] 
selectiveSaver = tf.train.Saver(varsToRestore)

saverAll = tf.train.Saver()



##when testing restore the  model


##would throw an error if now already intialised

## restore all except the dense layers
##change to selectiveSaver when training
if isTrain == True and trainFromScratch == False: ##when training
    selectiveSaver.restore(sess, save_dir+"model.ckpt")

if isTrain == False: ##when testing
    saverAll.restore(sess, save_dir+"model.ckpt")

    


xData = np.array(dictOfXandY['X'])
yData = tf.one_hot(np.array(dictOfXandY['Y']),depth = len(set(dictOfXandY['Y']))).eval()

x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.1, random_state=42)


if isTrain == True:
    for epoch in range(epochs):
    
        for iteration, offset in enumerate(range(0, len(x_train), batchSize)):
            
            x_batch, yBatch = x_train[offset: offset + batchSize], y_train[offset: offset + batchSize]
        
            feedTrain = {X:x_batch,Y:yBatch}
    
            sess.run(train,feed_dict = feedTrain)
    
            
        feedTest = {X:x_test,Y:y_test}
        loss, acc = sess.run([lossCalcu, accuracy], feed_dict=feedTest) 
            
            
        if epoch%print_step == 0:
            
          print("Step " + str(epoch) + ", Loss= " + str(loss) + ", Testing Accuracy= " + str(acc))
        
        if epoch%save_step == 0:
            
            save_path = saverAll.save(sess, save_dir+"model.ckpt")





if isTrain == False:
    vidCapHandle = makeInference.initVidCap(camNum=1)
    prevIngredientsDict = {}
    CLS_MAPPING_DICT = pickle.load(open('./Data/CLS_MAP_DICT.pkl','rb'))
    while True:
        

        testImg = makeInference.getFrame(vidCapHandle,mirror=True)
   
        listOfObjArr,objects,rectangles = makeInference.getObjectsFromTestImg(testImg)
        
        listOfObjArr = np.array(listOfObjArr)
        
        if cv.waitKey(1) == 27: 
                break
        
        
        if len(listOfObjArr) != 0:
            Labels,probs = sess.run([labelPreds,maxSigmoidProbs],feed_dict={X:listOfObjArr})
            
            
            makeInference.drawBoxesAndText(testImg,objects,rectangles,Labels,probs,CLS_MAPPING_DICT)
            
            
            ##recipe code
#            currIngredientsDict = makeInference.enumerateObjects(Labels)
#            
#            if prevIngredientsDict != currIngredientsDict:
#                topRecipe,topRecipeSummary = getRecipe.getSuggestedRecipe(currIngredientsDict)
#            
#            makeInference.putTextWrap(testImg,topRecipe,(50,50))
            
        cv.imshow("Tracking", testImg)
    
        
    cv.destroyAllWindows()
    vidCapHandle.release()   


#tf.global_variables(scope=None)
#[var for var in tf.global_variables() if not var.name.startswith('dense')]




#CLS_MAPPING_DICT = {'apple':0,'carrot':1,'cucumber':2}
#for i in range(len(x_test)):
#    img = x_test[i]
#    imgReshaped = np.reshape(img,(1,28,28,3))
#    reverseMappingDict = {value:key for key,value in CLS_MAPPING_DICT.items()}
#    Label = sess.run(labelPreds,feed_dict={X:imgReshaped})
##    plt.title(reverseMappingDict[Label[0]])
#    
#    plt.title(Label)
#    plt.imshow(img)
#    plt.pause(3)
#    plt.close()