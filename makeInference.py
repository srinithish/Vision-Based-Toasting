# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:34:35 2019

@author: AI Lab
"""

import cv2 as cv

import os 
import numpy as np
import collections as col

import matplotlib.pyplot as plt

def enumerateObjects(Label,classMappingDict):
    
    reverseMappingDict = {value:key for key,value in classMappingDict.items()}
    enumeratedObjs = col.Counter(Label)
    enumeratedObjs ={reverseMappingDict[key]:value for  key,value in enumeratedObjs.items() if value != 0}
    
    
    return enumeratedObjs

def initVidCap(camNum=0):
    vidCapHandle = cv.VideoCapture(camNum)
    return vidCapHandle
    
    
def getFrame(vidCapHandle,mirror=False):
    ret_val, img = vidCapHandle.read()
    if mirror: 
        img = cv.flip(img, 1)
        
    return img

def showFrame(img,text):
    
    
    font                   = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (250,200)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2
    
    
    
    cv.putText(img,str(text), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    cv.imshow('LiveFeed', img)
    
    pass








def putTextWrap(imgArray,text,location):
    
    
    ##location (width,height)
    
    font                   = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = location
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2
    
    
    
    cv.putText(imgArray,str(text), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    return imgArray
    
    
def getTestImg(imgArray):
    
    
    resizedImg = cv.resize(imgArray,(28,28))
    
    
    return [resizedImg]

if __name__ == '__main__':
    
    y1 = 0
    y2 = 470
    x1 = 143
    x2 = 530
    
    img = cv.imread('./imageCaptures/TestImgsAll_9.jpg')
   
    
    listOfObjArr,objects,rectangles = getObjectsFromTestImg(img) ##o are the objects you need Natish
    
    drawBoxesAndText(img,objects, rectangles,[0,1,1,0,2])

    cv.imshow("Tracking", img)
    
    
    
    
#    plt.imshow(img)
#    showFrame(img,'hello')
