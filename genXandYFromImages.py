# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:13:36 2019

@author: GELab
"""


import glob

import os

import cv2

import collections as col

import pickle
import re

import tqdm








def genXandY(folderPath):
    
    dictOfImgAndLabels = col.defaultdict(list)
    
    
    fileList = sorted(glob.glob(folderPath+'*'))    
    
    
    for file in tqdm.tqdm(fileList):
        
        
        filename = os.path.basename(file)
        filenameNoExt = os.path.splitext(os.path.basename(filename))[0]
        
        
        
#        Label = filenameNoExt.split('_')[1].replace('brown','')
        m = re.match("\d+brown(?P<Labels>\d+)", 
                         filenameNoExt,re.IGNORECASE)
        
        
        Label = m.group('Labels')
        
        
        imgArray = cv2.imread(file)
        resizedImg = cv2.resize(imgArray,(28,28))
        
        
        dictOfImgAndLabels['X'].append(resizedImg)
        dictOfImgAndLabels['filename'].append(filename)
        LabelInt = int(Label)-1 ## adjusting as per onehot encoding
        dictOfImgAndLabels['Label'].append(Label) ##all labels
        dictOfImgAndLabels['Y'].append(LabelInt)
    
    return dictOfImgAndLabels


if __name__ == '__main__':
    
    
    dictOfImgAndLabels = genXandY('./images/train/')

    pickle.dump(dictOfImgAndLabels,open('./Data/AllObjectLevelData.pkl','wb'))
    
    
