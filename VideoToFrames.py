# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:44:54 2019

@author: nithish k
"""

import cv2

import numpy as np
import os

def extarctImages(reqFps,InfilePath,OutFolderPath):
    
    cap = cv2.VideoCapture(InfilePath)
    count = 0
    Actualfps = cap.get(cv2.CAP_PROP_FPS)
    
    samplingFreq = int(Actualfps/reqFps)
    while (cap.isOpened()):
 
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == True and count % samplingFreq == 0:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(OutFolderPath, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            
        elif (ret == False):
            break
        
        count += 1
    cap.release()
    cv2.destroyAllWindows()

inpPath = "C:/Users/ntihish/Videos/Hangouts video call - Google Chrome 08-01-2019 00_15_19.mp4"
extarctImages(10,inpPath,"C:/Users/ntihish/Documents/IUB/CV Reza/Images")




