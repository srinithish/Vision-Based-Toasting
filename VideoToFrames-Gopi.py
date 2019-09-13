import os
import argparse
import time
import cv2
import enum

def extractImages(reqFps, tot_brown, InfilePath, OutFolderPath):
    
    cap = cv2.VideoCapture(InfilePath)
    count = 0
    Actualfps = cap.get(cv2.CAP_PROP_FPS)
    print(Actualfps)
    brown = 0
    samplingFreq = int(Actualfps/reqFps)
    print(samplingFreq)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(totalFrames)
    while cap.isOpened():

        x = True
        a = 0
        while x:
             a += 2
             if count < int(2 * totalFrames/((2*tot_brown) + 3)):
                 brown = 0
                 x = False

             if count > int(((2*tot_brown) + 2) * totalFrames/((2*tot_brown) + 3)):
                 brown = 0
                 x = False

             if int(a * totalFrames/((2*tot_brown) + 3)) <= count <= int((a+2) * totalFrames/((2*tot_brown) + 3)):
                 print(a)
                 brown = int(a/2)
                 x = False


        # Capture frame-by-frame
        ret, frame1 = cap.read()
        frame = frame1[140:650, 360:950]
        
        if ret is True and count % samplingFreq == 0 and brown != 0:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(OutFolderPath, "trail18frame{:d}_brown{:d}.jpg".format(count, brown)), frame)  # save frame as JPEG file
            
        elif ret is False:
            break
        
        count += 1
    cap.release()
    cv2.destroyAllWindows()

inpPath = r".\images\18.mp4"
extractImages(10,10, inpPath,r".\images\18")


