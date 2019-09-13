import numpy as np
import argparse
import time
import cv2
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from pyvesync import VeSync


class Toast:
    class WhiteToast:
        def __init__(self):
            self.checkpoint = [0 for i in range(5)]
            self.checkpoint[0] = [29.468043434429468,
                                  17.998631873037834,
                                  170.48640513840266]
            self.checkpoint[1] = [14.02681869279675,
                                  25.705728509485066,
                                  166.41480648787618]
            self.checkpoint[2] = [16.240619786299003,
                                  53.52060629725393,
                                  156.3137978772478]
            self.checkpoint[3] = [14.385065086927959,
                                  52.54102034453614,
                                  155.5156286538287]
            self.checkpoint[4] = [132.84593700559958,
                                  39.63402847691675,
                                  165.57704587947944]

    class WholeToast:
        def __init__(self):
            self.checkpoint = [0 for i in range(6)]
            self.checkpoint[0] = [14.623071367614308,
                                  18.58698268881373,
                                  157.24172192677926]
            self.checkpoint[1] = [14.993267194862128,
                                  18.618823305500214,
                                  161.08966764208577]
            self.checkpoint[2] = [11.535315170529778,
                                  23.30728215616035,
                                  158.9387675601969]
            self.checkpoint[3] = [163.95510822126946,
                                  21.21115762595961,
                                  147.78156824322997]
            self.checkpoint[4] = [166.61919642860357,
                                  21.484611344541577,
                                  146.2332983193259]
            self.checkpoint[5] = [127.48667170395028,
                                  31.850041293994256,
                                  143.01692175227586]


def get_dominant_color(image, k=4, image_processing_size=None):
    """
    takes an image as input
    returns the dominant color of the image as a list

    dominant color is found by running k means on the
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image;
    this resizing can be done with the image_processing_size param
    which takes a tuple of image dims as input

    > get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size,
                           interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(4)[0][0]]

    return list(dominant_color)


def cropRequiredPart(frame):
    ## nabil crop as required

    x1 = 300
    x2 = 600
    y1 = 300
    y2 = 600

    frame = frame[y1:y2, x1:x2]

    return frame


def isPopNow(smoothList, threshValue):
    if smoothList[-1] > smoothList[-2] and smoothList[-1] < threshValue:
        SwitchOff()
        print("true")
    else:
        # print("false")
        pass


#    booleanList = [True if smoothList[i]>smoothList[i-1] and smoothList[i]<threshValue else False for i in range(1,len(smoothList))]

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


def saveHSVDistances(pickleFileName, checkPointHSV, camNum=0):
    vidCap = cv2.VideoCapture(camNum)  ## from camera

    valueList = []
    smoothList = []

    # checkPointImg = cv2.imread(checkPointImgPath)
    # checkPointFrame = cropRequiredPart(checkPointImg)

    hsvCheckPoint = np.array(checkPointHSV)

    print("Camera on Bro!!")
    while True:

        ##        time.sleep(0.1)

        ret_val, currFrame = vidCap.read()

        #        cv.imwrite(os.path.join(saveFolder,fileName),img)

        if cv2.waitKey(1) == 27:
            break

        currFrame = cropRequiredPart(currFrame)
        cv2.imshow('Reading Cam', currFrame)

        currFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2HSV)
        hsvCurrent = np.array(get_dominant_color(currFrame))

        distValue = np.linalg.norm(hsvCurrent - hsvCheckPoint)
        valueList.append(distValue)
        ##        if len(valueList) > 102:

        meanHistory = np.mean(valueList[-100:])
        smoothList.append(meanHistory)
        if len(smoothList) > 2:
            isPopNow(smoothList, 20)
    #        print('time : ', time.time(),'Value: ', distValue)

    pk.dump(valueList, open(pickleFileName, 'wb'))
    cv2.destroyAllWindows()
    vidCap.release()
    print('all done')


selected_toast = Toast.WhiteToast()
print("Give check number")
input_value = int(input())
selected_checkpoint = selected_toast.checkpoint[input_value - 1]
##path="C:\\image\\5thcheck_Moment.jpg"
##frame =  cv2.imread(path)


# NABIL YOU DECIDE THESE CROP VALUES

##x1 = 300
##x2 = 700
##y1 = 300
##y2 = 600
##
##x1 = 350
##x2 = 600
##y1 = 300
##y2 = 600
##frame = frame[y1:y2, x1:x2]
##cv2.imshow("f",frame)
##cv2.waitKey(0)
##frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# DOMINANT COLOR IS HSV WILL BE PRINTED
##print(get_dominant_color(frame))
print("start")
##saveHSVDistances('./allDist.pkl',[16.240619786299003,53.52060629725393, 156.3137978772478],0)

saveHSVDistances('./allDist.pkl', selected_checkpoint, 0)


