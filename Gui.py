# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tkinter as tk
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

selected_checkpoint = None
class Toast:
    class WhiteToast:
        def __init__(self):
            self.checkpoint = [0 for i in range(5)]

            self.checkpoint[0] = [14.02681869279675,
                                  25.705728509485066,
                                  166.41480648787618]
            self.checkpoint[1] = [10.96570238876549,
                                  36.018262838781034,
                                  155.04439696105982]
            #self.checkpoint[2] = [16.240619786299003,
            #                      53.52060629725393,
            #                      156.3137978772478]
            self.checkpoint[2]=[16.245588444564774,
                                53.545147289025905,
                                156.32362672548555]

            self.checkpoint[3] = [169.57960260169412,
                                  34.39089425806489,
                                  151.7104892604135]

            self.checkpoint[4] = [132.8568631403055,
                                  39.63068329062197,
                                  165.59129257969138]


    class WholeToast:
        def __init__(self):
            self.checkpoint = [0 for i in range(5)]
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
        #        time.sleep(0.1)
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


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        #        self.pack()
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        self.img_white_one = tk.PhotoImage(file="C:\\Users\\240028730\\Pictures\\Camera Roll\\White Bread\\actualcheck4.png")
        self.img_white_one = self.img_white_one.subsample(5, 5)
        self.checkpoint_white_one_Btn = tk.Button(self, image=self.img_white_one)
        self.checkpoint_white_one_Btn.image = self.img_white_one
        self.checkpoint_white_one_Btn["command"] = lambda:self.setNumber(checkpoint_number=1)
        self.checkpoint_white_one_Btn.grid(row=0, column=0, sticky="nswe")

        self.img_white_two = tk.PhotoImage(file="C:\\Users\\240028730\\Pictures\\Camera Roll\\White Bread\\actualcheck4.png")
        self.img_white_two = self.img_white_two.subsample(5, 5)
        self.checkpoint_white_two_Btn = tk.Button(self, image=self.img_white_two)
        self.checkpoint_white_two_Btn.image = self.img_white_two
        self.checkpoint_white_two_Btn["command"] = lambda:self.setNumber(checkpoint_number=2)
        self.checkpoint_white_two_Btn.grid(row=0, column=1, sticky="nswe")

        self.img_white_three = tk.PhotoImage(file="C:\\Users\\240028730\\Pictures\\Camera Roll\\White Bread\\actualcheck4.png")
        self.img_white_three = self.img_white_three.subsample(5, 5)
        self.checkpoint_white_three_Btn = tk.Button(self, image=self.img_white_three)
        self.checkpoint_white_three_Btn.image = self.img_white_three
        self.checkpoint_white_three_Btn["command"] = lambda:self.setNumber(checkpoint_number=3)
        self.checkpoint_white_three_Btn.grid(row=0, column=2, sticky="nswe")

        self.img_white_four = tk.PhotoImage(file="C:\\Users\\240028730\\Pictures\\Camera Roll\\White Bread\\actualcheck4.png")
        self.img_white_four = self.img_white_four.subsample(5, 5)
        self.checkpoint_white_four_Btn = tk.Button(self, image=self.img_white_four)
        self.checkpoint_white_four_Btn.image = self.img_white_four
        self.checkpoint_white_four_Btn["command"] = lambda:self.setNumber(checkpoint_number=4)
        self.checkpoint_white_four_Btn.grid(row=0, column=3, sticky="nswe")

        self.img_white_five = tk.PhotoImage(file="C:\\Users\\240028730\\Pictures\\Camera Roll\\White Bread\\actualcheck6.png")
        self.img_white_five = self.img_white_five.subsample(5, 5)
        self.checkpoint_white_five_Btn = tk.Button(self, image=self.img_white_five)
        self.checkpoint_white_five_Btn.image = self.img_white_five
        self.checkpoint_white_five_Btn["command"] = lambda:self.setNumber(checkpoint_number=5)
        self.checkpoint_white_five_Btn.grid(row=0, column=4, sticky="nswe")

        self.img_whole_one = tk.PhotoImage(file="C:\\Users\\240028730\\Pictures\\Camera Roll\\White Bread\\actualcheck4.png")
        self.img_whole_one = self.img_whole_one.subsample(5, 5)
        self.checkpoint_whole_one_Btn = tk.Button(self, image=self.img_whole_one)
        self.checkpoint_whole_one_Btn.image = self.img_whole_one
        self.checkpoint_whole_one_Btn["command"] = lambda:self.setNumber(checkpoint_number=6)
        self.checkpoint_whole_one_Btn.grid(row=1, column=0, sticky="nswe")

        self.img_whole_two = tk.PhotoImage(file="C:\\Users\\240028730\\Pictures\\Camera Roll\\White Bread\\actualcheck4.png")
        self.img_whole_two = self.img_whole_two.subsample(5, 5)
        self.checkpoint_whole_two_Btn = tk.Button(self, image=self.img_whole_two)
        self.checkpoint_whole_two_Btn.image = self.img_whole_two
        self.checkpoint_whole_two_Btn["command"] = lambda:self.setNumber(checkpoint_number=7)
        self.checkpoint_whole_two_Btn.grid(row=1, column=1, sticky="nswe")

        self.img_whole_three = tk.PhotoImage(
            file="C:\\Users\\240028730\\Pictures\\Camera Roll\\White Bread\\actualcheck4.png")
        self.img_whole_three = self.img_whole_three.subsample(5, 5)
        self.checkpoint_whole_three_Btn = tk.Button(self, image=self.img_whole_three)
        self.checkpoint_whole_three_Btn.image = self.img_whole_three
        self.checkpoint_whole_three_Btn["command"] = lambda:self.setNumber(checkpoint_number=8)
        self.checkpoint_whole_three_Btn.grid(row=1, column=2, sticky="nswe")

        self.img_whole_four = tk.PhotoImage(
            file="C:\\Users\\240028730\\Pictures\\Camera Roll\\White Bread\\actualcheck4.png")
        self.img_whole_four = self.img_whole_four.subsample(5, 5)
        self.checkpoint_whole_four_Btn = tk.Button(self, image=self.img_whole_four)
        self.checkpoint_whole_four_Btn.image = self.img_whole_four
        self.checkpoint_whole_four_Btn["command"] = lambda:self.setNumber(checkpoint_number=9)
        self.checkpoint_whole_four_Btn.grid(row=1, column=3, sticky="nswe")

        self.img_whole_five = tk.PhotoImage(
            file="C:\\Users\\240028730\\Pictures\\Camera Roll\\White Bread\\actualcheck4.png")
        self.img_whole_five = self.img_whole_five.subsample(5, 5)
        self.checkpoint_whole_five_Btn = tk.Button(self, image=self.img_whole_five)
        self.checkpoint_whole_five_Btn.image = self.img_whole_five
        self.checkpoint_whole_five_Btn["command"] = lambda:self.setNumber(checkpoint_number=10)
        self.checkpoint_whole_five_Btn.grid(row=1, column=4, sticky="nswe")
        ## quit button
        self.quit_btn = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)

        self.quit_btn.grid(row=3, column=1)

        self.cancel_btn = tk.Button(self, text="CANCEL", fg="red",state=tk.DISABLED,
                              command=lambda:SwitchOff())

        self.cancel_btn.grid(row=3, column=2)

        self.start_btn = tk.Button(self, text="START", fg="green",
                              command=self.startToasting)

        self.start_btn.grid(row=3, column=3)
        #        self.quit.pack(side="right")

        #   message button:

        self.currAction = tk.Label()
        self.currAction["text"] = "Hello"
        self.currAction.grid(row=2, column=0)

    def setNumber(self, checkpoint_number):
        global selected_checkpoint
       # self.currAction["text"] = 'You pressed a button'
        print("hi there, everyone!")
        if(checkpoint_number<6):
            selected_toast = Toast.WhiteToast()
            selected_checkpoint = selected_toast.checkpoint[checkpoint_number-1]
        else :
            selected_toast = Toast.WholeToast()
            selected_checkpoint = selected_toast.checkpoint[checkpoint_number-6]

    def startToasting(self):
        if not selected_checkpoint == None:
            Application.disableButtons(self)
            self.cancel.state=tk.NORMAL
            saveHSVDistances('./allDist.pkl', selected_checkpoint, 0)
            Application.enableButtons(self)
            self.cancel.state =tk.DISABLED

        else :
            print("Please select the desired brownness first!")

    def disableButtons(self):

        self.checkpoint_white_one_Btn.config(state=tk.DISABLED)
        self.checkpoint_white_two_Btn.config(state=tk.DISABLED)
        self.checkpoint_white_three_Btn.config(state=tk.DISABLED)
        self.checkpoint_white_four_Btn.config(state=tk.DISABLED)
        self.checkpoint_white_five_Btn.config(state=tk.DISABLED)
        self.checkpoint_whole_one_Btn.config(state=tk.DISABLED)
        self.checkpoint_whole_two_Btn.config(state=tk.DISABLED)
        self.checkpoint_whole_three_Btn.config(state=tk.DISABLED)
        self.checkpoint_whole_four_Btn.config(state=tk.DISABLED)
        self.checkpoint_whole_five_Btn.config(state=tk.DISABLED)

    def enableButtons(self):

        self.checkpoint_white_one_Btn.config(state=tk.NORMAL)
        self.checkpoint_white_two_Btn.config(state=tk.NORMAL)
        self.checkpoint_white_three_Btn.config(state=tk.NORMAL)
        self.checkpoint_white_four_Btn.config(state=tk.NORMAL)
        self.checkpoint_white_five_Btn.config(state=tk.NORMAL)
        self.checkpoint_whole_one_Btn.config(state=tk.NORMAL)
        self.checkpoint_whole_two_Btn.config(state=tk.NORMAL)
        self.checkpoint_whole_three_Btn.config(state=tk.NORMAL)
        self.checkpoint_whole_four_Btn.config(state=tk.NORMAL)
        self.checkpoint_whole_five_Btn.config(state=tk.NORMAL)

root = tk.Tk()
app = Application(master=root)
app.mainloop()


