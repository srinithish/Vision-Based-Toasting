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





class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        #        self.pack()
        self.grid()
        self.create_widgets()
        self.reqCheckpoint = None
        
        
        
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
        self.reqCheckpoint = checkpoint_number
       # self.currAction["text"] = 'You pressed a button'
       
        self.currAction["text"] = 'You selected {}'.format(checkpoint_number)

    def startToasting(self):
        if self.reqCheckpoint is not None:
            self.disableButtons()
            self.cancel.state=tk.NORMAL
            
            self.enableButtons()
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


