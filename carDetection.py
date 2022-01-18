from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image
import sys
import numpy as np
import cv2
import os
from LPDetection import *
import time

NS_TO_MS = 1/1000000

class CarDetector():
    # Class constructor, the values are setted
    def __init__(self, modelCfg = "", modelWeights = "", labels=[], imagePath="", threshold=0):
        self.detectionType = "Yolo"
        self.threshold = threshold
        self.modelConfig = modelCfg
        self.modelWeights = modelWeights
        if os.path.isfile(self.modelConfig) and os.path.isfile(self.modelWeights):
            self.loadModel()
        self.labelNames = labels
        self.imagePath = imagePath
        self.image = None
        if os.path.isfile(self.imagePath):
            self.image = cv2.imread(self.imagePath)
            self.imageResults = self.image.copy()
        # Results
        self.validOutputBoxes = []
        weights_path = "./checkpoints/custom-416"
        nameClasses = ['license_plate']
        self.LPDetector = LPDetector(weights=weights_path, nameClass=nameClasses)
        self.LPDetector.loadModel()

    def getImagePath (self):
        return self.imagePath

    def getImage(self):
        return self.image

    def setImagePath (self, imagePath):
        if os.path.isfile(imagePath):
            self.imagePath = imagePath
            self.image = cv2.imread(self.imagePath)
            self.imageResults = self.image.copy()
        else:
            print ("There is not "+imagePath)

    def setImage (self, image):
        self.image = image.copy()
        self.imagePath = self.image.copy()

    def getThreshold (self):
        return self.threshold
    
    def setThreshold(self, threshold):
        self.threshold = threshold

    def cleanModel(self):
        self.net.__del__()

    def loadModel(self):
        # The model files are seleccted
        if self.detectionType == "Yolo":
            self.modelConfig = 'yolov3.cfg'
            self.modelWeights = 'yolov3.weights'
            self.setThreshold(0.7)
        elif self.detectionType == "Yolo Tiny":
            self.modelConfig = 'yolov3-tiny.cfg'
            self.modelWeights = 'yolov3-tiny.weights'
            self.setThreshold(0.2)

        # And the model is created
        self.net = cv2.dnn.readNetFromDarknet(self.modelConfig,self.modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    def detection(self):
        t0 = time.time_ns()
        # The image format is changed to input it to the network
        blob = cv2.dnn.blobFromImage(self.image,1/255,(320,320),[0,0,0],crop=False)
        self.net.setInput(blob)

        layerNames = self.net.getLayerNames()
        outputNames = [layerNames[i[0]-1] for i in self.net.getUnconnectedOutLayers()]
        outputBoxes = self.net.forward(outputNames)
        t = (time.time_ns() - t0) * NS_TO_MS
        print ("Car detection time: " + str(t))
        
        imgHeigth, imgWidth, aux = self.image.shape
        self.validOutputBoxes = []
        self.imageResults = self.image.copy()
        for i in range(len(outputBoxes)):
            for box in outputBoxes[i]:
                # 7th column correspond to car
                # 8th column correspond to motorbikes 
                # 12th column correspond to trucks 
                if box[7] > self.threshold or box[8] > self.threshold or box[12] > self.threshold:
                    # We are only adding the x,y coordinates of the center and the 
                    # width and heigth of the box
                    x,y = int(box[0]*imgWidth), int(box[1]*imgHeigth)
                    # We check if the box is repeated
                    repeat = 0
                    for i in range(len(self.validOutputBoxes)):
                        if x > self.validOutputBoxes[i][0] and x < self.validOutputBoxes[i][0]+self.validOutputBoxes[i][2] and y > self.validOutputBoxes[i][1] and y < self.validOutputBoxes[i][1]+self.validOutputBoxes[i][3]:
                            repeat = 1
                            break
                    if not repeat:
                        # Width and heihnt are calculated
                        w,h = int(box[2]*imgWidth), int(box[3]*imgHeigth)
                        # We store the value of the bottom left corner of the box
                        x = int(x-w/2)
                        y = int(y-h/2)
                        self.validOutputBoxes.append((x,y,w,h))
                        # The bounding box of the car is shown
                        cv2.rectangle(self.imageResults, (x,y), (x+w,y+h), (255, 0, 0), 2)
                        self.LPDetector.setImage(self.image[y:y+h,x:x+w])
                        self.detectLP(x,y) 
        # If there is no car detected, it migth happen that the car is in the image
        # but it is to close so the LP algoritmh detection is executed
        if not len(self.validOutputBoxes):
            self.LPDetector.setImage(self.imageResults)
            self.detectLP() 
        return(self.imageResults)

    def detectLP(self,x=0,y=0):
        # LP detection
        t0 = time.time_ns()
        if self.LPDetector.detectionType == "Yolo":
            xLP, yLP, wLP, hLP = self.LPDetector.yoloDetection()
        elif self.LPDetector.detectionType == "Edge detection":
            xLP, yLP, wLP, hLP = self.LPDetector.edgeDetection()
        t = (time.time_ns() - t0) * NS_TO_MS
        print ("License plate detection time: " + str(t))
        if xLP != -1:
            # The bounding box of the license plate is shown
            cv2.rectangle(self.imageResults, (x+xLP,y+yLP), (x+xLP+wLP,y+yLP+hLP), (0,148,71), 2)
            # And the characters are shown just upside the LP
            t0 = time.time_ns()
            LPnum = self.LPDetector.characterRecognition()
            t = (time.time_ns() - t0) * NS_TO_MS
            print ("Character recognition time: " + str(t))
            if len(LPnum):
                cv2.rectangle(self.imageResults, (x+xLP,y+yLP), (x+xLP+int(12.5*len(LPnum)),y+yLP-int(17)), (0,148,71), -1)
                cv2.putText(img = self.imageResults, text = LPnum, org = (x+xLP,y+yLP), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = (255,255,255), thickness = 1)
            else:
                cv2.rectangle(self.imageResults, (x+xLP,y+yLP), (x+xLP+int(8*len("Not recon.")),y+yLP-int(13)), (0,148,71), -1)
                cv2.putText(img = self.imageResults, text = "Not recon.", org = (x+xLP,y+yLP), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,255,255), thickness = 1)
