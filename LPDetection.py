import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from core.config import cfg
import pytesseract
import re

class LPDetector():
    # Class constructor, the values are setted
    def __init__(self, weights="", image="", nameClass = []):
        self.weightsPath = weights
        self.imagePath = image
        self.image = 0
        if os.path.isfile(self.weightsPath):
            self.loadModel(self)
        if os.path.isfile(self.imagePath):
            self.image = cv2.imread(self.imagePath)
        self.nameClass = nameClass
        # Detection selector variable
        self.detectionType = "Yolo"
        # Result detections
        self.boxes = []
        self.scores = []
        self.classes = []
        self.validDetections = []
        # Result images
        self.imageResults = []
        self.LPimage = []
        # Result License plate number
        self.licenseNumbers= []
        
    def getWeights(self):
        return self.weightsPath

    def getImage(self): 
        return self.imagePath

    def getImageRead(self):
        return self.image

    def setWeights(self, weights):
        if (os.path.isfile(weights)):
            self.weightsPath = weights

    def setImagePath(self, imagePath):
        if (os.path.isfile(imagePath)):
            self.imagePath = imagePath
            self.image = cv2.imread(self.imagePath)

    def setImage (self, image):
        self.image = image.copy()

    def configuration (self):
        # Some configuration parameters are loaded
        self.strides = np.array(cfg.YOLO.STRIDES)
        anch = np.array(cfg.YOLO.ANCHORS)
        self.anchors = anch.reshape(3,3,2)
        self.xyscale = cfg.YOLO.XYSCALE
        if len(self.nameClass)>0:
            self.numClass = len(self.nameClass)
    
    def loadModel (self):
        # The model files are loaded
        self.configuration()
        self.model = tf.saved_model.load(self.weightsPath, tags=[tag_constants.SERVING])

    # Function for detecting LP using Yolo algorithm
    def yoloDetection(self, input_size = 416):
        imageCopy = self.image.copy()
        try:
            imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)
        except:
            pass
        imageCopy = cv2.resize(imageCopy, (input_size, input_size))

        image_data = imageCopy/255.
        imagesData = []
        imagesData.append(image_data)
        imagesData = np.asarray(imagesData).astype(np.float32)
        infer = self.model.signatures['serving_default']
        batchData = tf.constant(imagesData)
        pred_bbox = infer(batchData)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4] # bounding box coordinates
            pred_conf = value[:, :, 4:] # confianza del bounding box 

        # run non max suppression on detections
        self.boxes, self.scores, self.classes, self.validDetections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.49,
            score_threshold=0.50)

        boundingBoxes = self.boxes.numpy()[0]
        height, width, channels = self.image.shape
        for box in boundingBoxes:
            ymin = int(box[0] * height)
            xmin = int(box[1] * width)
            ymax = int(box[2] * height)
            xmax = int(box[3] * width)
            box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax

        pred_bbox = [boundingBoxes, self.scores.numpy()[0], self.classes.numpy()[0], self.validDetections.numpy()[0]]
        for object in range (pred_bbox[3]):
            if int(pred_bbox[2][object])>= 0 and int(pred_bbox[2][object])<len(self.nameClass):
                boundBox = pred_bbox[0][object]
                xmin, ymin, xmax, ymax = int(boundBox[0]), int(boundBox[1]), int(boundBox[2]), int(boundBox[3])
        # The license plate image is saved for the character recognition
        self.LPimage = self.image[ymin:ymax,xmin:xmax]
        return xmin, ymin, xmax-xmin, ymax-ymin
    
    # Function for detecting LP using an edge detection algorithm
    def edgeDetection(self):
        imageCopy = self.image.copy()
        areaImage = np.size(imageCopy)/3

        # Constants for detection
        MIN_LENGTH = 200
        MIN_AREA = 200
        MAX_LENGTH = 4000
        MAX_AREA = 4000
        MAX_RECTANGULARITY = 2
        temp = []

        imgGray = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2GRAY)

        # Contours are detected
        ret, thresh = cv2.threshold(imgGray, 100, 255, 0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # The contours are filtered, for selecting the LP bounding one
        j=0; 
        for i in range(len(contours)):
            perimeter = cv2.arcLength(contours[i],1)
            area = cv2.contourArea(contours[i])

            if perimeter == 0 or area == 0 :
                continue

            minRect = cv2.minAreaRect(contours[i])
            boundRect = cv2.boundingRect(contours[i])

            width = minRect[1][0]
            height = minRect[1][1]
            area1= width*height; 

            width2 = boundRect[2]
            height2 = boundRect[3]
            area2= width2*height2; 

            rectangularity1 = area1/area
            rectangularity2 = area2/area
            
            if ((perimeter > MIN_LENGTH) and (area > MIN_AREA) and (rectangularity1 < MAX_RECTANGULARITY) and (rectangularity2 < MAX_RECTANGULARITY) and (perimeter < MAX_LENGTH) and (area < MAX_AREA)):
                j=j+1; 
                temp.append(i)

        # If LP is not detected, -1 is returned
        if len(temp) == 0:
            return -1,-1,-1,-1

        # The bounding box is returned
        x,y,w,h = cv2.boundingRect(contours[temp[0]])
        self.LPimage = self.image[y:y+h,x:x+w]
        return x,y,w,h
  
    def getDetections (self):
        return self.boxes, self.scores, self.classes, self.validDetections
    
    def predictionDone (self):
        pred = True
        if len(self.boxes) == 0 or len(self.scores) == 0 or len(self.classes) == 0 or len(self.validDetections) == 0:
            pred = False
        return pred

    # The bounding box is returned
    def characterRecognition (self):
        LPnum = ""
        self.licenseNumbers = ""
        # The image is resized, changed to gray scale and blurred
        lpGray = self.LPimage.copy()
        lpGray = cv2.cvtColor(lpGray, cv2.COLOR_RGB2GRAY)
        lpGray = cv2.resize(lpGray, None, fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
        lpBlur = cv2.GaussianBlur(lpGray, (5,5), 0)
        lpGray = cv2.medianBlur(lpGray, 3)

        # The image is umbralized
        ret, lpUmbralized = cv2.threshold(lpBlur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        # A dilated element is created
        dilatElement = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        lpDilated = cv2.dilate(lpUmbralized, dilatElement, iterations=1)

        # Contours are detected
        contours, hier = cv2.findContours(lpDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        countSorted = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        im2 = lpGray.copy()
        # The character contours are filtered
        for count in countSorted:
            boundBox = cv2.boundingRect(count)
            x, y, w, h = boundBox
            height, width = im2.shape
            if height / float(h) <= 6:
                correlation = h/float(w)
                if correlation >= 1.5:
                    if width / float(w) <= 15:
                        area = w * h
                        if area >= 100:
                            regionInterest = lpUmbralized[y-5:y+h+5, x-5:x+w+5]
                            regionInterest = cv2.bitwise_not(regionInterest)
                            regionInterest = cv2.medianBlur(regionInterest, 5)
                            # Each character is detected and added to the LPnum variable
                            try:
                                txt = pytesseract.image_to_string(regionInterest, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                                cleanText = re.sub('[\W_]+', '', txt)
                                LPnum += cleanText[0]
                            except:
                                LPnum +=""
        self.licenseNumbers += LPnum
        return self.licenseNumbers