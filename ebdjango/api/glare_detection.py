import numpy as np
import cv2
from skimage import measure
import imutils
from .apps import ApiConfig 
from .helper_functions import *

def pred(score):
    pred_class = "" 
    if score[0][0] <0.3: 
        pred_class = "Glare" 
    else:
        pred_class = "Not Glare" 
    return pred_class

def create_mask(image):
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    blurred = cv2.GaussianBlur( gray, (9,9), 0 )
    _,thresh_img = cv2.threshold( blurred, 245, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode( thresh_img, None, iterations=2 )
    thresh_img  = cv2.dilate( thresh_img, None, iterations=4 )
    labels = measure.label( thresh_img, background=0)
    mask = np.zeros( thresh_img.shape, dtype="uint8" )

    for label in np.unique( labels ):
        # if this is the background label, ignore it
        if label == 0:
            continue
        
        labelMask = np.zeros( thresh_img.shape, dtype="uint8" )
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero( labelMask )
        
        if numPixels > 300:
            mask = cv2.add( mask, labelMask )
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = [cv2.boundingRect(c) for c in cnts]
    cnts = [(x,y,x+w,y+h) for (x, y, w, h) in cnts]
    return cnts

def rcnn_glare(im,cnts):
    image = im.copy()
    boxes = []
    for c in cnts:
        (startX, startY, endX, endY) = c
        clip = image[startY:endY,startX:endX]
        clip = cv2.resize(clip, (64,64), interpolation = cv2.INTER_AREA)
        clip = np.expand_dims(clip, axis = 0) 
        result = ApiConfig.model_loader.glareCNN.predict(clip/255,verbose = 0)
        p = pred(result)
        if p == "Glare":
            boxes.append(c)
    return (boxes,image)


def area(x1,y1,x2,y2):
    return (x2-x1)*(y2-y1)


def iou(target,box):
    (sx1, sy1 , ex1, ey1) = target
    a1 = area(sx1, sy1 , ex1, ey1)
    (sx2, sy2 , ex2, ey2) = box
    a2 = area(sx2, sy2 , ex2, ey2)
    
    isx = max(sx1,sx2)
    isy = max(sy1,sy2)
    iex = min(ex1,ex2)
    iey = min(ey1,ey2)
    h = max(0 , (iex-isx) )
    w = max(0 , (iey-isy) )
    i = h*w
    
    u = a1 + a2 -i
    return i/u

def detect_flash(image,text_boxes):
    cnts = create_mask(image)
    flash_boxes,viz = rcnn_glare(image,cnts)
    inters = []
    flash = False
    for target in flash_boxes:
        for box in text_boxes:
            a = iou(target,box)
            if a > 0.1:
                inters.append(target)
                flash = True
    viz = drawROIs(viz,inters)
    return flash,viz