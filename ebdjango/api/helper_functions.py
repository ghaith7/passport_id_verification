"""_summary_
exiled functions 
includes
    improve_quality :      srcnn for image upscaling (a bit shitty) 
    drawROIs :             for trouble shouting in certain stages
    make_json :            links predictions with labels (to be modified)
"""

#from tensorflow.keras.models import load_model
import numpy as np
import cv2


def unblur(image):
    kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
    image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image

def increase_contrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img
def show_grid(test):
    h,w = test.shape[:2]
    grid = test.copy()
    horiz = [0.15,0.5,0.75]
    vertc = [0.1,0.22,0.33,0.43,0.55,0.65,0.76,0.87]
    for i in horiz:
        y = int( i * w )
        cv2.line(grid,(y,0),(y,h),(255,0,0),5)
    for i in vertc:
        x = int( i * h )
        cv2.line(grid,(0,x),(w,x),(255,0,0),5)
    return grid

def drawROIs(image,boxes):
    im = image.copy()
    for box in boxes:
        (startX, startY, endX, endY) = box
        cv2.rectangle(im,(startX, startY),(endX, endY),(255,0,0),2)
    return im

def make_json(predictions,labels):
    res = {}
    for i in range(len(labels)):
        res[labels[i]]= predictions[i][0].split("\n")[0]
    return res


