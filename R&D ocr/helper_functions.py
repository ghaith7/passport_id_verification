import cv2
import imutils
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from deskew import determine_skew
import face_recognition
import time


def face_detect_rotate(image):
    trials = 0
    while(trials<4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(gray)
        if(len(face_locations)>0):
            main_face = max(face_locations,key = lambda x : abs(x[0] - x[2]) * abs(x[1] - x[3]))
            leftX = main_face[1]
            return (leftX,image)
        else : 
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) 
            trials += 1
    return 0
def traditional_processing(image,kw,kh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    gray = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(gray)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold(grad, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.dilate(thresh, None, iterations=3)
    return thresh
def show(image):
    fig = plt.figure(figsize=(10, 10))
    rows = 1
    columns = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)
    plt.title("image")
    plt.axis('off')
def show2(im1,im2):
    fig = plt.figure(figsize=(15, 10))
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(im1)
    plt.title("image1")
    plt.axis('off')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(im2)
    plt.title("image1")
    plt.axis('off')