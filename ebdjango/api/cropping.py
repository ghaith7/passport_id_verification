"""_summary_
functions for zooming into the area of intersest 
"""

import imutils
import face_recognition
import cv2
import numpy as np
from difflib import SequenceMatcher
from .helper_functions import *
from .apps import ApiConfig 

def face_detect_rotate(image):
    trials = 0
    while(trials<4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(gray)
        if(len(face_locations)>0):
            main_face = max(face_locations,key = lambda x : abs(x[0] - x[2]) * abs(x[1] - x[3]))
            right= main_face[1]
            bottom = main_face[2]
            return (right,bottom,image)
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

def top_slash_coordinates(image,boxes):
    boxes = sorted(boxes,key=lambda x : x[1])
    message = "REPUBLIC OF TUNISIA"
    for box in boxes:
        (startX, startY, endX, endY) = box
        clip = image[startY:endY,startX:endX]
        results = ApiConfig.model_loader.readerLatin.readtext(clip)
        texts = [res[1] for res in results]
        text="".join(texts)
        if SequenceMatcher(None, text, message).ratio()>0.5:
            return endY
    return 0

def extract_bar(image,kw,kh,delim,left):
    original = image.copy()
    image = traditional_processing(image,kw,kh)
    im_h,im_w = image.shape[:2]
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = [cv2.boundingRect(c) for c in cnts]
    cnts = [(x,y,x+w,y+h) for (x, y, w, h) in cnts]
    bar_position = [x for x in cnts if x[3] >delim and left<x[0]and (x[2]-x[0])> 3 * (x[3]-x[1])]
    cnts = sorted(cnts,key = lambda x : x[1])
    for c in bar_position:
        (xstart,ystart,xend,yend)= c
        clip = original[ystart:yend,xstart:xend]
        results = ApiConfig.model_loader.readerLatin.readtext(clip)
        texts = [res[1] for res in results]
        text=" ".join(texts)
        if text == "":
            clip = cv2.resize(clip, (300,70), interpolation = cv2.INTER_AREA)
            clip = clip.flatten()
            probs = ApiConfig.model_loader.svm_bar_classifier.predict_proba([clip])
            if probs[0][0]>probs[0][1]+0.5:
                break
    top = top_slash_coordinates(original,cnts)
    return (ystart,top)
def boundary_check(original,cnts):
    text = ""
    while(text==""):
        (xstart,ystart,xend,yend) = cnts[0]
        clip = original[ystart:yend,xstart:xend]
        clip = unblur(clip)
        results = ApiConfig.model_loader.readerLatin.readtext(clip)
        results = results + ApiConfig.model_loader.readerArabic.readtext(clip)
        texts = [res[1] for res in results]
        text = "".join(texts)
        if text == "" or "<" in text:
            cnts.remove(cnts[0])
    return cnts
def adjust(image):
    image = unblur(image)
    original = image.copy()
    
    image = traditional_processing(image,40,8)
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = [cv2.boundingRect(c) for c in cnts]
    cnts = [(x,y,x+w,y+h) for (x, y, w, h) in cnts if w*h > 50*10*4 and w>h]
    
    #cnts = fetch_text(original,cnts)
    cnts = sorted(cnts , key = lambda x : x[0])
    cnts = boundary_check(original,cnts)
    
    cnts = sorted(cnts , key = lambda x : x[3],reverse = True)
    cnts = boundary_check(original,cnts)
    
    cnts = sorted(cnts , key = lambda x : x[1])
    cnts = boundary_check(original,cnts)
    
    (h,w) = image.shape[:2]
    buffer = 10
    top = max([min(cnts , key = lambda x : x[1])[1]-buffer,0])
    bottom = min([max(cnts , key = lambda x : x[3])[3]+buffer,h])
    right = min([max(cnts , key = lambda x : x[2])[2]+buffer,w])
    left = max([min(cnts , key = lambda x : x[0])[0]-buffer,0])
    
    final = original[top:bottom,left:right]
    enhanced = increase_contrast(final)
    return final

def close_up(im):
    original = im.copy()
    pre_adjust = im
    try :
            left,bottom,im=face_detect_rotate(im)
            try : 
                (bar,top) = extract_bar(im,40,8,bottom,left)
                im = im[top:bar,left:]
                (h,w) = im.shape[:2]
                pre_adjust = im
                if (h*w < 1024*1024):
                    im = cv2.resize(im,(1500,1000),interpolation = cv2.INTER_AREA)
                im = adjust(im)
            except Exception as e :
                print(e)
    except Exception as e:
        print(e)
    return (im,pre_adjust)