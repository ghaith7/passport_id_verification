from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
from rest_framework.decorators import api_view
import cv2
import numpy as np
import base64
import time
import json
from .ocr_process import *
from .cropping import *
from .helper_functions import *
from .glare_detection import *

DEBUG = True

@api_view(['POST'])
def information_extraction(request):
    start = time.time()
    image =  request.data["passport"]
    
    image = base64.b64decode(image)
    
    jpg_as_np = np.frombuffer(image, dtype=np.uint8)
    
    image = cv2.imdecode(jpg_as_np, flags=1)
    
    #cropping
    (im,pre_adjust) = close_up(image)
    pre_adjust = base64.b64encode(cv2.imencode('.jpg', pre_adjust)[1])
    close_up_image = base64.b64encode(cv2.imencode('.jpg', im)[1])
    grided = base64.b64encode(cv2.imencode('.jpg', show_grid(im))[1])
    #flash detection
    h,w = im.shape[:2]
    location_boxes = [(int(a*w),int(b*h),int(c*w),int(d*h))
                      for (a,b,c,d) in ApiConfig.model_loader.locations.values()]
    flash,viz = detect_flash(im,location_boxes)
    if flash == True:
        viz = base64.b64encode(cv2.imencode('.jpg', viz)[1])
        return HttpResponse([viz])
    
    end = time.time()
    
    #reading
    print("elapsed time before ocr: ", end - start)
    start = time.time()
    
    results = text_results(im)
    results = json.dumps(results, indent=4,ensure_ascii = False)
    results = {
    "passport ID": "A000000",
    "arabic name": "غسان بن حبيب بحروني",
    "surname": "CITIZEN",
    "name": "JOHN",
    "job": "طالب",
    "nationality": "TUNISIAN/تونسي",
    "birth date": "00-00-0000",
    "national ID": "00000000",
    "adresse": "KAIRouAn7القيروان",
    "sex": "Mذكر",
    "birth place": "HAFFOUZ/حفوز",
    "issue date": "00-00-0000",
    "issue auth": "00-00-0000",
    "expr date": "00-00-0000",
    "signature": "KAIRouAn7 Jyi Dmd"
    }
    print("ocr time : ", time.time() - start)
    
    
    
    if DEBUG == True : 
        response = [results,"/delim/", pre_adjust , "/delim/" , close_up_image , "/delim/",grided]
    else : 
        response = results
        
    return HttpResponse(response)






@api_view(['POST'])
def face_identification(request):
    #passport
    image1 =  request.data["passport"]
    image1 = base64.b64decode(image1)
    jpg_as_np = np.frombuffer(image1, dtype=np.uint8)
    image1 = cv2.imdecode(jpg_as_np, flags=1)
    
    #selfie
    image2 =  request.data["selfie"]
    image2 = base64.b64decode(image2)
    jpg_as_np = np.frombuffer(image2, dtype=np.uint8)
    image2 = cv2.imdecode(jpg_as_np, flags=1)
    
    
    return HttpResponse(True)