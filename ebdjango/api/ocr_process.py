"""_summary_
function for reading with the right label
"""

from .helper_functions import *
from .apps import ApiConfig

def readValue(label,im,hybrid = False,arabic = False, date = False):
    h,w = im.shape[:2]
    ( x0 , y0 , x1 , y1 ) = ApiConfig.model_loader.locations[label]
    layout_box = ( int(x0*w) , int(y0*h) , int(x1*w) , int(y1*h) )
    (startX, startY , endX, endY) = layout_box
    clip = im[startY: endY,startX:endX]
    clip = increase_contrast(clip)
    if arabic == True :
        results = ApiConfig.model_loader.readerArabic.readtext(clip)
        results = sorted(results,key = lambda x : x[0][2][0],reverse = True)
        texts = [res[1] for res in results]
    else :
        if hybrid == True :
            texts = [res[1] for res in ApiConfig.model_loader.readermix.readtext(clip)]
            if len(texts) > 1 :
                texts = [t for t in texts if " / " in t]
        else :
            if date ==True:
                texts = [res[1] for res in ApiConfig.model_loader.readermix.readtext(clip)]
                texts = [t for t in texts if "-" in t and len(t)==10]
            else :
                texts = [res[1] for res in ApiConfig.model_loader.readerLatin.readtext(clip)]
                if label == "national ID":
                    texts = [t for t in texts if len(t)==8 and t.isnumeric()]
                if label == "passport ID":
                    texts = [t for t in texts if t[0].isupper() and t[1:7].isnumeric() ]
                if label in ["name","surname"]:
                    texts = [t for t in texts if t[0].isupper()]
    text = " ".join(texts)
    return text
def text_results(im):
    res = {}
    l = ApiConfig.model_loader.locations
    for label in l:
        if label in ["arabic name","job","adresse"]:
            text = readValue(label,im,arabic = True)
        else : 
            if label in ["nationality","sex","birth place","issue auth"]:
                text = readValue(label,im,hybrid = True)
            else : 
                if label in ["birth date","issue date","expr date"]:
                    text = readValue(label,im,date = True)
                else : 
                    text = readValue(label,im)
        res[label] = text
    return res