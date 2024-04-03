from django.apps import AppConfig
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from easyocr import Reader
import tensorflow as tf

class ModelLoader():
    def __init__(self) :
        self.readerLatin = Reader(["en"], gpu=True)
        self.readerArabic = Reader(["ar"], gpu=True)
        self.readermix = Reader(["ar","en"], gpu=True)
        self.svm_bar_classifier = pickle.load(open("api/models/svm_bar_classifier", 'rb'))
        self.glareCNN = tf.keras.models.load_model('api/models/glare/glare_detect.h5')
        print("models have been loaded")
        self.locations = {
            "passport ID" : ( 0.5 , 0 , 0.75 , 0.1 ) ,
            "arabic name" : ( 0.5 , 0.1 , 1 , 0.22 ) ,
            "surname"     : ( 0 , 0.22 , 0.5 , 0.33 ) ,
            "name"        : ( 0 , 0.33 , 0.5 , 0.43 ) ,
            "job"         : ( 0.5 , 0.43 , 1 , 0.55 ) ,
            "nationality" : ( 0 , 0.43 , 0.5 , 0.55 ) ,
            "birth date"  : ( 0 , 0.55 , 0.5 , 0.65 ) ,
            "national ID" : ( 0.5 , 0.55 , 0.75 , 0.65 ) ,
            "adresse"     : ( 0.75 , 0.55 , 1 , 0.65 ) ,
            "sex"         : ( 0 , 0.65 , 0.15 , 0.76 ) ,
            "birth place" : ( 0.15 , 0.65 , 1 , 0.76 ) ,
            "issue date"  : ( 0 , 0.76 , 0.5 , 0.87 ) ,
            "issue auth"  : ( 0.5 , 0.76 , 1 , 0.87 ) ,
            "expr date"   : ( 0 , 0.87 , 0.5 , 1 ) ,
            "signature"   : ( 0.5 , 0.87 , 1 , 1 ) ,
        }
        
class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    model_loader = ModelLoader()