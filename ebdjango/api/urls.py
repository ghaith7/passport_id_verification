from django.urls import path
from . import views 

urlpatterns = [
    path('information_extraction/', views.information_extraction,name = "information_extraction"),
    path('face_identification/', views.face_identification,name = "face_identification")
]