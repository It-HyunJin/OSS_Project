from django.urls import path
from . import views

urlpatterns = [
    path('index/', views.upload_audio, name='upload_audio'),
    path('speech-to-text/', views.speech_to_text, name='speech_to_text'),
]