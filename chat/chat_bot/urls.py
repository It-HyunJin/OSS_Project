from django.urls import path
from . import views

urlpatterns = [
  path('', views.home, name = 'home'),
  path('chat-response/', views.chat_response, name='chat_response'),
]
