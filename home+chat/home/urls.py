from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name = 'home'),
    path('diary/', views.diary, name = 'diary'),
    path('chatbot/', views.chatbot, name = 'chatbot'),
    path('chat-response/', views.chat_response, name='chat_response'),
]