from django.http import JsonResponse
from django.shortcuts import render
from . import models
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

def home(request):
  
  return render(request, 'home.html')
  
def diary(request):
  
  return render(request, 'diary.html')
  
def chatbot(request):
  
  return render(request, 'chatbot.html')

@csrf_exempt
def chat_response(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        history = request.session.get('history', [])
        history = models.chat(user_input, history)
        request.session['history'] = history
        return JsonResponse({'response': history[-1][1], 'history': history})
    return JsonResponse({'error': 'Invalid request method'}, status=400)