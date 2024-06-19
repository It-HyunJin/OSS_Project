from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
from .models import chat
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

@csrf_exempt
def chat_response(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        history = request.session.get('history', [])
        history = chat(user_input, history)
        request.session['history'] = history
        return JsonResponse({'response': history[-1][1], 'history': history})
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def home(request):

  return render(request, 'index.html')
