from django.shortcuts import render
from .depression_model import depression_predict
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def diary(request):
    prediction = None
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            prediction = depression_predict(user_input)

    context = {'prediction': prediction}
    return render(request, 'diary.html', context)

