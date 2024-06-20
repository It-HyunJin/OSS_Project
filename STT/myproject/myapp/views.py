from django.shortcuts import render

from django.http import JsonResponse
from google.cloud import speech
import io

def speech_to_text(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']

        client = speech.SpeechClient()

        audio = speech.RecognitionAudio(content=audio_file.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US'
        )

        response = client.recognize(config=config, audio=audio)

        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        return JsonResponse({'transcript': transcript})
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)

def upload_audio(request):
    return render(request, 'myapp/templates/index.html')