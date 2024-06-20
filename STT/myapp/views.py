from django.shortcuts import render
from django.http import JsonResponse
from google.cloud import speech
from pydub import AudioSegment
import io
import os

from myproject.settings import credentials

def index(request):
    return render(request, 'index.html')

def speech_to_text(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']
        file_name, file_extension = os.path.splitext(audio_file.name)

        # 업로드된 오디오 파일의 샘플 레이트 확인
        audio_file.seek(0)
        try:
            audio_segment = AudioSegment.from_file(audio_file)
            sample_rate = audio_segment.frame_rate
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

        # 인코딩 형식을 설정합니다.
        if file_extension.lower() == ".flac":
            encoding = speech.RecognitionConfig.AudioEncoding.FLAC
        elif file_extension.lower() == ".wav":
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        else:
            return JsonResponse({'error': 'Unsupported audio format'}, status=400)

        client = speech.SpeechClient(credentials=credentials)

        audio_file.seek(0)  # 파일 포인터를 처음으로 되돌립니다.
        audio_content = audio_file.read()
        audio = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=sample_rate,
            language_code='en-US'
        )

        response = client.recognize(config=config, audio=audio)

        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        return JsonResponse({'transcript': transcript})
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)