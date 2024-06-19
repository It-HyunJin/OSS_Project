from django.db import models
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Create your models here.

# 모델 로드
model_encoder = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 예제 문장 임베딩
sentences = ["안녕하세요?", "한국어 문장 임베딩을 위한 버트 모델입니다."]
embeddings = model_encoder.encode(sentences)

# 데이터 로드
df = pd.read_csv('/content/diary_project/wellness_dataset.csv')

# 임베딩 생성
df['embedding'] = df['유저'].map(lambda x: list(model_encoder.encode(x)))

def chat(user, history=[]):
    embedding = model_encoder.encode(user)
    df['similarity'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['similarity'].idxmax()]

    history.append([user, answer['챗봇']])
    return history
