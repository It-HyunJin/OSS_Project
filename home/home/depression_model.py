import os
import torch
import kss
import gluonnlp as nlp
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel, BertTokenizer
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from django.conf import settings
from pathlib import Path

# BERTClassifier 클래스 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=6, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooler = self.bert(input_ids=input_ids, token_type_ids=token_type_ids.long(),
                              attention_mask=attention_mask.float().to(input_ids.device))
        if self.dr_rate:
            pooler = self.dropout(pooler)
        return self.classifier(pooler)


# 모델과 토크나이저 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
model1 = BERTClassifier(bertmodel,  dr_rate = 0.5)

# 모델 가중치 로드
model_path = settings.BASE_DIR / 'home' / '6emotions_model_state_dict.pt'
model1.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)
model1.eval()

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        self.sentences = [bert_tokenizer(dataset[i][sent_idx], padding='max_length', max_length=max_len, truncation=True) for i in range(len(dataset))]
        self.labels = [np.int32(dataset[i][label_idx]) for i in range(len(dataset))]

    def __getitem__(self, i):
        input_ids = torch.tensor(self.sentences[i]['input_ids'])
        attention_mask = torch.tensor(self.sentences[i]['attention_mask'])
        token_type_ids = torch.tensor(self.sentences[i]['token_type_ids'])
        label = self.labels[i]
        return (input_ids, attention_mask, token_type_ids, label)

    def __len__(self):
        return len(self.labels)

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, None, 64, True, False)  # 토큰화한 문장
    test_dataloader = DataLoader(another_test, batch_size=1, num_workers=0)  # torch 형식 변환

    model1.eval()

    test_eval = []
    for batch_id, (input_ids, attention_mask, token_type_ids, label) in enumerate(test_dataloader):
      input_ids = input_ids.long().to(device)
      attention_mask = attention_mask.long().to(device)
      token_type_ids = token_type_ids.long().to(device)

      with torch.no_grad():
        out = model1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

      logits = out.squeeze(0)  # batch dimension 제거
      logits = logits.detach().cpu().numpy()

      if np.argmax(logits) == 0:
        test_eval.append("기쁨 혹은 중립이")
      elif np.argmax(logits) == 1:
        test_eval.append("우울이")
      elif np.argmax(logits) == 2:
        test_eval.append("공포가")
      elif np.argmax(logits) == 3:
        test_eval.append("놀람이")
      elif np.argmax(logits) == 4:
        test_eval.append("분노가")
      elif np.argmax(logits) == 5:
        test_eval.append("불안이")

# test_eval 리스트에 예측된 감정이 모두 저장되어 있음
    predicted_emotion = test_eval[0]  # 예시로 첫 번째 예측 결과를 가져옴
    print(">> 입력하신 내용에서 " + predicted_emotion + " 느껴집니다.")
    return np.argmax(logits)  # 마지막 로짓스 값을 반환할지는 사용 환경에 따라 다를 수 있음



def depression_predict(sentence):
    text = list()
    depression = 0
    for sent in kss.split_sentences(sentence):
        print(sent)
        text.append(sent)
        i = predict(sent)
        if i == 1:
            depression += 1
        elif i in [5, 4, 3]:
            depression += 0.3
    print(str(round(depression / len(text) * 100, 1)) + "%로 우울합니다.")
    return round(depression / len(text) * 100, 1)

