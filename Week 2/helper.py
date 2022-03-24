
import os
import sys
import pandas as pd
import numpy as np 
import torch
import random

# seed
seed = 7777
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# device type
def set_device():
  if torch.cuda.is_available():
      device = torch.device("cuda")
      print(f"# available GPUs : {torch.cuda.device_count()}")
      print(f"GPU name : {torch.cuda.get_device_name()}")
  else:
      device = torch.device("cpu")
  print(device)

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split

class CustomDataset(Dataset):
  """
  - input_data: list of string
  - target_data: list of int
  """

  def __init__(self, input_data:list, target_data:list) -> None:
      self.X = input_data #input data X
      self.Y = target_data #target data Y

  def __len__(self):
      return len(self.Y) 

  def __getitem__(self, index):
      return self.X[index], self.Y[index]

!pip install transformers
from transformers import BertTokenizer, BertModel

tokenizer_bert = BertTokenizer.from_pretrained("klue/bert-base")

def custom_collate_fn(batch):
  """
  - batch: list of tuples (input_data(string), target_data(int))
  
  한 배치 내 문장들을 tokenizing 한 후 텐서로 변환함. 
  이때, dynamic padding (즉, 같은 배치 내 토큰의 개수가 동일할 수 있도록, 부족한 문장에 [PAD] 토큰을 추가하는 작업)을 적용
  토큰 개수는 배치 내 가장 긴 문장으로 해야함.
  또한 최대 길이를 넘는 문장은 최대 길이 이후의 토큰을 제거하도록 해야 함
  토크나이즈된 결과 값은 텐서 형태로 반환하도록 해야 함
  
  한 배치 내 레이블(target)은 텐서화 함.
  
  (input, target) 튜플 형태를 반환.
  """
  global tokenizer_bert

  input_list, target_list = [i for i,x in batch], [x for i,x in batch]
  tensorized_input = tokenizer_bert(input_list, padding='longest', truncation=True, return_tensors='pt') # text=input_list, pad가장 긴 문장에 맞춰서, truncation 최대길이 넘으면 제거, return_tensors tensor로 반환
  
  tensorized_label = torch.tensor(target_list)
  
  return tensorized_input, tensorized_label

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

def train(model, data_loader):
  global loss_fct

  # 배치 단위 평균 loss와 총 평균 loss 계산하기위해 변수 생성
  total_loss, batch_loss, batch_count = 0,0,0
  
  # model을 train 모드로 설정 & device 할당
  model.train()
  model.to(device)
  
  # data iterator를 돌면서 하나씩 학습
  for step, batch in enumerate(data_loader):
      batch_count+=1
      
      # tensor 연산 전, 각 tensor에 device 할당
      batch = tuple(item.to(device) for item in batch)
      
      batch_input, batch_label = batch
      
      # batch마다 모델이 갖고 있는 기존 gradient를 초기화
      model.zero_grad()
      
      # forward
      logits = model(**batch_input)
      
      # loss
      loss = loss_fct(logits, batch_label)
      batch_loss += loss.item()
      total_loss += loss.item()
      
      # backward -> 파라미터의 미분(gradient)를 자동으로 계산
      loss.backward()
      
      # optimizer 업데이트
      optimizer.step()
      
      # 배치 10개씩 처리할 때마다 평균 loss를 출력
      if (step % 10 == 0 and step != 0):
          print(f"Step : {step}, Avg Loss : {batch_loss / batch_count:.4f}")
          
          # 변수 초기화 
          batch_loss, batch_count = 0,0
  
  print(f"Mean Loss : {total_loss/(step+1):.4f}")
  print("Train Finished")

"""### 지금까지 구현한 함수와 클래스를 모두 불러와 `train()` 함수를 실행하자
- fine-tuning 모델 클래스 (`CustomClassifier`)
    - hidden_size = 768
    - n_label = 2
- 데이터 이터레이터 함수 (`data_iterator`)
    - batch_size = 32
- loss 
    - `CrossEntropyLoss()`
- optimizer
    - `AdamW()`
    - lr = 2e-5

"""

# Week2-2에서 구현한 클래스와 동일

class CustomClassifier(nn.Module):

  def __init__(self, hidden_size: int, n_label: int):
    super(CustomClassifier, self).__init__()
  
    self.bert = BertModel.from_pretrained("klue/bert-base")

    for param in self.bert.parameters():
      param.requires_grad==False

    dropout_rate = 0.1
    linear_layer_hidden_size = 32

    self.classifier = nn.Sequential(
        nn.Linear(hidden_size, linear_layer_hidden_size),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(linear_layer_hidden_size,2),
        ) 
    
  def forward(self, input_ids=None, attention_mask=True, token_type_ids=True):
    
    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
        )
    
    cls_token_last_hidden_states = outputs['pooler_output']

    logits = self.classifier(cls_token_last_hidden_states)

    return logits