# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:08:03 2021

@author: User
"""



import os

from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import numpy as np

import pandas as pd

import seaborn as sns

from pylab import rcParams

import matplotlib.pyplot as plt

from matplotlib import rc

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from collections import defaultdict

from textwrap import wrap

from torch import nn, optim

import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

#%%
os.chdir("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/First Work")

#%%
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case = True)
#%%

#POTENCIAIS FONTES DE PROBLEMA:
    #O TOKENIZER NAO TEM NADA DE ADD_SPECIAL_TOKENS
    #ACRESCENTEI O DO_LOWER_CASE NO TOKENIZER, MAS O DEFAULT JÁ É TRUE

#EXPERIMENTAR:
    # MUDAR O NOME DE LABELS1 E LABELS2 PARA LABELS_VAL E LABELS_ARO

class MyDataset(Dataset):
    def __init__(self, filename,maxlen):
        data = pd.read_csv(filename)
        self.maxlen = maxlen
        self.texts = data["Word/Sentence"].tolist()
        self.labels1 = data['Valence'].tolist()
        self.labels2 = data['Arousal'].tolist()
    def __getitem__(self, idx):
        item = { }
        aux = tokenizer(self.texts[idx], max_length=self.maxlen, truncation=True, padding='max_length')
        item['input_ids'] = torch.tensor(aux['input_ids'])
        item['attention_mask'] = torch.tensor(aux['attention_mask'])
        item['labels'] = torch.tensor( [ self.labels1[idx], self.labels2[idx] ] )
        return item
    def __len__(self):
        return len(self.texts)
    
    
#%%
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        predictions = outputs[0]
        predictions = torch.sigmoid(predictions)
        loss = torch.nn.MSELoss()
        return loss(predictions.view(-1), labels.view(-1))
    

#%%
full_dataset = MyDataset("First_Dataset.csv",80)
#%%
print(len(full_dataset))
#%%
n = 0
train_len = int(0.8*len(full_dataset))
valid_len = int(0.1*len(full_dataset))
test_len = int(0.1*len(full_dataset))

while sum([train_len,valid_len,test_len]) != len(full_dataset):
    if sum([train_len,valid_len,test_len]) > len(full_dataset):
        train_len -=1
    else:
        train_len +=1
#%%
train_dataset, val_dataset, test_dataset = random_split(full_dataset,[train_len,test_len,valid_len])
#%%
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=2,   # batch size per device during training
    per_device_eval_batch_size=2,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10
)

trainer = MyTrainer(
    model=model,                         # the instantiated :hugging_face: Transformers model to be traine
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
)

#%%
trainer.train()