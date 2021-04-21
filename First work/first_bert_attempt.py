# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 12:46:48 2021

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



#%%
os.chdir("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/First Work")

from first_functions import normalize_val_aro_tocresc01
#%%

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%%
nas = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', 'NA', '', '#NA', 'NaN', '-NaN', '']

os.chdir("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/First Work")
data1 = pd.read_csv("../Datasets/First Attempts/Raw/1_Ratings_Warriner_et_al (English Words).csv",usecols=["Word","V.Mean.Sum","A.Mean.Sum"],na_values=nas,keep_default_na=False)
data2 = pd.read_excel("../Datasets/First Attempts/Raw/3_ANPST_718_Dataset (Polish sentences).xlsx",usecols=["Polish sentence","Valence all","Arousal all"],skiprows=[1],na_values=nas,keep_default_na=False)
#%%

print(data1.head())

data1 = data1.rename(columns={"V.Mean.Sum":"Valence","A.Mean.Sum":"Arousal","Word":"Word/Sentence"})

print(data1.head())
print(data1.shape)

#Checking for nuls
data1.info()
#%%
print(data2.head())

data2 = data2.rename(columns={"Valence all":"Valence","Arousal all":"Arousal","Polish sentence":"Word/Sentence"})

print(data2.head())
print(data2.shape)

#Checking for nuls
data2.info()


#%%
norm1 = normalize_val_aro_tocresc01(data1,1,9,"crescent")
norm2 = normalize_val_aro_tocresc01(data2,1,9,"crescent")
print(norm1)
print(norm2)

data = norm1.append(norm2,ignore_index=True,)

#%%
data.to_csv("First_Dataset.csv",index=False)
#%%
print(data.info())
loaded_data= pd.read_csv("First_Dataset.csv")
print(loaded_data.info())
#%%
print(data.info())

print(data.iloc[600:620])
print(data.iloc[14600:14620])


#%%
print(data)
#%%
configuration = BertConfig()

model = BertModel(configuration)

print(model.config)
#%%
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#%%
sentences = data["Word/Sentence"].values
print(' Original: ', sentences[14600])
# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[14600]))
# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[14600])))

#%%
max_len = 0

# For every sentence...
for sent in sentences:

    #print (sent)
    #print(type(sent))
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)

#%%
#The maximum sentence length was 66. I'll use 80 for now, need to maybe increase if new
#datasets force it
MAX_LEN = 80
#%%
# Tokenize all of the sentences and map the tokens to their word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        truncation=True,
                        max_length = MAX_LEN,           # Pad & truncate all sentences.
                        padding="max_length",
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

#%%
# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
#labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])

#%%
print(type(input_ids))
print(type(attention_masks))
print(input_ids)

#%%
dataset = TensorDataset(input_ids, attention_masks, labels)

#%%
labels = data[["Valence","Arousal"]]
print(labels)
#%%
test = torch.cat(labels)