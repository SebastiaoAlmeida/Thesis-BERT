# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:08:47 2021

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

#A normalização do dataset está no outro código


#PONTOS QUE PODEM VIR A SER NECESSÁRIOS PARA DEBUG:
    #- O Prof fazia tolist() nos dataframes das labels e das features

class MyData(Dataset):
    def __init__(self,filename,maxlen,test_size = 0.2,train=True):
        #Loading the dataset
        my_df = pd.read_csv("First_Dataset.csv")
        #Leaving only the word/sentence
        feature=my_df.drop(["Valence","Arousal"])
        #Setting the labels
        label
        