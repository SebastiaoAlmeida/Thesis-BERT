# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 12:46:48 2021

@author: User
"""

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

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

from torch.utils.data import Dataset, DataLoader

#%%

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%%
import os
print(os.getcwd())

os.chdir("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/First Work")
data = pd.read_csv("../Datasets/First Attempts/Raw/1_Ratings_Warriner_et_al.csv",index_col=0,usecols=["Word","V.Mean.Sum","A.Mean.Sum"])

#%%

print(data.head())

data = data.rename(columns={"V.Mean.Sum":"Valence","A.Mean.Sum":"Arousal"})

print(data.head())
print(data.shape)

#Checking for nuls
data.info()
#%%
