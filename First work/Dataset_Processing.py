# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:11:44 2021

@author: User
"""

import os

os.chdir("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/First Work")

from first_functions import normalize_val_aro_tocresc01

import pandas as pd

import ftfy

import nltk

import numpy as np

import random

from sklearn.model_selection import train_test_split

#%%
nas = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', 'NA', '', '#NA', 'NaN', '-NaN', '']
#%%

data1 = pd.read_csv("../Datasets/First Attempts/Raw/1.csv",usecols=["Word","V.Mean.Sum","A.Mean.Sum"],na_values=nas,keep_default_na=False)
data1 = data1.rename(columns={"V.Mean.Sum":"Valence","A.Mean.Sum":"Arousal","Word":"Word/Sentence"})
data1norm = normalize_val_aro_tocresc01(data1,1,9,"crescent")
#%%
data2 = pd.read_excel("../Datasets/First Attempts/Raw/2.xlsx",usecols=["Words","M V","M A"],na_values=nas,keep_default_na=False)
data2 = data2.rename(columns={"M V":"Valence","M A":"Arousal","Words":"Word/Sentence"})
data2norm = normalize_val_aro_tocresc01(data2,1,9,"crescent")
#%%
data3 = pd.read_excel("../Datasets/First Attempts/Raw/3.xlsx",usecols=["Polish sentence","Valence all","Arousal all"],skiprows=[1],na_values=nas,keep_default_na=False)
data3 = data3.rename(columns={"Valence all":"Valence","Arousal all":"Arousal","Polish sentence":"Word/Sentence"})
data3norm = normalize_val_aro_tocresc01(data3,1,9,"crescent")
#%%
data4 = pd.read_csv("../Datasets/First Attempts/Raw/4.csv",usecols=["Word","ValenceMean","ArousalMean"],na_values=nas,keep_default_na=False,encoding="utf-8")
data4 = data4.rename(columns={"ValenceMean":"Valence","ArousalMean":"Arousal","Word":"Word/Sentence"})
data4norm = normalize_val_aro_tocresc01(data4,1,9,"crescent")
data4norm["Word/Sentence"] = data4norm["Word/Sentence"].str.replace("[*]","")
#%%
data5 = pd.read_excel("../Datasets/First Attempts/Raw/5.xlsx",usecols=["polish word","Valence_M","arousal_M"],na_values=nas,keep_default_na=False)
data5 = data5.rename(columns={"Valence_M":"Valence","arousal_M":"Arousal","polish word":"Word/Sentence"})
data5norm = normalize_val_aro_tocresc01(data5,1,9,"crescent")
#%%
data6 = pd.read_csv("../Datasets/First Attempts/Raw/6.csv",na_values=nas,keep_default_na=False,encoding="utf-8")
data6["Anonymized Message"] = data6["Anonymized Message"].map(lambda a: ftfy.fix_text(str(a)))
data6["Valence"] = data6[["Valence1","Valence2"]].mean(axis=1)
data6["Arousal"] = data6[["Arousal1","Arousal2"]].mean(axis=1)
data6.drop(["Valence1","Valence2","Arousal1","Arousal2"],axis=1,inplace=True)
data6 = data6.rename(columns={"Anonymized Message":"Word/Sentence"})
#nltk.download("names")
People = nltk.corpus.names.words("male.txt")
Phone = ["202-555-0194","515-555-0144","843-555-0175","+1-617-555-0156","+1-916-555-0154","907-555-0161","0141 9496 0378","020 7946 0756","0117 7496 0205","+44 151 9496 0350","08 7010 2846","02 7010 0649","+61 3 7010 4939","06 55 908 947","+36 55 676 078"]
#Emails are removed, it's only one mention
#Addresses are removed, they are only 2
#URL will be removed. There is one row with only <URL>
data6["Word/Sentence"] = data6["Word/Sentence"].str.replace("<EMAIL>","")
data6["Word/Sentence"] = data6["Word/Sentence"].str.replace("<ADDRESS>","")
data6["Word/Sentence"] = data6["Word/Sentence"].str.replace("<URL>","")
data6 = data6[(data6["Word/Sentence"] != "")]
for i in range(len(data6["Word/Sentence"])):
    sent = data6.iloc[i,0]
    sent = sent.replace("<PERSON>",random.choice(People))
    sent = sent.replace("<PHONE>",random.choice(Phone))
    data6.iloc[i,0] = sent
data6norm = normalize_val_aro_tocresc01(data6,1,9,"crescent")
#%%
#data7 = pd.read_table("../Datasets/First Attempts/Raw/7.txt",sep="\t",na_values=nas,keep_default_na=False,encoding="utf-8")
#data7["Sentence"] = data7["Sentence"].map(lambda a: ftfy.fix_text(str(a)))

#%%
xl8 = pd.ExcelFile("../Datasets/First Attempts/Raw/8.xlsx")
sheet_names = xl8.sheet_names
data8 = xl8.parse(sheet_names[0])

for sheet_name in sheet_names[1:]:
    df = xl8.parse(sheet_name) 
    data8 = data8.append(df,ignore_index=True)

data8["Sentence"] = data8["Sentence"].map(lambda a: ftfy.fix_text(str(a)))
data8 = data8.rename(columns={"Sentence":"Word/Sentence"})
data8norm = normalize_val_aro_tocresc01(data8,1,9,"crescent")
#%%
data9 = pd.read_csv("../Datasets/First Attempts/Raw/9.csv",na_values=nas,usecols=["text","V","A"],keep_default_na=False,encoding="utf-8")
data9 = data9.rename(columns={"V":"Valence","A":"Arousal","text":"Word/Sentence"})
data9norm = normalize_val_aro_tocresc01(data9,1,5,"crescent")
#%%
data10 = pd.read_excel("../Datasets/First Attempts/Raw/10.xls",usecols=["EP-Word","Val-M","Arou-M"],na_values=nas,keep_default_na=False)
data10 = data10.rename(columns={"Val-M":"Valence","Arou-M":"Arousal","EP-Word":"Word/Sentence"})
data10norm = normalize_val_aro_tocresc01(data10,1,9,"crescent")
#%%
data11 = pd.read_excel("../Datasets/First Attempts/Raw/11.xlsx",usecols=["Sentence","Valence","Arousal"],na_values=nas,keep_default_na=False)
data11 = data11.rename(columns={"Sentence":"Word/Sentence"})
data11norm = normalize_val_aro_tocresc01(data11,1,9,"crescent")
#%%
data12 = pd.read_excel("../Datasets/First Attempts/Raw/12.xlsx",usecols=["French Words","valence Mean","arousal Mean"],na_values=nas,keep_default_na=False)
data12 = data12.rename(columns={"valence Mean":"Valence","arousal Mean":"Arousal","French Words":"Word/Sentence"})
data12norm = normalize_val_aro_tocresc01(data12,1,9,"crescent")
#%%
data13 = pd.read_csv("../Datasets/First Attempts/Raw/13.csv",na_values=nas,usecols=["Words","VAL M","AROU M"],keep_default_na=False,encoding="utf-8")
data13 = data13.rename(columns={"VAL M":"Valence","AROU M":"Arousal","Words":"Word/Sentence"})
data13norm = normalize_val_aro_tocresc01(data13,1,9,"crescent")
#%%
data14 = pd.read_table("../Datasets/First Attempts/Raw/14.txt",sep="\t",na_values=nas,usecols=["word","valence_mean","arousal_mean"],keep_default_na=False,encoding="ANSI")
data14 = data14.rename(columns={"valence_mean":"Valence","arousal_mean":"Arousal","word":"Word/Sentence"})
data14norm = normalize_val_aro_tocresc01(data14,1,9,"crescent")
#%%
data15 = pd.read_excel("../Datasets/First Attempts/Raw/15.xlsx",usecols=["Ita_Word","M_Val","M_Aro"],na_values=nas,keep_default_na=False)
data15 = data15.rename(columns={"M_Val":"Valence","M_Aro":"Arousal","Ita_Word":"Word/Sentence"})
data15norm = normalize_val_aro_tocresc01(data15,1,9,"crescent")
#%%
#print(data1)
#print(data1norm)
#print(data2)
#print(data2norm)
#print(data3)
#print(data3norm)
#print(data4)
#print(data4norm)
#print(data5)
#print(data5norm)
#print(data6)
#print(data6norm)
#print(data7)
#print(data7.iloc[5:15,8])
#print(data8)
#print(data8norm)
#print(data9)
#print(data9norm)
#print(data10)
#print(data10norm)
#print(data11)
#print(data11norm)
#print(data12)
#print(data12norm)
#print(data13)
#print(data13norm)
#print(data14)
#print(data14norm)
#print(data15)
#print(data15norm)

processed_datasets = [data1norm,data2norm,data3norm,data4norm,data5norm,data6norm,
                      data8norm,data9norm,data10norm,data11norm,data12norm,data13norm,
                      data14norm,data15norm]
#%%
fold1, fold2 = train_test_split(data1norm, test_size=0.5)
for dataset in processed_datasets[1:]:
    dfold1, dfold2 = train_test_split(dataset, test_size=0.5)
    fold1 = fold1.append(dfold1,ignore_index=True)
    fold2 = fold2.append(dfold2,ignore_index=True)
    
#%%
print(fold1.shape)
print(fold2.shape)
print(fold1)
#%%
fold1 = fold1.sample(frac=1).reset_index(drop=True)
fold2 = fold2.sample(frac=1).reset_index(drop=True)

#%%
print(fold1.shape)
print(fold2.shape)
print(fold1)

#%%
fold1.to_csv("../Processed/fold1.csv", encoding="utf-8") 
fold2.to_csv("../Processed/fold2.csv", encoding="utf-8") 