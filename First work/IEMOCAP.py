# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:42:51 2021

@author: User
"""

import os

os.chdir("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/Datasets/First Attempts/Raw/IEMOCAP_full_release/IEMOCAP_full_release")

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
#%%
files = ["Session1","Session2","Session3","Session4","Session5"]

#%%
def process_scores(filename):
    df = pd.read_table(filename,sep="\n\n",names=["Col"])
    df = df[df["Col"].str.startswith("[")]
    df[["Timing","id","categorical","Scores"]] = df["Col"].str.split("\t",expand=True)
    df = df[["id","Scores"]]
    df["Scores"] = df["Scores"].str.strip("[]").str.split(",")
    df["Scores"] = df.Scores.apply(lambda s: [float(x.strip(" ")) for x in s])
    df[["Valence","Arousal","Dominance"]] = pd.DataFrame(df.Scores.tolist(),index=df.index)
    df.drop(columns = ["Scores","Dominance"],inplace=True)
    return(df)
#%%
def process_lines(filename):
    df = pd.read_table(filename,sep="\n\n",names=["Col"])
    df[["id","timing","sentence"]] = df["Col"].str.split(" ",2,expand=True)
    df = df[["id","sentence"]]
    return(df)

#%%
scores_list = []
for filename in files:
    os.chdir("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/Datasets/First Attempts/Raw/IEMOCAP_full_release/IEMOCAP_full_release/"+filename+"/dialog/EmoEvaluation/")
    for filename2 in os.listdir("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/Datasets/First Attempts/Raw/IEMOCAP_full_release/IEMOCAP_full_release/"+filename+"/dialog/EmoEvaluation/"):
        if filename2.endswith(".txt"):
            df = process_scores(filename2)
            scores_list.append(df)            
#%%
scores_df = pd.concat(scores_list,ignore_index = True)
#%%
print(scores_df)
#%%
lines_list = []
for filename in files:
    os.chdir("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/Datasets/First Attempts/Raw/IEMOCAP_full_release/IEMOCAP_full_release/"+filename+"/dialog/transcriptions/")
    for filename2 in os.listdir("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/Datasets/First Attempts/Raw/IEMOCAP_full_release/IEMOCAP_full_release/"+filename+"/dialog/transcriptions/"):
        if filename2.endswith(".txt"):
            df = process_lines(filename2)
            lines_list.append(df)
            
#%%
lines_df = pd.concat(lines_list,ignore_index = True)
#%%
print(lines_df)
#%%
merged_df = pd.merge(left=lines_df,right=scores_df)
final_df = merged_df.drop(columns="id")
#%%
print(final_df)
print(final_df.Arousal)

#%%
final_df = final_df.groupby("sentence").mean().reset_index()

#%%
print(final_df)

#Passou de 10039 frases para 8068

#%%
final_df.to_csv("E:/Pastas dos Semestres MECD/2º Ano 2º Semestre/Tese/Datasets/First Attempts/Raw/16.csv",index = False,encoding="utf-8")
#%%
print(final_df.sentence.map(len).max())
#%%
plt.scatter(np.arange(0,len(final_df),1),final_df["sentence"].map(len).sort_values())

#%%
for t in [400,350,300,250,225,200,175,150,125,100]:
    print(sum(final_df["sentence"].map(len)>t))