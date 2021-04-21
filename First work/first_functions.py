# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:10:37 2021

@author: User
"""

import pandas as pd
import copy
#from sklearn import preprocessing

def normalize_val_aro_tocresc01(df,orig_min,orig_max,orig_orientation):
    print(df)
    df_c = copy.deepcopy(df)
    to_norm = df_c[["Valence","Arousal"]]
    print(to_norm)
    if orig_orientation == "decrescent":
        #Here the values get mirrored to be crescent instead of decrescent
        to_norm = (to_norm-(orig_max+orig_min)).abs()
    print(to_norm)
    normalized_columns = (to_norm-orig_min)/(orig_max-orig_min)
    df_c[["Valence","Arousal"]] = normalized_columns
    return(df_c)
    
    
    