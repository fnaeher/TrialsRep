# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:33:21 2024

@author: AnatolFiete.Naeher
"""

import os
import re
import pandas as pd
from dht_flag import FlagDHT



def combine_texts(series):
    return ';\n '.join(series.dropna().unique())


d_CTD = {}

                   
d_CTD['flag'] = list(pd.read_csv("../Data/combined_flags.csv", encoding="utf-8")['combined_flag'])


d_CTD['CTD_6'] = pd.read_csv('../Data/CTD_6.csv')

data = d_CTD['CTD_6'].drop(columns = ['intervention_type','group_type'])

data = pd.DataFrame(data)

data = data.groupby('nct_id').agg(combine_texts).reset_index()

flags = d_CTD['flag']
columns = ['official_title', 'name', 'q_7_desc',
       'title', 'q_8_desc', 'measure', 'q_9_desc', 'q_10_desc']
dummies = 'No'
flagger = FlagDHT()

D = flagger.DHT_search(data, flags, columns,dummies)

print(D.head())
print(D.shape)
D.to_csv('../Data/Flagged_DHT.csv')
