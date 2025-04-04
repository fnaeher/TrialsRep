"""
Created on Tue April 01 23:08:00 2025
@author: Akhyar.Ahmed

"""

import pandas as pd
import re


df_components = pd.read_csv('../Data/Components-Data pull for Dr. Stern Feb 2025.csv')
models = df_components['Model'].tolist()

with open('../Data/DHTlist_old.txt', 'r') as file:
    text = file.read()

flags = re.findall(r'"(.*?)"', text)

flags = list(dict.fromkeys(flags))

combined = models + flags

combined_unique = list(dict.fromkeys(combined))

df_combined = pd.DataFrame(combined_unique, columns=['combined_flag'])

df_combined.to_csv('../Data/combined_flags.csv', index=False)
print(df_combined.head())
print(df_combined.shape)

# df = pd.read_csv('../Data/Flagged_DHT.csv')

# print((100/df.shape[0])*df[df['DHT']==1].shape[0])
# print(df.shape[0])

# print(df[df['DHT']==1].shape[0])