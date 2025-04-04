"""
Created on Sun April  3 13:17:21 2025

@author: Akhyar.Ahmed
"""

import os
import re
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Read the CTD CSV file.
ctd_df = pd.read_csv('../Data/Flagged_CTD_6_De_T_XTN_q10.csv')

# Remove duplicate rows based on nct_id.
ctd_df = ctd_df.drop_duplicates(subset=['nct_id'])

flag_both = 0
flag_keywords = 0
flag_exception = 0

# Count flags for each unique nct_id.
for i, row in tqdm(ctd_df.iterrows(), total=ctd_df.shape[0], desc="Processing rows"):
    if row['keyword_flag'] == 1:
        flag_keywords += 1
    if row['exception_flag'] == 1:
        flag_exception += 1
    if row['keyword_flag'] == 1 and row['exception_flag'] == 1:
        flag_both += 1

print(f'keyword found: {flag_keywords}')
print(f'exception found: {flag_exception}')
print(f'Both keyword and exception found: {flag_both}')

# Plot overall flag counts and save the plot.
labels = ['Keyword Found', 'Exception Found', 'Both Found']
counts = [flag_keywords, flag_exception, flag_both]

plt.figure(figsize=(8,6))
plt.bar(labels, counts, color=['blue', 'green', 'red'])
plt.xlabel('Flag Category')
plt.ylabel('Count')
plt.title('Overall Flags Count')
plt.savefig('../Data/overall_flags_count.png')
plt.close()

# Plot distribution of found keywords and save the plot.
# Filter out empty keyword values.
keyword_dist = ctd_df[ctd_df['found_keywords'] != ""]['found_keywords'].value_counts()

plt.figure(figsize=(10,6))
keyword_dist.plot(kind='bar', color='purple')
plt.xlabel('Keyword')
plt.ylabel('Frequency')
plt.title('Distribution of Found Keywords')
plt.savefig('../Data/keyword_distribution.png')
plt.close()
