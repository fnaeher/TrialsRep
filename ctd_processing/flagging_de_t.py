# """
# Created on Sun April  3 13:17:21 2025

# @author: Akhyar.Ahmed
# """

# import os
# import re
# import pandas as pd
# from tqdm import tqdm
# from dht_flag import FlagDHT

# # Read the Excel file containing keywords and exception lists,
# # using the second row as the header.
# exceptions_df = pd.read_excel('../Data/tcp-32-41-s003.xls', engine='xlrd', header=1)
# print(exceptions_df.head())

# # Read the CTD CSV file.
# ctd_df = pd.read_csv('../Data/CTD_6.csv')

# # Define the columns to check for keyword matching.
# cols_to_check = ['nct_id', 'q_7_desc', 'q_10_desc']

# # Filter for the columns that actually exist in ctd_df.
# available_cols = [col for col in cols_to_check if col in ctd_df.columns]

# # Create new columns for the flags and found keywords/exceptions.
# ctd_df['keyword_flag'] = 0
# ctd_df['exception_flag'] = 0
# ctd_df['found_keywords'] = ""
# ctd_df['found_exceptions'] = ""

# # Build a list of tuples from exceptions_df: (keyword, [exception words])
# keyword_exceptions = []
# for i, row in exceptions_df.iterrows():
#     keyword = str(row['Keywords']).strip()
#     exception_list_raw = str(row['Exception list'])
#     exception_words = [ex.strip() for ex in exception_list_raw.split(',')]
#     keyword_exceptions.append((keyword, exception_words))

# # Iterate over each row in ctd_df with tqdm progress bar.
# for i, row in tqdm(ctd_df.iterrows(), total=ctd_df.shape[0], desc="Processing rows"):
#     # Combine text from all available columns to check for keywords.
#     text_to_check = " ".join(
#         str(row[col]) for col in available_cols if pd.notnull(row[col])
#     ).lower()
    
#     k_flag = 0
#     e_flag = 0
#     found_keyword = ""
#     found_exception = ""
    
#     # Loop through each (keyword, exceptions) tuple.
#     for keyword, exceptions in keyword_exceptions:
#         if keyword.lower() in text_to_check:
#             k_flag = 1
#             found_keyword = keyword
#             # Check if any of the exception words are present.
#             for ex_word in exceptions:
#                 if ex_word.lower() in text_to_check:
#                     e_flag = 1
#                     found_exception = ex_word
#                     break  # Exception word found; break inner loop.
#             break  # Keyword matched; break the keyword loop.
    
#     ctd_df.at[i, 'keyword_flag'] = k_flag
#     ctd_df.at[i, 'exception_flag'] = e_flag
#     ctd_df.at[i, 'found_keywords'] = found_keyword
#     ctd_df.at[i, 'found_exceptions'] = found_exception

# # Create a new DataFrame with only the required columns: nct_id and the flags.
# cols_to_save = ['nct_id', 'keyword_flag', 'exception_flag', 'found_keywords', 'found_exceptions']
# result_df = ctd_df[cols_to_save]

# # Save the resulting DataFrame to a CSV file.
# result_df.to_csv('../Data/Flagged_CTD_6_De_T_XTN.csv', index=False)


"""
Created on Sun April  3 13:17:21 2025

@author: Akhyar.Ahmed
"""

import os
import re
import pandas as pd
from tqdm import tqdm
from dht_flag import FlagDHT

# Read the Excel file containing keywords and exception lists,
# using the second row as the header.
exceptions_df = pd.read_excel('../Data/tcp-32-41-s003.xls', engine='xlrd', header=1)
print(exceptions_df.head())

# Read the CTD CSV file.
ctd_df = pd.read_csv('../Data/CTD_6.csv')

# Narrow down the columns for keyword matching to only these two.
search_cols = ['q_10_desc']
available_cols = [col for col in search_cols if col in ctd_df.columns]

# Create new columns for the flags and found keywords/exceptions.
ctd_df['keyword_flag'] = 0
ctd_df['exception_flag'] = 0
ctd_df['found_keywords'] = ""
ctd_df['found_exceptions'] = ""

# Build a list of tuples from exceptions_df: (keyword, [exception words])
keyword_exceptions = []
for i, row in exceptions_df.iterrows():
    keyword = str(row['Keywords']).strip()
    exception_list_raw = str(row['Exception list'])
    exception_words = [ex.strip() for ex in exception_list_raw.split(',')]
    keyword_exceptions.append((keyword, exception_words))

# Iterate over each row in ctd_df with a tqdm progress bar.
for i, row in tqdm(ctd_df.iterrows(), total=ctd_df.shape[0], desc="Processing rows"):
    # Combine text from only the two specified columns for keyword matching.
    text_to_check = " ".join(
        str(row[col]) for col in available_cols if pd.notnull(row[col])
    ).lower()
    
    k_flag = 0
    e_flag = 0
    found_keyword = ""
    found_exception = ""
    
    # Loop through each (keyword, exceptions) tuple.
    for keyword, exceptions in keyword_exceptions:
        if keyword.lower() in text_to_check:
            k_flag = 1
            found_keyword = keyword
            # Check if any of the exception words are present.
            for ex_word in exceptions:
                if ex_word.lower() in text_to_check:
                    e_flag = 1
                    found_exception = ex_word
                    break  # Exception word found; break inner loop.
            break  # Keyword matched; break out of the keyword loop.
    
    ctd_df.at[i, 'keyword_flag'] = k_flag
    ctd_df.at[i, 'exception_flag'] = e_flag
    ctd_df.at[i, 'found_keywords'] = found_keyword
    ctd_df.at[i, 'found_exceptions'] = found_exception

# Create a new DataFrame with only the required columns: nct_id and the flags.
cols_to_save = ['nct_id', 'keyword_flag', 'exception_flag', 'found_keywords', 'found_exceptions']
result_df = ctd_df[cols_to_save]

# Save the resulting DataFrame to a CSV file.
result_df.to_csv('../Data/Flagged_CTD_6_De_T_XTN_q10.csv', index=False)
