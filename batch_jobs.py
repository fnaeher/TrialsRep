# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:09:14 2024

@author: AnatolFiete.Naeher
"""

path = ("path\\")


import pandas as pd
import json
from openai import OpenAI
import time
import os
import sys

sys.path.append(path.replace('Data', 'Code'))

from more_itertools import chunked   
from data_prep import data_prep2
from lists import d_CTD, age, gender, race, enrollment


def countdown(minutes):
    for remaining in range(minutes * 60, 0, -1):
        mins, secs = divmod(remaining, 60)
        timeformat = f"{mins} min and {secs} s"
        print(f"Checking again in {timeformat}."
              " ", end = '\r')
        time.sleep(1)


def extract_json(json_df):
    results = []
    try:
        for i in range(len(json_df)):
            cont = (json_df[i]['response']['body']['choices'][0]['message']  
                ['content'])
            p_cont = json.loads(cont)
            results.append((p_cont.get('cat_title'), p_cont.get('cat_exp'), 
                            p_cont.get('piv_cat')))
        
    except KeyError as e:
        print(f"KeyError: {e} at index {i}")
        results.append([None, None])
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e} at index {i}")
        results.append([None, None])
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        results.append([None, None])
       
    return results 


client = OpenAI()

chunks = 5

ctdwn = 3

b_job_dict = {}

b_output = []

cat_dict = {
    'gender': ['male', 'female', 'other', 'unknown_g'],
    'age' : ['<18 years', 'between 18 and 65 years', '>65 years', 'unknown_a'],
    'race' : ['American Indian or Alaska Native', 'Asian',
              'Black or African American', 'Hispanic or Latino',
              'Native Hawaian or Other Pacific Islander', 'White', 'unknown_r']
}


prompt_template = '''
You are provided with a labeling task: label the phrase {exp} with one and only 
one of the labels in the list {cat}. Output the result as a list:

['cat_title', 'cat_exp', 'piv_cat'],

where

'cat_title' = {title},
'cat_exp' = {exp}
and 'piv_cat' designates the assigned label from list {cat}.

You must use labels in {cat}, only. You must not create any other 
labels. You must use expressions {title} and {exp}, only. You must not
create any other expressions.

Output without any additional formatting, code blocks, or extra text, 
ensuring proper 'cat_title', 'cat_exp', and 'piv_cat' keys. 

If {exp} contains an ambiguous age range that cannot be labelled unambigously
based on labels in list{cat}, classify it as 'unknown_a'.
'''

d_CTD['CTD_2'] = data_prep2(d_CTD, age, gender, race, enrollment)

if os.path.join(path, 'b_output.csv'):
    b_output = pd.read_csv(os.path.join(path, 'b_output.csv'))

else: 
    for cat, scat in cat_dict.items():
       
        exp = list(set(map(str, d_CTD['CTD_2'][(d_CTD['CTD_2']
            ['cat_title'] == cat)]['cat_exp'])))
    
        exp = list(chunked(exp, (len(exp) // chunks)))
    
        tasks = []
        
        for chunk_i, exp_i in enumerate(exp, start = 1):
        
            
            for i in exp_i:
                prompt = prompt_template.format(title = cat, exp = i, cat=scat)
    
                task = {
                    "custom_id": f"task-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "temperature": 0,
                        "max_tokens": 200,
                        "top_p": 1,
                        "presence_penalty": -0.5,
                        "frequency_penalty": -0.5,
                        "response_format": {"type": "json_object"},    
                        "messages": [{"role": "user", "content": 
                                   f"{prompt} Please respond in JSON format."}]
                            }
                }
                    
                tasks.append(task)
    
            file_name = f"{cat}_chunk_{chunk_i}.jsonl"
            
            with open(file_name, 'w') as file:
                for obj in tasks:
                    file.write(json.dumps(obj) + '\n')
    
            b_file = client.files.create(
                file = open(file_name, "rb"),
                purpose="batch"
            )
    
            b_job = client.batches.create(
                input_file_id=b_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
    
            b_job_dict[f"{cat}_chunk_{chunk_i}"] = b_job.id
    
            print(f"\nWaiting for batch job for chunk {chunk_i} out of"
                  F" {chunks} of category {cat} to complete.")
            
            completed = False
            while not completed:
                completed = True
                
                for b_id in [b_job_dict[f"{cat}_chunk_{chunk_i}"]]:
                    if client.batches.retrieve(b_id).status != 'completed':
                        completed = False
                        countdown(ctdwn)
                        break
    
            print(f"\nBatch job for chunk {chunk_i} out of {chunks} of"
                  f" category {cat} completed. Proceeding with next chunk...")
    
    print("\nAll batch jobs completed.")
    
    
    for cat, b_id in b_job_dict.items():
       output_file_id = client.batches.retrieve(b_id).output_file_id
       json_df = client.files.content(output_file_id).content
       
       extracted_df_name = f"{cat}.jsonl"
       with open(extracted_df_name, 'wb') as file:
           file.write(json_df)
    
       json_df = []
       with open(extracted_df_name, 'r') as file:
           for line in file:
               json_object = json.loads(line.strip())
               json_df.append(json_object)
       
       b_output.append(extract_json(json_df))
    
    
    b_output = pd.DataFrame([item for sublist in b_output for item in sublist], 
                      columns = ['cat_title','cat_exp', 'piv_cat'])
