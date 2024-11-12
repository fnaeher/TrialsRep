# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:09:14 2024

@author: AnatolFiete.Naeher
"""

import pandas as pd
import json
from openai import OpenAI
import time
import os
from more_itertools import chunked 
from config.settings import path, chunks, ctdwn, openAI_Model, batch_filename
from ctd_processing.lists import d_CTD, age, gender, race, enrollment
from ctd_processing.data_prep import data_prep2


client = OpenAI()

data = d_CTD['CTD_2'] = data_prep2(d_CTD, age, gender, race, enrollment)

columns = ['cat_title','cat_exp', 'piv_cat']


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
'''


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


def process_batches(path, batch_filename, cat_dict, exp, openAi_Model,
    prompt_template, client, chunks, ctdwn, columns):
    
    os.chdir(path)
    
    b_job_dict = {}

    b_output = []
    
    if os.path.isfile(os.path.join(path, f"{batch_filename}.csv")):
        b_output = pd.read_csv(os.path.join(path, f"{batch_filename}.csv"))
    
    else:
        
        cat_exp = []
        
        for cat, scat in cat_dict.items():
           
            exp = list(set(map(str, data[(data['cat_title'] == cat)]
                ['cat_exp'])))
            
            cat_exp.extend(list(set(exp)))
            
            exp = list(chunked(exp, (len(exp) // chunks)))
             
            for chunk_i, exp_i in enumerate(exp, start = 1):
            
                tasks = [
                    {
                        "custom_id": f"task-{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": openAI_Model,
                            "temperature": 0,
                            "max_tokens": 200,
                            "top_p": 1,
                            "presence_penalty": 0,
                            "frequency_penalty": 0,
                            "response_format": {"type": "json_object"},
                            "messages": [{"role": "user", "content": 
                                prompt_template
                                .format(title = cat, exp = i, cat = scat) + 
                                " Please respond in JSON format."}]
                        }
                    }
                    for i in exp_i
                ]
        
                file_name = f"{cat}_chunk_{chunk_i}.jsonl"
                
                with open(file_name, 'w') as file:
                    for obj in tasks:
                        file.write(json.dumps(obj) + '\n')
        
                b_file = client.files.create(
                    file = open(file_name, "rb"),
                    purpose = "batch"
                )
        
                b_job = client.batches.create(
                    input_file_id = b_file.id,
                    endpoint = "/v1/chat/completions",
                    completion_window = "24h"
                )
        
                b_job_dict[f"{cat}_chunk_{chunk_i}"] = b_job.id
        
                print(f"\nWaiting for batch job for chunk {chunk_i} out of"
                      F" {len(exp)} of category {cat} to complete.")
                
                completed = False
                while not completed:
                    completed = True
                    
                    for b_id in [b_job_dict[f"{cat}_chunk_{chunk_i}"]]:
                        if client.batches.retrieve(b_id).status != 'completed':
                            completed = False
                            countdown(ctdwn)
                            break
        
                print(f"\nBatch job for chunk {chunk_i} out of {len(exp)} of"
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
        
        b_output = pd.DataFrame([item for sublist in b_output for item 
           in sublist], columns = columns)
        
        b_output['cat_exp'] = [
            tuple(item) if isinstance(item, list) else item for item 
                in b_output['cat_exp']
            ]
        
        if set(cat_exp) - set(b_output['cat_exp']):
            tasks = []
            
            f_data = data[data['cat_exp'].isin(set(cat_exp) - 
                set(b_output['cat_exp']))]
            
            for cat, scat in cat_dict.items():
                exp_list = list(set(f_data[f_data['cat_title'] == 
                    cat]['cat_exp']))
                
                for i in exp_list:
                    task = {
                        "custom_id": f"task-{cat}{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": openAI_Model,
                            "temperature": 0,
                            "max_tokens": 200,
                            "top_p": 1,
                            "presence_penalty": 0,
                            "frequency_penalty": 0,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {
                                    "role": "user",
                                    "content": (
                                        prompt_template.format(title=cat, 
                                            exp=i, cat=scat) +
                                        " Please respond in JSON format."
                                    )
                                }
                            ]
                        }
                    }
                    
                    tasks.append(task)
            
            file_name = "correct.jsonl"
            with open(file_name, 'w') as file:
                for obj in tasks:
                    file.write(json.dumps(obj) + '\n')
            
            with open(file_name, "rb") as file:
                b_file = client.files.create(file=file, purpose="batch")
            
            b_job = client.batches.create(
                input_file_id=b_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            b_job_dict["correct"] = b_job.id
            
            print("\nWaiting for batch job corrections to complete.")
            
            completed = False
            while not completed:
                completed = True
                for b_id in [b_job_dict["correct"]]:
                    if client.batches.retrieve(b_id).status != 'completed':
                        completed = False
                        countdown(ctdwn)
                        break
            
            print("Batch job corrections completed.")
          
            output_file_id = client.batches.retrieve(b_job_dict['correct']) \
                .output_file_id
            json_df = client.files.content(output_file_id).content
            
            extracted_df_name = "correct.jsonl"
            with open(extracted_df_name, 'wb') as file:
                file.write(json_df)
         
            json_df = []
            with open(extracted_df_name, 'r') as file:
                for line in file:
                    json_object = json.loads(line.strip())
                    json_df.append(json_object)
                    
            cb_output = pd.DataFrame(extract_json(json_df))
            cb_output.columns = b_output.columns
            b_output = pd.concat([b_output, cb_output], ignore_index=True)
            
            b_output['cat_exp'] = [
                tuple(item) if isinstance(item, list) else item for item 
                    in b_output['cat_exp']
                ]
            
            b_output = b_output.drop_duplicates()
        
        b_output.to_csv(os.path.join(path, f"{batch_filename}.csv"), 
            index = False)
        print(f"Batch jobs saved as {batch_filename}.csv"
              f" saved in {path}")
        
    d_CTD['b_output'] = b_output
    
    return(d_CTD)


b_output = process_batches(path, batch_filename, cat_dict, data, openAI_Model,
    prompt_template, client, chunks, ctdwn, columns)
