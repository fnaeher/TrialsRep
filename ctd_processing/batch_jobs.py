# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:01:33 2024

@author: AnatolFiete.Naeher
"""

import os
import time
import json
import logging
import pandas as pd
from more_itertools import chunked
from config.settings import chunks, ctdwn, openAI_Model


class DefineTask:
    def __init__(self, openAI_model, prompt_template):
        logging.basicConfig(level = logging.INFO, 
            format = "%(asctime)s - %(levelname)s - %(message)s")
        
        self.model = openAI_model
        self.prompt_template = prompt_template

    def b_create(self, exp_i, cat, scat):
        logging.info("Creating batch job...")
        
        return [
            {
                "custom_id": f"task-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": 200,
                    "top_p": 1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": 
                        self.prompt_template.format(title = cat, exp = i, 
                        cat = scat) + 
                        " Please respond in JSON format."}]    
                }
            }
            for i in exp_i
        ]


class RetryClient:
   def __init__(self, client, retries = 3, delay = 2):
       logging.basicConfig(level = logging.INFO, 
           format = "%(asctime)s - %(levelname)s - %(message)s")
       
       self.client = client
       self.retries = retries
       self.delay = delay

   def retry(self, func, *args, **kwargs):
        for attempt in range(self.retries):
           try:
               return func(*args, **kwargs)
           
           except Exception as e:
                logging.error(f"Error: {e}. Retrying {attempt + 1}/" 
                      f"{self.retries} after {self.delay} seconds.")
                
                time.sleep(self.delay)
                
        raise Exception("Maximum retry attempts reached.")
        
   def __getattr__(self, attr):
        return getattr(self.client, attr)


class ProcessBatch:
    def __init__(self, client, task_creator):
        logging.basicConfig(level = logging.INFO, 
            format = "%(asctime)s - %(levelname)s - %(message)s")
        
        self.client = client
        self.task_creator = task_creator
        self.b_completion_count = 0

    def b_submit(self, file_name, tasks):
        with open(file_name, 'w') as file:
            for task in tasks:
               file.write(json.dumps(task) + '\n')
                
        with open(file_name, "rb") as file:
            return self.client.files.create(file = file, purpose = "batch")

    def b_completion(self, b_id):
        self.b_completion_count += 1
        
        logging.info(f"\nWaiting for batch {self.b_completion_count} "
                     f" to complete...")
       
        while True:
            batch_status = self.client.batches.retrieve(b_id).status
        
            if batch_status == 'completed':
                logging.info("Batch completed. Proceeding...")
               
                break
        
            else:
                self._countdown(ctdwn)

    def _countdown(self, minutes):
        for remaining in range(minutes * 60, 0, -1):
            mins, secs = divmod(remaining, 60)
            
            print(f"Checking again in {mins} min and {secs} s.", end='\r')
            
            time.sleep(1)


class HandleResult:
    def __init__(self, client):
        logging.basicConfig(level = logging.INFO, 
            format = "%(asctime)s - %(levelname)s - %(message)s")
        
        self.client = client

    def process_results(self, batch_id):
        logging.info("Retrieving results...")
        
        output_file_id = self.client.batches.retrieve(batch_id).output_file_id
        json_dta = self.client.files.content(output_file_id).content
        json_df = []
        
        with open("output.jsonl", 'wb') as file:
            file.write(json_dta)
            
        with open("output.jsonl", 'r') as file:
            for line in file:
                json_obj = json.loads(line.strip())
                json_df.append(json_obj)
                
        return self._extract_json(json_df)

    def _extract_json(self, json_df):
        results = []
        
        for i, row in enumerate(json_df):
            try:
                content = row['response']['body']['choices'][0]['message']['content']
                parsed_content = json.loads(content)
                results.append((parsed_content.get('cat_title'), 
                                parsed_content.get('cat_exp'), 
                                parsed_content.get('piv_cat')))
                
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error at index {i}: {e}")
                
                results.append([None, None, None])
                
        return results


def b_jobs(path, batch_filename, prompt_template, cat_dict, data, client):
    
    os.chdir(path)
    
    b_job_dict = {}

    b_output = pd.DataFrame()
    
    if os.path.isfile(os.path.join(path, f"{batch_filename}.csv")):
        
        b_output = pd.read_csv(os.path.join(path, f"{batch_filename}.csv"))
    
    else:
        
        cat_exp = []
    
        task_creator = DefineTask(openAI_Model, prompt_template)
        retry_client = RetryClient(client)
        batch_processor = ProcessBatch(retry_client, task_creator)
        result_handler = HandleResult(retry_client)
    
        for cat, scat in cat_dict.items():
            exp = list(set(map(str, data[(data['cat_title'] == cat)]
                ['cat_exp'])))
            
            cat_exp.extend(list(set(exp)))
            
            exp = list(chunked(exp, (len(exp) // chunks)))
             
            for chunk_i, exp_i in enumerate(exp, start = 1):
                tasks = task_creator.b_create(exp_i, cat, scat)

                file_name = f"{cat}_chunk_{chunk_i}.jsonl"
                
                b_file = batch_processor.b_submit(file_name, tasks)
                
                b_job = client.batches.create(
                    input_file_id = b_file.id,
                    endpoint = "/v1/chat/completions",
                    completion_window = "24h"
                )
                
                b_job_dict[f"{cat}_chunk_{chunk_i}"] = b_job.id
                
                batch_processor.b_completion(b_job.id)
                
        for cat, b_id in b_job_dict.items():
                result = result_handler.process_results(b_id)

                if isinstance(result, list):
                    result = pd.DataFrame(result, 
                        columns = ['cat_title', 'cat_exp', 'piv_cat'])

                b_output = pd.concat([b_output, result], ignore_index = True)

                b_output['cat_exp'] = [
                    tuple(item) if isinstance(item, list) else item for item 
                        in b_output['cat_exp']
                    ]
                
                b_output = b_output.drop_duplicates()

        b_output['cat_exp'] = [
            tuple(item) if isinstance(item, list) else item for item 
                in b_output['cat_exp']
            ]
        
        b_output.to_csv(os.path.join(path, f"{batch_filename}.csv"), 
                index = False)
        
    print(f"Batch job results saved as {batch_filename}.csv in {path}")
    
    return b_output