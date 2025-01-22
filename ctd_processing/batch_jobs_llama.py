"""
Created on Fri Dec 20 13:17:00 2024
@author: Akhyar.Ahmed

A job script for an open-source LLama model.

Usage:
  - This version uses HF's InferenceClient for a remote Llama model.
  - The function `llama_jobs(...)` chunks the data, calls HF in a loop, 
    and returns a DataFrame with [cat_title, cat_exp, piv_cat].
"""

import os
import time
import json
import logging
import pandas as pd
from more_itertools import chunked
from config.settings import chunks, ctdwn

from huggingface_hub import InferenceClient


class DefineTaskLlama:
    """
    Creates 'tasks' for each expression. We just store
    the final prompt in memory. 
    """
    def __init__(self, prompt_template):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.prompt_template = prompt_template

    def b_create(self, exp_i, cat, scat):
        """
        Returns a list of tasks (one per expression). Each item has a
        'prompt' field we will later send to HF InferenceClient.
        """
        logging.info("Creating batch job...")

        tasks = []
        for i in exp_i:
            final_prompt = self.prompt_template.format(title=cat, exp=i, cat=scat)

            tasks.append({
                "expression": i,
                "category": cat,
                "subcat": scat,
                "prompt": final_prompt
            })
        return tasks


def llama_jobs(
    path,
    batch_filename,
    prompt_template,
    cat_dict,
    data,
    hf_model_name="meta-llama/Llama-2-7b-chat-hf",
    hf_token=None
):
    """
    Main function for chunking data and calling a remote HF Llama model.
    We produce a final CSV named {batch_filename}.csv with columns
    [cat_title, cat_exp, piv_cat].
    """
    # If the CSV already exists, skip
    os.chdir(path)
    csv_path = os.path.join(path, f"{batch_filename}.csv")
    if os.path.isfile(csv_path):
        logging.info(f"{batch_filename}.csv already exists. Loading it.")
        return pd.read_csv(csv_path)

    b_output = pd.DataFrame()

    logging.info(f"Creating HF InferenceClient with model={hf_model_name}")
    client = InferenceClient(api_key=hf_token)

    task_creator = DefineTaskLlama(prompt_template)

    for cat, scat in cat_dict.items():
        exp_list = list(set(map(str, data[data['cat_title'] == cat]['cat_exp'])))
        if not exp_list:
            continue

        chunk_size = max(1, len(exp_list)//chunks)
        exp_chunks = list(chunked(exp_list, chunk_size))

        for chunk_i, exp_i in enumerate(exp_chunks, start=1):
            tasks = task_creator.b_create(exp_i, cat, scat)

            chunk_output = []
            for t in tasks:
                prompt_text = t['prompt']
                messages = [
                    {"role": "user", "content": prompt_text}
                ]
                try:
                    completion = client.chat.completions.create(
                        model=hf_model_name,
                        messages=messages,
                        max_tokens=500
                    )
                    # Parse the result from HF
                    # e.g. completion.choices[0].message['content']
                    raw_text = completion.choices[0].message['content']

                    # Attempt JSON parse
                    try:
                        parsed = json.loads(raw_text)
                        cat_title = parsed.get('cat_title')
                        cat_exp   = parsed.get('cat_exp')
                        piv_cat   = parsed.get('piv_cat')
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON parse error for expression {t['expression']}: {e}")
                        logging.error(f"{raw_text}")
                        logging.error(f"\n")
                        cat_title = None
                        cat_exp   = None
                        piv_cat   = None

                except Exception as e:
                    logging.error(f"HF Inference API error: {e}")
                    cat_title = None
                    cat_exp   = None
                    piv_cat   = None

                chunk_output.append((cat_title, cat_exp, piv_cat))

            df_chunk = pd.DataFrame(chunk_output, columns=['cat_title','cat_exp','piv_cat'])
            chunk_csv = f"hf_llama_{cat}_chunk_{chunk_i}.csv"
            df_chunk.to_csv(os.path.join(path, chunk_csv), index=False)

            b_output = pd.concat([b_output, df_chunk], ignore_index=True).drop_duplicates()

    b_output.to_csv(csv_path, index=False)
    logging.info(f"LLama batch job results saved as {batch_filename}.csv in {path}")
    return b_output
