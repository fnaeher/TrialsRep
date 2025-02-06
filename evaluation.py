"""
Created on Mon Feb 06 09:51:00 2025

Authors: Akhyar.Ahmed
"""

import time
import json
import pandas as pd
import openai
from openai import ChatCompletion 
from huggingface_hub import InferenceClient
from tqdm import tqdm  # For progress bars

from config.settings import cat_dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

PROMPT_TEMPLATE = '''
You are provided with a labeling task for clinical trial data.
- The phrase is "{exp}".
- The category is "{title}".
- The only valid labels for this category are: {cat}.

IMPORTANT:
- For the "age" category, if the phrase implies conflicting age ranges (e.g. "40-75" which conflicts with both "between 18 and 65 years" and ">65 years"), output "unknown_a".
- For the "gender" category, if the phrase is ambiguous or does not clearly match "male", "female", or "other", output "unknown_g".
- For the "race" category, if the phrase contains conflicting or unclear racial information, output "unknown_r".
Output strictly valid JSON with no additional commentary, in the format:

{{
  "cat_title": "{title}",
  "cat_exp": "{exp}",
  "piv_cat": "<one label from {cat}>"
}}

Where:
- "cat_title" = {title},
- "cat_exp" = {exp},
- "piv_cat" is exactly one from the list {cat}.

Return only valid JSON as a string. No disclaimers, no extra text.
'''


def extract_piv_cat(raw_text):
    if not raw_text:
        return "missing"
    try:
        parsed = json.loads(raw_text)
        if "piv_cat" in parsed:
            value = parsed["piv_cat"]
            if not isinstance(value, str):
                value = str(value)
            return value.strip('"').strip()
        else:
            return "missing"
    except Exception:
        key = '"piv_cat":'
        idx = raw_text.find(key)
        if idx == -1:
            return "missing"
        substring = raw_text[idx + len(key):]
        end_idx = substring.find("}")
        if end_idx == -1:
            end_idx = substring.find(",")
        if end_idx != -1:
            substring = substring[:end_idx]
        return substring.strip(' ,"')


def clean_annotation_df(annotation_df):
    annotation_df['cat_label'].mask(
        annotation_df['cat_label'].str.lower() == 'american indian or alaska native',
        'American Indian or Alaska Native',
        inplace=True
    )
    annotation_df['cat_label'].mask(
        annotation_df['cat_label'].str.lower() == 'asian',
        'Asian',
        inplace=True
    )
    annotation_df['cat_label'].mask(
        annotation_df['cat_label'].str.lower() == 'black or african american',
        'Black or African American',
        inplace=True
    )
    annotation_df['cat_label'].mask(
        annotation_df['cat_label'].str.lower() == 'hispanic or latino',
        'Hispanic or Latino',
        inplace=True
    )
    annotation_df['cat_label'].mask(
        annotation_df['cat_label'].str.lower() == 'native hawaian or other pacific islander',
        'Native Hawaian or Other Pacific Islander',
        inplace=True
    )
    annotation_df['cat_label'].mask(
        annotation_df['cat_label'].str.lower() == 'white',
        'White',
        inplace=True
    )
    annotation_df['cat_label'].mask(
        annotation_df['cat_label'].str.lower() == 'unknown_r',
        'unknown_r',
        inplace=True
    )
    return annotation_df


def predict_with_gpt(annotation_df):
    openai.api_key = "sk-proj-XXXX"
    
    piv_label_dict = {}
    models = ['chatgpt-4o-latest', 'gpt-4o-mini']
    for model in models:
        piv_label_ls = []
        for idx, row in tqdm(annotation_df.iterrows(), total=len(annotation_df), desc=f"GPT ({model}) predictions"):
            cat_title = row['cat_title']
            cat_exp = row['cat_exp']
            allowed_labels = cat_dict.get(cat_title, [])
            prompt = PROMPT_TEMPLATE.format(title=cat_title, exp=cat_exp, cat=allowed_labels)
            
            try:
                response = ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200
                )
                raw_text = response.choices[0].message.get('content', '').strip()
                if '```json' in raw_text:
                    raw_text = raw_text.replace('```json', '')
                if '```' in raw_text:
                    raw_text = raw_text.replace('```', '')

                if not raw_text:
                    raise ValueError("Empty response content")
                parsed = json.loads(raw_text)
                piv_cat = parsed.get('piv_cat')
            except Exception as e:
                print(f"Error processing GPT prediction for row {idx} with model {model}: {e}")
                piv_cat = "missing"
            
            piv_label_ls.append(piv_cat)
            time.sleep(0.5)

        if 'mini' in model:
            key = 'gpt_4o_mini_piv_cat'
        else:
            key = 'gpt_4o_piv_cat'
        
        piv_label_dict[key] = piv_label_ls
    return piv_label_dict


def predict_with_llama(annotation_df):
    hf_token = "hf_XXXX"
    hf_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    client = InferenceClient(api_key=hf_token)
    piv_label_dict = {}
    for model in ['meta-llama/Llama-3.3-70B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct']:
        piv_label_ls = []
        for idx, row in tqdm(annotation_df.iterrows(), total=len(annotation_df), desc=f"Llama ({model}) predictions"):
            cat_title = row['cat_title']
            cat_exp = row['cat_exp']
            allowed_labels = cat_dict.get(cat_title, [])
            prompt = PROMPT_TEMPLATE.format(title=cat_title, exp=cat_exp, cat=allowed_labels)
            
            try:
                if model == 'meta-llama/Llama-3.3-70B-Instruct':
                    pro_token = 'hf_XXXX'
                    client_pro = InferenceClient(api_key=pro_token)
                    response = client_pro.chat.completions.create(
                        model = model,
                        messages=[
                            {"role": "user", "content": prompt}],
                        max_tokens = 256
                    )
                else:
                    response = client.chat.completions.create(
                        model = model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens = 500
                    )
                raw_text = response.choices[0].message.get('content', '').strip()
                if not raw_text or not raw_text.startswith('{'):
                    print(f"Warning: Empty or invalid response for row {idx} with Llama.")
                    piv_cat = "missing"
                else:
                    piv_cat = extract_piv_cat(raw_text)
            except Exception as e:
                print(f"Error processing Llama prediction for row {idx}: {e}")
                piv_cat = "missing"
            piv_label_ls.append(piv_cat)
            time.sleep(0.5)
        if '3-8B' in model:
            model_name = 'llama_3_8b_piv_cat'
        else:
            model_name = 'llama_3_3_70B_piv_cat'
        piv_label_dict[model_name] = piv_label_ls
    return piv_label_dict


def compute_metrics(df):
    true_labels = df['cat_label']
    
    metrics_list = []
    for col_names in list(df.columns.values):
        if col_names in ['gpt_4o_mini_piv_cat', 'gpt_4o_piv_cat', 'llama_3_8b_piv_cat', 'llama_3_3_70B_piv_cat']:
            df[col_names] = df[col_names].fillna("missing")
    
    for model in ['gpt_4o_mini_piv_cat', 'gpt_4o_piv_cat', 'llama_3_8b_piv_cat', 'llama_3_3_70B_piv_cat']:
        if model in list(df.columns.values):
            preds = df[model]
            accuracy = accuracy_score(true_labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted', zero_division=0)
            
            print(f"Classification Report for {model}:")
            print(classification_report(true_labels, preds, zero_division=0))
            print("\n----------------------------------------\n")
            
            if 'mini' in model:
                model_name = 'gpt-4o-mini'
            elif 'gpt_4o' in model:
                model_name = 'gpt-4o'
            elif 'llama_3_8b' in model:
                model_name = 'llama-3-8B-Instruct'
            else:
                model_name = 'llama-3.3-70B-Instruct'
            metrics_list.append({
                "Model": model_name,
                "Accuracy": round(accuracy, 3),
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1 Score": round(f1, 3)
            })
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('result_metrics.csv', index=False)
    return metrics_df


def evaluate(label_path):
    cols = ['cat_exp', 'cat_title', 'cat_label']
    annotation_df = pd.read_csv(label_path, sep=';', encoding='utf-8', usecols=cols)
    annotation_df = clean_annotation_df(annotation_df)
    
    print("Obtaining predictions using GPT models...")
    gpt_predictions_dict = predict_with_gpt(annotation_df)

    annotation_df['gpt_4o_mini_piv_cat'] = gpt_predictions_dict.get('gpt_4o_mini_piv_cat', [])
    annotation_df['gpt_4o_piv_cat'] = gpt_predictions_dict.get('gpt_4o_piv_cat', [])
    
    print("Obtaining predictions using Llama model...")
    llama_predictions_dict = predict_with_llama(annotation_df)
    annotation_df['llama_3_8b_piv_cat'] = llama_predictions_dict['llama_3_8b_piv_cat']
    annotation_df['llama_3_3_70B_piv_cat'] = llama_predictions_dict['llama_3_3_70B_piv_cat']

    metrics_df = compute_metrics(annotation_df)
    print("Metrics computed:")
    print(metrics_df)
    
    return annotation_df


if __name__ == "__main__":
    label_path = './Data/Annotation_List_TrailRepGov.csv'
    results_df = evaluate(label_path)
    results_df.to_csv('llama_gpt_predictions.csv', encoding='utf-8', index=False)
