"""
Created on Mon Jan 27 10:58:00 2025

@author: Akhyar.Ahmed
"""
import pandas as pd
import numpy as np
from ctd_processing.lists import d_CTD, age, gender, race, enrollment
from ctd_processing.data_prep import data_prep2


def sample_three_cat_titles(d_CTD, cat_titles=None, total_samples=200, seed=42):
    if cat_titles is None:
        cat_titles = ["age", "gender", "race"]  # default

    df = d_CTD["CTD_2"].copy()
    df = df[df["cat_title"].isin(cat_titles)]
    
    if df.empty:
        print("No rows found for the given cat_titles.")
        return df
    
    np.random.seed(seed)
    L = len(cat_titles)
    per_label = total_samples // L  
    remainder = total_samples % L  

    subsets = []

    for i, ct in enumerate(cat_titles):
        group_df = df[df["cat_title"] == ct]
        
        if group_df.empty:
            print(f"No rows found for cat_title={ct}, skipping.")
            continue
        
        n_sample = per_label + (1 if i < remainder else 0)
        
        if len(group_df) <= n_sample:
            chosen = group_df
        else:
            chosen = group_df.sample(n=n_sample, random_state=seed)
        
        subsets.append(chosen)

    sampled_df = pd.concat(subsets, ignore_index=True)
    sampled_df.drop_duplicates(inplace=True) 

    if len(sampled_df) > total_samples:
        sampled_df = sampled_df.sample(n=total_samples, random_state=seed)
    elif len(sampled_df) < total_samples:
        print(f"WARNING: only {len(sampled_df)} rows in final sample (some categories too small).")

    return sampled_df


if __name__ == "__main__":
    d_CTD['CTD_2'] = data_prep2(d_CTD, age, gender, race, enrollment)

    sampled_rows = sample_three_cat_titles(
        d_CTD, 
        cat_titles=["age","gender","race"], 
        total_samples=200, 
        seed=42
    )
    print(f"Sampled {len(sampled_rows)} rows for 'age','gender','race'.")

    sampled_rows["annotator_1"] = np.nan
    sampled_rows["annotator_2"] = np.nan

    sampled_rows.to_csv("./Data/CTD_annotation_file_200_samples.csv", index=False)
    print("Saved 'CTD_annotation_file_200_samples.csv'.")