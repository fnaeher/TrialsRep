"""
Created on Fri April 04 16:05:08 2025

@author: Akhyar.Ahmed
"""

if __name__ == "__main__":
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from config.settings import (
        path, batch_filename, prompt_template, cat_dict, filename
    )
    from ctd_processing.lists import d_CTD, age, gender, race, enrollment
    from ctd_processing.data_prep import data_prep2, merge
    
    # Ensure all 'NCT_id' columns in d_CTD are of type string
    for key in d_CTD.keys():
        if isinstance(d_CTD[key], pd.DataFrame) and 'NCT_id' in d_CTD[key].columns:
            d_CTD[key]['NCT_id'] = d_CTD[key]['NCT_id'].astype(str)
    # Create a progress bar with a total number of steps.
    pbar = tqdm(total=6, desc="Processing full pipeline")

    # Step 1: Data preparation - compute CTD_2.
    d_CTD['CTD_2'] = data_prep2(d_CTD, age, gender, race, enrollment)
    pbar.update(1)

    # Step 2: Merge CTD data and adjust column names.
    df_ctd = merge(d_CTD)
    df_ctd.rename(columns={'NCT_id': 'nct_id'}, inplace=True)
    pbar.update(1)

    # Step 3: Process Flagged_DHT.csv.
    df_dht = pd.read_csv('Data/Flagged_DHT.csv')
    df_dht = df_dht.drop_duplicates(subset=['nct_id'])
    df_dht = df_dht[['nct_id', 'DHT']]
    pbar.update(1)

    # Step 4: Process Flagged_CTD_6_De_T_XTN_q10.csv.
    df_flags = pd.read_csv('Data/Flagged_CTD_6_De_T_XTN_q10.csv')
    df_flags = df_flags.drop_duplicates(subset=['nct_id'])
    df_flags = df_flags[['nct_id', 'keyword_flag', 'exception_flag']]
    df_flags['De_T'] = np.where(
        (df_flags['keyword_flag'] == 1) & (df_flags['exception_flag'] == 0),
        1,
        0
    )
    df_de_t = df_flags[['nct_id', 'De_T']]
    pbar.update(1)

    # Step 5: Merge all three dataframes on nct_id (inner join).
    merged_df = df_ctd.merge(df_dht, on='nct_id', how='inner') \
                      .merge(df_de_t, on='nct_id', how='inner')
    pbar.update(1)

    print(merged_df[merged_df.De_T==1].shape)
    # Step 6: Save the final merged dataframe to CSV.
    merged_df.to_csv('Data/merged_CTD_DHT_DeT.csv', index=False)
    pbar.update(1)

    pbar.close()
