"""
Created on Wed Oct  9 18:11:47 2024

@author: AnatolFiete.Naeher and Akhyar.Ahmed

A unified approach that:
- merges GPT or Llama outputs
- pivot the data
- flattens single-element lists
- unify_columns() to fill known final columns (like Male, White, etc.)
- keeps any extra pivot columns, no forced rename
- pivots dynamically and flattens single-element lists,
- always includes certain "core" columns (e.g. 'other') via unify_columns(),
- computes percentages without KeyErrors if 'other' doesn't exist in the model output.
"""

import pandas as pd
import numpy as np
import re

def c_title(title, age, gender, race, enrollment):
    """
    Classify 'title' into 'age','race','gender','enrollment' if it matches
    the given lists. Otherwise returns 'unknown'.
    """
    if isinstance(title, str):
        title_lower = title.lower()
        if title_lower in (kw.lower() for kw in age):
            return 'age'
        elif title_lower in (kw.lower() for kw in race):
            return 'race'
        elif title_lower in (kw.lower() for kw in gender):
            return 'gender'
        elif title_lower in (kw.lower() for kw in enrollment):
            return 'enrollment'
    return 'unknown'


def clean_ICD(s):
    """
    Remove trailing .xxx from an ICD code, e.g. 123.45 -> 123
    """
    return re.sub(r'\..*', '', s)


def data_prep1(d_CTD):
    """
    1) rename 'nct_id' -> 'NCT_id' if present
    2) keep rows where 'count' is max for that NCT_id
    3) drop rows with param_value_num < 0
    """
    if 'nct_id' in d_CTD['CTD_2'].columns:
        d_CTD['CTD_2'].rename(columns={'nct_id': 'NCT_id'}, inplace=True)

    max_count = d_CTD['CTD_2'].groupby('NCT_id')['count'].transform('max')
    d_CTD['CTD_2'] = d_CTD['CTD_2'][d_CTD['CTD_2']['count'] == max_count].reset_index(drop=True)
    d_CTD['CTD_2'] = d_CTD['CTD_2'][d_CTD['CTD_2']['param_value_num'] >= 0]
    return d_CTD['CTD_2']


def data_prep2(d_CTD, age, gender, race, enrollment):
    """
    1) rename 'nct_id' -> 'NCT_id' if present
    2) classify 'title' => cat_title in [age,race,gender,enrollment]
    3) define cat_exp from classification/category
    4) sample 15 random NCT_ids for demonstration
    """
    if 'nct_id' in d_CTD['CTD_2'].columns:
        d_CTD['CTD_2'].rename(columns={'nct_id': 'NCT_id'}, inplace=True)

    d_CTD['CTD_2']['cat_title'] = d_CTD['CTD_2']['title'].apply(
        c_title, args=(age, gender, race, enrollment)
    )
    d_CTD['CTD_2'] = d_CTD['CTD_2'][d_CTD['CTD_2']['cat_title'] != 'unknown']

    d_CTD['CTD_2']['nan_status'] = np.select(
        [
            pd.isna(d_CTD['CTD_2']['category']) & pd.isna(d_CTD['CTD_2']['classification']),
            pd.isna(d_CTD['CTD_2']['category']),
            pd.isna(d_CTD['CTD_2']['classification'])
        ],
        [3, 1, 2],
        default=4
    )

    print(
        d_CTD['CTD_2'].groupby('nan_status')['NCT_id'].nunique().reset_index(name='count')
    )

    # define cat_exp if cat_title in [age,race,gender]
    d_CTD['CTD_2']['cat_exp'] = np.where(
        (d_CTD['CTD_2']['cat_title'].isin(['gender','race','age']) & (d_CTD['CTD_2']['nan_status']==1)),
        d_CTD['CTD_2']['classification'],
        np.where(
            (d_CTD['CTD_2']['cat_title'].isin(['gender','race','age']) & d_CTD['CTD_2']['nan_status'].isin([2,4])),
            d_CTD['CTD_2']['category'],
            np.nan
        )
    )

    # sample 15 NCT_ids
    unique_ids = d_CTD['CTD_2']['NCT_id'].unique()
    if len(unique_ids) > 15:
        keep_ids = np.random.choice(unique_ids, size=15, replace=False)
        d_CTD['CTD_2'] = d_CTD['CTD_2'][d_CTD['CTD_2']['NCT_id'].isin(keep_ids)]
    return d_CTD['CTD_2']


def merge(d_CTD, model='gpt'):
    """
    GPT-based merge. 
    1) rename 'nct_id'->'NCT_id' in CTD_1 if needed
    2) merges b_output_gpt if present
    3) pivot + flatten + unify + compute
    4) do NOT skip any row => pivot with dropna=False. 
       If 'piv_cat' is null => set it to 'unknown_cat' so they're included.
    """
    # rename in CTD_1
    if 'nct_id' in d_CTD['CTD_1'].columns:
        d_CTD['CTD_1'].rename(columns={'nct_id': 'NCT_id'}, inplace=True)

    # handle date columns
    d_CTD['CTD_1']['start_date'] = pd.to_datetime(d_CTD['CTD_1']['start_date'], errors='coerce').dt.strftime('%m-%d-%Y')
    d_CTD['CTD_1']['start_year'] = pd.to_datetime(d_CTD['CTD_1']['start_date'], errors='coerce').dt.strftime('%Y')
    d_CTD['CTD_1']['completion_date'] = pd.to_datetime(d_CTD['CTD_1']['completion_date'], errors='coerce').dt.strftime('%m-%d-%Y')
    d_CTD['CTD_1']['completion_year'] = pd.to_datetime(d_CTD['CTD_1']['completion_date'], errors='coerce').dt.strftime('%Y')

    # group/agg
    d_CTD['CTD_1_agg'] = d_CTD['CTD_1'].groupby('NCT_id').agg(
        Value1_sum=('enrollment','sum'),
        Value1_list=('enrollment',list),
        Value2_list=('name',list)
    )
    d_CTD['CTD_1'] = d_CTD['CTD_1'].drop(columns=['enrollment','name'], axis=1).drop_duplicates()
    d_CTD['CTD_1'] = pd.merge(d_CTD['CTD_1_agg'], d_CTD['CTD_1'], on='NCT_id', how='left')

    d_CTD['CTD_1'].columns = [
        'NCT_id','Enrollment','list_Enrollment','study_Country','study_Type',
        'FDAreg_Drug','FDAreg_Device','unapp_Device','Status','st_Date',
        'comp_Date','Phase','no_Arms','Sampling','e_Gender','e_min_Age',
        'e_max_Age','st_Year','comp_Year'
    ]
    d_CTD['CTD_1']['study_Country'] = d_CTD['CTD_1']['study_Country'].apply(
        lambda x: ', '.join(map(str,x)) if isinstance(x, list) else x if pd.notna(x) else ''
    )

    # merges for DD
    d_CTD['DD']['Year'] = d_CTD['DD']['Year'].astype(str)
    d_CTD['DD']['study_Country'] = 'United States'

    d_CTD['DD'].columns = [
        'Unnamed: 0','st_Year','st_Male','st_Female','st_<18 Years','st_18:65 Years','st_>65 years',
        'st_White','st_Hispanic','st_Black','st_am_Indian','st_Asian','st_Hawaiian','st_mixed','study_Country'
    ]
    d_CTD['CTD_1'] = pd.merge(d_CTD['CTD_1'], d_CTD['DD'], on=['st_Year','study_Country'], how='left')

    d_CTD['DD'].columns = [
        'Unnamed: 0','comp_Year','comp_Male','comp_Female','comp_<18 Years','comp_18:65 Years','comp_>65 years',
        'comp_White','comp_Hispanic','comp_Black','comp_am_Indian','comp_Asian','comp_Hawaiian','comp_mixed','study_Country'
    ]
    d_CTD['CTD_1'].rename(columns={'completion_date': 'comp_Year'}, inplace=True)
    d_CTD['CTD_1'] = pd.merge(d_CTD['CTD_1'], d_CTD['DD'], on=['comp_Year','study_Country'], how='left')
    d_CTD['CTD_1'] = d_CTD['CTD_1'].drop(['Unnamed: 0_x','Unnamed: 0_y'], axis=1)
    d_CTD['CTD_1']['study_Country'] = d_CTD['CTD_1']['study_Country'].apply(lambda x: [x] if isinstance(x,str) else x)
    d_CTD['CTD_1']['study_Country_ct'] = d_CTD['CTD_1']['study_Country'].apply(len)

    # merges for DHT
    d_CTD['DHTx'].columns = ['NCT_id']
    d_CTD['DHTx']['DHT'] = 1
    d_CTD['CTD_1'] = pd.merge(d_CTD['CTD_1'], d_CTD['DHTx'], on='NCT_id', how='left')
    d_CTD['CTD_1']['DHT'] = d_CTD['CTD_1']['DHT'].fillna(0)

    # merges for CTD_3
    if 'nct_id' in d_CTD['CTD_3'].columns:
        d_CTD['CTD_3'].rename(columns={'nct_id':'NCT_id'}, inplace=True)
    d_CTD['CTD_3'] = d_CTD['CTD_3'].groupby('NCT_id').agg(
        Intervention=('intervention_type', list),
        int_Name=('name', list)
    )
    d_CTD['CTD_1'] = pd.merge(d_CTD['CTD_1'], d_CTD['CTD_3'], on='NCT_id', how='left')

    # merges for CTD_4
    if 'nct_id' in d_CTD['CTD_4'].columns:
        d_CTD['CTD_4'].rename(columns={'nct_id':'NCT_id'}, inplace=True)
    d_CTD['CTD_4'] = d_CTD['CTD_4'].groupby('NCT_id').agg(
        sp_Class=('agency_class', list),
        sp_Role=('lead_or_collaborator', list)
    )
    d_CTD['CTD_1'] = pd.merge(d_CTD['CTD_1'], d_CTD['CTD_4'], on='NCT_id', how='left')

    # merges for Map_1, Map_2
    d_CTD['Map_1'] = d_CTD['Map_1'][['ICD10CM_ID','MESH_ID']].drop_duplicates()
    d_CTD['Map_1']['ICD10CM_ID'] = d_CTD['Map_1']['ICD10CM_ID'].apply(clean_ICD)
    d_CTD['Map_2'] = d_CTD['Map_2'][['ICD10CM_ID','MESH_ID']].drop_duplicates()
    d_CTD['Map_2']['ICD10CM_ID'] = d_CTD['Map_2']['ICD10CM_ID'].apply(clean_ICD)

    d_CTD['Map_3'] = pd.concat([d_CTD['Map_1'], d_CTD['Map_2']]).drop_duplicates().reset_index(drop=True)
    d_CTD['Map_3'].columns = ['ICD10CM_id','MESH_id']

    d_CTD['MICD']['ICD10CM_ID'] = d_CTD['MICD']['icd_code'].astype(str).apply(clean_ICD)
    d_CTD['MICD'] = d_CTD['MICD'][['label','ICD10CM_ID','mesh_code']].drop_duplicates()
    d_CTD['MICD'].columns = ['MESH_term','ICD10CM_id','MESH_id']
    d_CTD['MICD'] = d_CTD['MICD'].astype(str)

    d_CTD['MMap'] = pd.merge(d_CTD['MICD'], d_CTD['Map_3'], on='MESH_id', how='left')
    d_CTD['MMap']['ICD10CM_id_y'] = np.where(
        d_CTD['MMap']['ICD10CM_id_y'].isna(),
        d_CTD['MMap']['ICD10CM_id_x'],
        d_CTD['MMap']['ICD10CM_id_y']
    )
    d_CTD['MMap'].rename(columns={'ICD10CM_id_y':'ICD10CM_id'}, inplace=True)
    d_CTD['MMap'] = pd.merge(d_CTD['MMap'], d_CTD['Map_3'], on='ICD10CM_id', how='left')
    d_CTD['MMap'].drop(columns=['ICD10CM_id_x','MESH_id_x'], inplace=True)
    d_CTD['MMap']['MESH_term'] = d_CTD['MMap']['MESH_term'].str.lower()
    d_CTD['MMap'] = d_CTD['MMap'].groupby('MESH_term').agg(
        ICD10CM_id=('ICD10CM_id', lambda x: list(set(x))),
        MESH_id=('MESH_id_y', lambda x: list(set(x)))
    ).reset_index()

    d_CTD['CTD_5'].columns = ['NCT_id','MESH_term']
    d_CTD['CTD_5']['MESH_term'] = d_CTD['CTD_5']['MESH_term'].str.lower()
    d_CTD['MMap'] = pd.merge(d_CTD['MMap'], d_CTD['CTD_5'], on='MESH_term', how='left')
    d_CTD['MMap'] = d_CTD['MMap'].groupby('NCT_id').agg(
        MESH_term=('MESH_term', list),
        ICD10CM_id=('ICD10CM_id', lambda x: [item for sublist in x for item in sublist]),
        MESH_id=('MESH_id', lambda x: [item for sublist in x for item in sublist])
    ).reset_index()
    d_CTD['CTD_1'] = pd.merge(d_CTD['CTD_1'], d_CTD['MMap'], on='NCT_id', how='left')

    # If GPT => merge b_output_gpt
    if model=='gpt' and 'b_output_gpt' in d_CTD:
        d_CTD['CTD_2'] = pd.merge(
            d_CTD['CTD_2'],
            d_CTD['b_output_gpt'],
            on=['cat_title','cat_exp'],
            how='left'
        )

    # rename nct_id->NCT_id in CTD_2
    if 'nct_id' in d_CTD['CTD_2'].columns:
        d_CTD['CTD_2'].rename(columns={'nct_id':'NCT_id'}, inplace=True)

    # if piv_cat is null => set to 'unknown_cat' instead of dropping
    d_CTD['CTD_2']['piv_cat'] = np.where(
        d_CTD['CTD_2']['param_type'].isin(['MEAN','MEDIAN']),
        d_CTD['CTD_2']['param_type'],
        np.where(
            d_CTD['CTD_2']['cat_title']=='enrollment',
            'enrollment',
            d_CTD['CTD_2']['piv_cat']
        )
    )
    # for cat_title in [race,gender,age], set piv_cat if missing
    missing_mask = (
        d_CTD['CTD_2']['cat_title'].isin(['race','gender','age']) & 
        d_CTD['CTD_2']['piv_cat'].isna()
    )
    d_CTD['CTD_2'].loc[missing_mask,'piv_cat'] = d_CTD['CTD_2'].loc[missing_mask,'cat_title']

    # wherever piv_cat is STILL null => set unknown_cat
    d_CTD['CTD_2']['piv_cat'] = d_CTD['CTD_2']['piv_cat'].fillna('unknown_cat')

    # define piv_val
    d_CTD['CTD_2']['piv_val'] = np.where(
        (d_CTD['CTD_2']['cat_title']=='enrollment') &
        (d_CTD['CTD_2']['classification'].isin([
            'Asia','Global','Cohort 1','Cohort 2','Treatment-Naive Population','ITT Population'
        ])),
        d_CTD['CTD_2']['category'],
        np.where(
            (d_CTD['CTD_2']['cat_title']=='enrollment') & d_CTD['CTD_2']['nan_status'].isin([1,4]),
            d_CTD['CTD_2']['classification'],
            np.where(
                (d_CTD['CTD_2']['cat_title']=='enrollment') & (d_CTD['CTD_2']['nan_status']==2),
                d_CTD['CTD_2']['category'],
                d_CTD['CTD_2']['param_value_num']
            )
        )
    )

    # we'll pivot with dropna=False so we do NOT skip any rows
    pivot_data = d_CTD['CTD_2'].copy()
    # pivot
    d_CTD['CTD_2'] = pivot_data.pivot_table(
        index='NCT_id',
        columns='piv_cat',
        values='piv_val',
        aggfunc=list,
        fill_value=np.nan,
        dropna=False
    ).reset_index()

    # flatten single-element lists
    for col in d_CTD['CTD_2'].columns:
        if col!='NCT_id':
            d_CTD['CTD_2'][col] = d_CTD['CTD_2'][col].apply(
                lambda x: x[0] if isinstance(x, list) and len(x)==1 else x
            )

    d_CTD['CTD'] = pd.merge(d_CTD['CTD_1'], d_CTD['CTD_2'], on='NCT_id', how='left')
    return finalize_ctd(d_CTD)


def merge_llama(d_CTD):
    """
    Llama-based merge, similarly does NOT skip data:
    1) rename nct_id->NCT_id in CTD_1 and CTD_2
    2) merges b_output_llama if present
    3) sets missing piv_cat => 'unknown_cat'
    4) pivot dropna=False => keep all rows
    """
    if 'nct_id' in d_CTD['CTD_1'].columns:
        d_CTD['CTD_1'].rename(columns={'nct_id':'NCT_id'}, inplace=True)
    if 'nct_id' in d_CTD['CTD_2'].columns:
        d_CTD['CTD_2'].rename(columns={'nct_id':'NCT_id'}, inplace=True)

    # if llama output => merge
    if 'b_output_llama' in d_CTD:
        d_CTD['b_output_llama']['cat_title'] = d_CTD['b_output_llama']['cat_title'].astype(str)
        d_CTD['b_output_llama']['cat_exp']   = d_CTD['b_output_llama']['cat_exp'].astype(str)

        d_CTD['CTD_2']['cat_title'] = d_CTD['CTD_2']['cat_title'].astype(str)
        d_CTD['CTD_2']['cat_exp']   = d_CTD['CTD_2']['cat_exp'].astype(str)

        d_CTD['CTD_2'] = pd.merge(
            d_CTD['CTD_2'],
            d_CTD['b_output_llama'],
            on=['cat_title','cat_exp'],
            how='left'
        )

    # set piv_cat
    d_CTD['CTD_2']['piv_cat'] = np.where(
        d_CTD['CTD_2']['param_type'].isin(['MEAN','MEDIAN']),
        d_CTD['CTD_2']['param_type'],
        np.where(
            d_CTD['CTD_2']['cat_title']=='enrollment',
            'enrollment',
            d_CTD['CTD_2']['piv_cat']
        )
    )
    missing_mask = (
        d_CTD['CTD_2']['cat_title'].isin(['race','gender','age']) &
        d_CTD['CTD_2']['piv_cat'].isna()
    )
    d_CTD['CTD_2'].loc[missing_mask,'piv_cat'] = d_CTD['CTD_2'].loc[missing_mask,'cat_title']

    # set unknown piv_cat => 'unknown_cat'
    d_CTD['CTD_2']['piv_cat'] = d_CTD['CTD_2']['piv_cat'].fillna('unknown_cat')

    # define piv_val
    d_CTD['CTD_2']['piv_val'] = np.where(
        (d_CTD['CTD_2']['cat_title']=='enrollment') &
        (d_CTD['CTD_2']['classification'].isin([
            'Asia','Global','Cohort 1','Cohort 2','Treatment-Naive Population','ITT Population'
        ])),
        d_CTD['CTD_2']['category'],
        np.where(
            (d_CTD['CTD_2']['cat_title']=='enrollment') & d_CTD['CTD_2']['nan_status'].isin([1,4]),
            d_CTD['CTD_2']['classification'],
            np.where(
                (d_CTD['CTD_2']['cat_title']=='enrollment') & (d_CTD['CTD_2']['nan_status']==2),
                d_CTD['CTD_2']['category'],
                d_CTD['CTD_2']['param_value_num']
            )
        )
    )

    pivot_data = d_CTD['CTD_2'].copy()
    d_CTD['CTD_2'] = pivot_data.pivot_table(
        index='NCT_id',
        columns='piv_cat',
        values='piv_val',
        aggfunc=list,
        fill_value=np.nan,
        dropna=False
    ).reset_index()

    for col in d_CTD['CTD_2'].columns:
        if col!='NCT_id':
            d_CTD['CTD_2'][col] = d_CTD['CTD_2'][col].apply(
                lambda x: x[0] if isinstance(x, list) and len(x)==1 else x
            )

    d_CTD['CTD'] = pd.merge(d_CTD['CTD_1'], d_CTD['CTD_2'], on='NCT_id', how='left')
    return finalize_ctd_llama(d_CTD)


def finalize_ctd(d_CTD):
    """
    GPT final logic => unify + compute => done
    """
    df = d_CTD['CTD']
    df = unify_columns(df)
    df = compute_percentages(df)
    d_CTD['CTD'] = df
    return d_CTD['CTD']

def finalize_ctd_llama(d_CTD):
    """
    Llama final => unify + compute => done
    """
    df = d_CTD['CTD']
    df = unify_columns(df)
    df = compute_percentages(df)
    d_CTD['CTD'] = df
    return d_CTD['CTD']

def unify_columns(df):
    """
    Summation approach for each known column. 
    We also handle 'other','age_Mean','age_Median' => ensures no KeyError if missing.
    """
    col_map = {
        'Male':      ['Male','st_Male','comp_Male'],
        'Female':    ['Female','st_Female','comp_Female'],
        'other':     ['other','st_other','comp_other'],

        'age_Mean':  ['age_Mean','MEAN'],
        'age_Median':['age_Median','MEDIAN'],

        'am_Indian': ['am_Indian','st_am_Indian','comp_am_Indian','American Indian or Alaska Native'],
        'Hawaian':   ['Hawaian','st_Hawaiian','comp_Hawaiian','Native Hawaian or Other Pacific Islander'],
        'Black':     ['Black','st_Black','comp_Black','Black or African American'],
        'Hispanic':  ['Hispanic','st_Hispanic','comp_Hispanic','Hispanic or Latino'],
        'Asian':     ['Asian','st_Asian','comp_Asian'],
        'White':     ['White','st_White','comp_White'],

        '<18 years': ['<18 years','st_<18 Years','comp_<18 Years'],
        '>65 years': ['>65 years','st_>65 years','comp_>65 years'],
        '18:65 years':['18:65 years','st_18:65 Years','comp_18:65 Years','between 18 and 65 years'],

        'unknown_g': ['unknown_g'],
        'unknown_a': ['unknown_a'],
        'unknown_r': ['unknown_r'],

        'Enrollment':['Enrollment','enrollment']
    }

    for final_col, possible_cols in col_map.items():
        existing_cols = [c for c in possible_cols if c in df.columns]
        if not existing_cols:
            df[final_col] = np.nan
        else:
            df[final_col] = df[existing_cols].apply(
                lambda row: pd.to_numeric(row, errors='coerce').sum(skipna=True),
                axis=1
            )
    return df


def compute_percentages(df):
    """
    Creates r_Race, r_Gender, r_Age, plus 'Perc_' columns, no KeyError
    because unify_columns() ensures columns exist even if the model doesn't produce them.
    """
    # Race/Gender/Age flags
    if set(['am_Indian','Asian','Black','Hawaian','Hispanic','White']).intersection(df.columns):
        df['r_Race'] = np.where(
            df[['am_Indian','Asian','Black','Hispanic','Hawaian','White']].notna().any(axis=1),
            1, 
            0
        )
    if set(['Female','Male','other']).intersection(df.columns):
        df['r_Gender'] = np.where(
            df[['Female','Male','other']].notna().any(axis=1),
            1, 
            0
        )
    if set(['<18 years','>65 years','18:65 years','age_Mean','age_Median']).intersection(df.columns):
        df['r_Age'] = np.where(
            df[['<18 years','>65 years','18:65 years','age_Mean','age_Median']].notna().any(axis=1),
            1,
            0
        )

    # convert relevant columns to numeric
    candidates = [
        'Male','Female','other',
        '<18 years','18:65 years','>65 years',
        'age_Mean','age_Median',
        'am_Indian','Hawaian','White','Black','Hispanic','Asian'
    ]
    existing = [c for c in candidates if c in df.columns]
    df[existing] = df[existing].apply(lambda col: pd.to_numeric(col, errors='coerce'))

    # if Enrollment=0 => NaN
    if 'Enrollment' in df.columns:
        df['Enrollment'] = df['Enrollment'].replace(0, np.nan)
        if 'Enrollment' in df.columns:
            # Gender
            if 'Male' in df.columns:
                df.loc[df['Enrollment'].notna(), 'Perc_Male'] = df['Male']/df['Enrollment']*100
            if 'Female' in df.columns:
                df.loc[df['Enrollment'].notna(), 'Perc_Female'] = df['Female']/df['Enrollment']*100
            if 'unknown_g' in df.columns:
                df.loc[df['Enrollment'].notna(), 'u_Perc_Gender'] = df['unknown_g']/df['Enrollment']*100

            # Age
            if '<18 years' in df.columns:
                df.loc[df['Enrollment'].notna(), 'Perc_<18 years'] = df['<18 years']/df['Enrollment']*100
            if '18:65 years' in df.columns:
                df.loc[df['Enrollment'].notna(), 'Perc_18:65 years'] = df['18:65 years']/df['Enrollment']*100
            if '>65 years' in df.columns:
                df.loc[df['Enrollment'].notna(), 'Perc_>65 years'] = df['>65 years']/df['Enrollment']*100
            if 'unknown_a' in df.columns:
                df.loc[df['Enrollment'].notna(), 'u_Perc_Age'] = df['unknown_a']/df['Enrollment']*100

            # Race
            df['R_other'] = df.get('am_Indian',0).fillna(0) + df.get('Hawaian',0).fillna(0)
            if 'White' in df.columns:
                df.loc[df['Enrollment'].notna(), 'Perc_White'] = df['White']/df['Enrollment']*100
            if 'Black' in df.columns:
                df.loc[df['Enrollment'].notna(), 'Perc_Black'] = df['Black']/df['Enrollment']*100
            if 'Hispanic' in df.columns:
                df.loc[df['Enrollment'].notna(), 'Perc_Hispanic'] = df['Hispanic']/df['Enrollment']*100
            if 'Asian' in df.columns:
                df.loc[df['Enrollment'].notna(), 'Perc_Asian'] = df['Asian']/df['Enrollment']*100
            if 'R_other' in df.columns:
                df.loc[df['Enrollment'].notna(), 'Perc_Other'] = df['R_other']/df['Enrollment']*100
            if 'unknown_r' in df.columns:
                df.loc[df['Enrollment'].notna(), 'u_Perc_Race'] = df['unknown_r']/df['Enrollment']*100

            # check sums > 100
            # Race
            if set(['Perc_White','Perc_Black','Perc_Other','Perc_Hispanic','Perc_Asian']).issubset(df.columns):
                df['>100_perc_r'] = df[['Perc_White','Perc_Black','Perc_Other','Perc_Hispanic','Perc_Asian']].sum(axis=1)>100
            # Age
            if set(['Perc_<18 years','Perc_18:65 years','Perc_>65 years']).issubset(df.columns):
                df['>100_perc_a'] = df[['Perc_<18 years','Perc_18:65 years','Perc_>65 years']].sum(axis=1)>100
            # Gender
            if set(['Perc_Female','Perc_Male']).issubset(df.columns):
                df['>100_perc_g'] = df[['Perc_Female','Perc_Male']].sum(axis=1)>100

            # White correction
            if 'Perc_White' in df.columns and 'Perc_Hispanic' in df.columns:
                df['c_Perc_White'] = np.where(
                    df['>100_perc_r']==True,
                    df['Perc_White'] - df['Perc_Hispanic'],
                    df['Perc_White']
                )
                if set(['c_Perc_White','Perc_Black','Perc_Other','Perc_Hispanic','Perc_Asian']).issubset(df.columns):
                    df['c_100_perc_r'] = df[['c_Perc_White','Perc_Black','Perc_Other','Perc_Hispanic','Perc_Asian']].sum(axis=1).between(0,100)

    return df
