# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 12:38:22 2025

@author: AnatolFiete.Naeher
"""

"""
ICD-10 and MeSH Code Mapping Processor

This module processes and maps relationships between:
- ICD-10 codes (International Classification of Diseases)
- MeSH terms/IDs (Medical Subject Headings)
- NCT IDs (Clinical trial identifiers)
"""

import re
import numpy as np
import pandas as pd
from config.settings import ICD10_descriptor as descriptor
from typing import Dict, Any


def assign_icd10(descriptor, icd_code: str) -> str:
    if not isinstance(icd_code, str) or len(icd_code) < 3:
        return 'Other'
        
    code = icd_code.strip().upper()
    prefix = code[:3]
    
    try:
        if not prefix[0].isalpha() or not prefix[1:].isdigit():
            return 'Other'
        
    except IndexError:
        return 'Other'
    
    for start, end, label in descriptor:
        if start <= prefix <= end:
            return label
            
    return 'Other'


def clean_icd(code: str) -> str:
    return re.sub(r'\..*', '', str(code))


def _proc_maps(data: Dict[str, Any]) -> Dict[str, Any]:
    data['Map_1'] = data['Map_1'][['ICD10CM_ID', 'MESH_ID']]
    data['Map_1']['ICD10CM_ID'] = (data['Map_1']['ICD10CM_ID']
        .apply(clean_icd))
    data['Map_1'] = data['Map_1'].drop_duplicates()

    data['Map_2'] = data['Map_2'][['ICD10CM_ID', 'MESH_ID']]
    data['Map_2']['ICD10CM_ID'] = (data['Map_2']['ICD10CM_ID']
        .apply(clean_icd))
    data['Map_2'] = data['Map_2'].drop_duplicates()

    data['Map_3'] = pd.concat([data['Map_1'], data['Map_2']])\
         .drop_duplicates().reset_index(drop=True)
    data['Map_3'].columns = ['ICD10CM_id', 'MESH_id']    
    
    return data


def _proc_micd(data: Dict[str, Any]) -> Dict[str, Any]:
    data['MICD']['ICD10CM_ID'] = (data['MICD']['icd_code']
        .astype(str))
    data['MICD']['ICD10CM_ID'] = (data['MICD']['ICD10CM_ID']
        .apply(clean_icd))
    
    data['MICD'] = data['MICD'][['label', 'ICD10CM_ID', 'mesh_code']]
    data['MICD'] = data['MICD'].drop_duplicates()
    data['MICD'].columns = ['MESH_term', 'ICD10CM_id', 'MESH_id']
    data['MICD'] = data['MICD'].astype(str)
    
    return data


def _merge_maps(data: Dict[str, Any]) -> Dict[str, Any]:
    data['MMap'] = pd.merge(
        data['MICD'], 
        data['Map_3'], 
        on=['MESH_id'], 
        how='left'
    )
    
    data['MMap']['ICD10CM_id_y'] = np.where(
        data['MMap']['ICD10CM_id_y'].isna(), 
        data['MMap']['ICD10CM_id_x'], 
        data['MMap']['ICD10CM_id_y']
    )
    
    data['MMap'] = (data['MMap']
        .rename(columns={'ICD10CM_id_y': 'ICD10CM_id'}))
    
    data['MMap'] = pd.merge(
        data['MMap'], 
        data['Map_3'], 
        on=['ICD10CM_id'], 
        how='left'
    )
    
    data['MMap'] = (data['MMap']
        .drop(columns=['ICD10CM_id_x', 'MESH_id_x']))
    data['MMap']['MESH_term'] = data['MMap']['MESH_term'].str.lower()
    
    data['MMap'] = data['MMap'].groupby('MESH_term').agg(
        ICD10CM_id=('ICD10CM_id', lambda x: list(set(x))),    
        MESH_id=('MESH_id_y', lambda x: list(set(x)))
    ).reset_index()
    
    return data


def _merge_mesh(data: Dict[str, Any]) -> Dict[str, Any]:
    data['CTD_5'].columns = ['NCT_id', 'MESH_term']
    data['CTD_5']['MESH_term'] = (data['CTD_5']['MESH_term']
        .str.lower())
    
    data['MMap'] = pd.merge(
        data['MMap'], 
        data['CTD_5'], 
        on='MESH_term', 
        how='left'
    )
    
    data['MMap'] = data['MMap'].groupby('NCT_id').agg(
        MESH_term = ('MESH_term', list),
        ICD10CM_id = ('ICD10CM_id', lambda x: [item for sublist in x 
            for item in sublist]),  
        MESH_id = ('MESH_id', lambda x: [item for sublist in x 
            for item in sublist]),  
    ).reset_index()
    
    data['MMap']['MESH_term_ct'] = data['MMap']['MESH_term'].apply(len)    
    data['MMap']['ICD10CM_ct'] = data['MMap']['ICD10CM_id'].apply(len)
    data['MMap']['MESH_id_ct'] = data['MMap']['MESH_id'].apply(len)
    
    data['MMap'] = data['MMap'].iloc[:, [0, 2]]
    data['MMap'].columns = ['NCT_id', 'ICD10']
    
    data['MMap'] = data['MMap'].explode('ICD10')
    
    data['MMap'] = data['MMap'][data['MMap']['ICD10'] != 'nan']
    data['MMap']['clinical_cat'] = (data['MMap']['ICD10']
        .apply(lambda x: assign_icd10(descriptor, x)))
    
    return data


def _proc_UMLSicd10(data: Dict[str, Any], path: str) -> Dict[str, Any]:
    data['dht_icd10'].columns = ['NCT_id', 'ICD10']
    data['dht_icd10']['ICD10'] = (data['dht_icd10']['ICD10']
        .astype(str)
        .apply(clean_icd))
    data['dht_icd10']['clinical_cat'] = (data['dht_icd10']['ICD10']
        .apply(lambda x: assign_icd10(descriptor, x)))
    
    return data


def _comb_icd10(data: Dict[str, Any]) -> Dict[str, Any]:
    data['ICD'] = pd.concat([data['MMap'], data['dht_icd10']], 
        ignore_index = True)
    
    data['ICD'] = data['ICD'].groupby('NCT_id').agg(
        ICD10 = ('ICD10', list),
        Clinical_Cat = ('clinical_cat', list)).reset_index()
    
    data['ICD']['ICD10'] = (data['ICD']['ICD10']
        .apply(lambda x: list(set(x))))
    data['ICD']['Clinical_Cat'] = (data['ICD']['Clinical_Cat']
        .apply(lambda x: list(set(x))))
    
    return data


def icd10_maps(d_CTD: Dict[str, Any], path: str) -> Dict[str, Any]:
    d_CTD = _proc_maps(d_CTD)
    d_CTD = _proc_micd(d_CTD)
    d_CTD = _merge_maps(d_CTD)
    d_CTD = _merge_mesh(d_CTD)
    d_CTD = _proc_UMLSicd10(d_CTD, path)
    d_CTD = _comb_icd10(d_CTD)
    
    return d_CTD