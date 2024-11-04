# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:18:41 2024

@author: AnatolFiete.Naeher
"""
import os
import pandas as pd
    
path = ('J:\\5_Forschung\\5.6_Research overview\\Forschungsprojekte\\'
        'Representativity in Digital Real-World Trials\\'
        'Did DHT increase representativity in Clinical Trials\\Data')

urls = {
    'Map_1': 'https://raw.githubusercontent.com/pantapps/cbms2019/refs/heads/'\
    'master/mesh_icd10cm_via_snomedct_not_mapped_umls.tsv',

    'Map_2': 'https://raw.githubusercontent.com/pantapps/cbms2019/refs/heads/'\
    'master/mesh_snomedct_via_icd10cm_not_mapped_umls.tsv'
    }
    
f_list = {
    'list_1':['CTD_1.csv', 'CTD_2.csv', 'CTD_3.csv', 'CTD_4.csv', 'CTD_5.csv'],
    
    'list_2': ['DD.csv', 'DHTx.csv','MICD.csv']
    }

d_CTD = {}

f_exist = True


for file in f_list['list_1']:
    f_path = os.path.join(path, file)

    if not os.path.isfile(f_path):
        f_exist = False
        
if f_exist:
    for files in f_list.values():
        for file in files:
            f_path = os.path.join(path, file)
            d_CTD[file.split('.')[0]] = pd.read_csv(f_path)
         
    for key, url in urls.items():
        d_CTD[key] = pd.read_csv(url, sep='\t')
    
else:
    from AACT_query import d_CTD


