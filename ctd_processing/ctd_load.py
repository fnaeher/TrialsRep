# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:18:41 2024

@author: AnatolFiete.Naeher
"""

import os
import pandas as pd
from config.settings import path, urls, f_list
    
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
        d_CTD[key] = pd.read_csv(url, sep ='\t')
    
else:
    from ctd_processing.AACT_queries import d_CTD
