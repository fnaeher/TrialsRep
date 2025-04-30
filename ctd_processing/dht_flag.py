# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:35:27 2024

@author: AnatolFiete.Naeher
"""

import logging
import pandas as pd

class FlagDHT():
    def __init__(self):
        logging.basicConfig(level = logging.INFO, 
            format = "%(asctime)s - %(levelname)s - %(message)s")
        
    def DHT_search(self, data, flags, columns, dummies, flag_type='DHT'):
        logging.info("Flagging DHT...")
   
        for col in columns:
            logging.info(f"... searching '{col}' column...")
            
            if data[col].notna().any():
                data.loc[: , col] = (data[col]
                           .str.strip()
                           .str.lower()
                           .replace({r'[.,;:~()]+': r' \g<0> '}, regex = True))
            
            kw_matches = data[col].apply(lambda x: [kw for kw in flags 
                if pd.notnull(x) and kw in str(x)])

            data.loc[:, f"_{col}"] = kw_matches.apply(lambda x: x 
                if len(x) > 0 else 'no matches')
            
            if dummies == "Yes":    
                for kw in flags:
                    data.loc[:, f"_{kw.replace(' ', '_')}"] = kw_matches.apply(
                        lambda x: 1 if kw in x else 0)
        
        data.loc[:, flag_type] = data[['_' + item for item in columns]] \
                    .apply(lambda row: 1 if (row != 'no matches').any() 
                    else 0, axis = 1)
        
        if dummies == "No":
            logging.info("Dummy variables ommitted.")            

        data = data.drop(columns = columns)
                
        return data