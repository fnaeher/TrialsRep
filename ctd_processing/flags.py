# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 00:36:09 2025

Flags already exist - continuing.
"""

class Flagger():
    def __init__(self):
        import logging
        import re
        import pandas as pd
        import os
        import numpy as np
        from collections import Counter
        
        self.logging = logging
        self.re = re
        self.pd = pd
        self.os = os
        self.np = np
        self.Counter = Counter
        
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")
    
    def DHT_search(self, data, flags, colname, columns, dummies = 'No'):
        self.logging.info("Flagging data...")
        
        lc_flags = list(set([kw.lower().strip() for kw in flags]))
        
        pattern = (r'\b(' + '|'.join([self.re.escape(kw) for kw in 
            lc_flags]) + r')\b')
        regex = self.re.compile(pattern, self.re.IGNORECASE)  
        
        temp_data = data.copy()
        
        for col in columns:
            self.logging.info(f"... searching '{col}' column...")
            
            if not temp_data[col].notna().any():
                temp_data[f"_{col}"] = 'no matches'
                continue
                
            temp_data[col] = temp_data[col].astype(str).replace({'nan': ''})
            
            matches = temp_data[col].apply(
                (lambda x: list(set([m.lower() for m in 
                    regex.findall(x)])) if self.pd.notnull(x) and 
                    x != '' else []))
            
            temp_data[f"_{col}"] = (matches
                .apply(lambda x: x if x else 'no matches'))
            
            if dummies == "Yes":
                for kw in flags:
                    kw_lower = kw.lower()
                    temp_data[f"_{kw.replace(' ', '_')}"] = matches.apply(
                        lambda x: 1 if kw_lower in x else 0)
        
        temp_data[colname] = (temp_data[[f"_{col}" for col in columns]]
            .apply(lambda row: 1 if 
                any(isinstance(v, list) and v for v in row) else 0, axis=1))
        

        temp_data['DHT_searched_text'] = (temp_data[columns]
            .astype(str) \
            .apply(lambda row: ' | '.join(row), axis=1))
        
        if dummies == "No":
            self.logging.info("Dummy variables omitted.")
            
        for col in temp_data.columns:
            if col not in data.columns:
                data[col] = temp_data[col]
                
        return data
    
    
    def DHT_flag(self, data, flags, colname, columns, path, filename):
        self.logging.info("Checking if flags already exist...")

        _Flags = self.pd.DataFrame()
        
        if self.os.path.isfile(self.os.path.join(path, f"{filename}.csv")):
             _Flags = self.pd.read_csv(f"{filename}.csv")
             
             self.logging.info(f"{colname} flags already exist - "
                               "continuing...")
        
        else:
            self.logging.info("Flags do not exist - preparing data...")
            
            _Flags = self.DHT_search(data, flags, colname, columns)
            _Flags.to_csv(self.os.path.join(path,f"{filename}.csv"), 
                index = False)
            
            all_matches = []
            
            for col in columns:
                for item in _Flags[f"_{col}"]:
                    if item != 'no matches':
                        all_matches.extend([s.strip() for s in item])
                    
            counter = self.Counter(all_matches)
            
            _Counts = self.pd.DataFrame(counter.items(), 
                columns = ['Element', 'Count'])
            _Counts.to_csv(f"c{filename}.csv", index = False)
            
            self.logging.info(f"c{filename}.csv saved to {path} for visual "
                         "inspection of false positives.")
            
            _Matches = _Flags[_Flags[colname] == 1]
            _Matches = (_Flags[_Flags
                            .columns[_Flags
                                .columns
                                .str
                                .startswith('_')]
                            .tolist()
                            + ['DHT_searched_text']])
            _Matches.to_csv(f"m{filename}.csv", index = False)
            
            self.logging.info(f"m{filename}.csv saved to {path} for visual " 
                         "inspection of false positives.")
        
        return _Flags