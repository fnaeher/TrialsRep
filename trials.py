# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:50:08 2024

@author: AnatolFiete.Naeher
"""

if __name__ == "__main__":
    from openai import OpenAI
    from config.settings import *
    from ctd_processing.lists import d_CTD, age, gender, race, enrollment
    from ctd_processing.data_prep import data_prep2    
    from ctd_processing.batch_jobs import b_jobs
    from ctd_processing.data_prep import merge
    
    d_CTD['CTD_2'] = data_prep2(d_CTD, age, gender, race, enrollment)
    d_CTD['b_output'] = b_jobs(path, batch_filename, prompt_template, cat_dict, 
        d_CTD['CTD_2'], OpenAI())
    CTD = merge(d_CTD)
    
    CTD.to_csv(f"{path}\\{filename}.csv")
    
    print(f"Clinical trial data processing completed. {filename}.csv"
          f" saved in {path}")
