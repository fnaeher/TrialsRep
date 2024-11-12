# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:05:13 2024

@author: AnatolFiete.Naeher
"""

if __name__ == "__main__":
    from config.settings import path, filename
    from ctd_processing.batch_jobs import d_CTD
    from ctd_processing.data_prep import merge
    CTD = merge(d_CTD)
    CTD.to_csv(f"{path}\\{filename}.csv")
    print(f"Clinical trial data processing completed. {filename}.csv"
          f" saved in {path}")
