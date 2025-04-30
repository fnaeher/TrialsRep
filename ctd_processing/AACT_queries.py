# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:35:26 2024

@author: AnatolFiete.Naeher
"""

import psycopg2
import pandas as pd
import os
from config.settings import param, path, filename


queries = {
    "q_1":"""SELECT s.nct_id, s.study_type, s.is_fda_regulated_drug, 
              s.is_fda_regulated_device, s.is_unapproved_device, s.overall_status, 
              s.enrollment, s.start_date, s.completion_date, s.phase, 
              s.number_of_arms, e.sampling_method, e.gender, e.minimum_age, 
              e.maximum_age, c.name, v.were_results_reported
              FROM studies s
              LEFT JOIN eligibilities e
              ON s.nct_id = e.nct_id
              LEFT JOIN countries c
              ON s.nct_id = c.nct_id
              LEFT JOIN calculated_values v
              ON s.nct_id = v.nct_id;
              """,

    "q_2":"""SELECT bc.nct_id, bc.ctgov_group_code, bc.count, bm.classification, 
                    bm.category, bm.title, bm.units, bm.param_type, 
                    bm.param_value_num
              FROM baseline_counts bc
              LEFT JOIN baseline_measurements bm
              ON bc.nct_id = bm.nct_id AND
                  bc.ctgov_group_code = bm.ctgov_group_code;
              """,
             
    "q_3":"""SELECT nct_id, intervention_type, name
              FROM interventions
              """,
           
    "q_4":"""SELECT DISTINCT nct_id, agency_class, lead_or_collaborator
              FROM sponsors
              """,
                 
    "q_5":"""SELECT nct_id, mesh_term
              FROM browse_conditions
              """,
     
    "q_6":"""SELECT nct_id, official_title
              FROM studies
              """,
           
    "q_7":"""SELECT nct_id, description
              FROM brief_summaries
              """,
           
    "q_8":"""SELECT nct_id, title,  description, group_type
              FROM design_groups
              """,
           
    "q_9":"""SELECT nct_id, measure, description
             FROM design_outcomes
             """,
           
    "q_10":"""SELECT nct_id, description
             FROM detailed_descriptions
             """,
    }


def q_aact(query, param):
    try:
        conn = psycopg2.connect(**param)
        df = pd.read_sql(query, conn)
        
        return df
    
    except Exception as e:
        print(f"Erroneous Request: {e}")


d_CTD = {}
d_CTD[f"{filename}_6"] = q_aact(queries['q_6'], param)

for i in range(1,11):
    if i in list(range(1,6)):
        d_CTD[f"{filename}_{i}"] = q_aact(queries[f"q_{i}"], param)
        
        d_CTD[f"{filename}_{i}"].to_csv(os.path.join(path,
            f"{filename}_{i}.csv"), index = False)
    
    if i in [3] + list(range(7,11)):
        print(i)
        CTD = q_aact(queries[f"q_{i}"], param)
        
        if "description" in CTD.columns:
            CTD = CTD.rename(columns = {'description': f"q_{i}_desc"})
        
        d_CTD[f"{filename}_6"] = pd.merge(d_CTD[f"{filename}_6"], CTD, on =
            'nct_id', how = 'left')
  
d_CTD[f"{filename}_6"].to_csv(os.path.join(path, f"{filename}_6.csv"), 
        index = False)

del CTD
