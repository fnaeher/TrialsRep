# -*- coding: utf-8 -*-
"""
Author: Fiete Näher
"""

import psycopg2
import pandas as pd
import os
from config.settings import path, filename

 
param = {
    'dbname': 'aact',
    'user': "user\\",  
    'password': "password",   
    'host': 'aact-db.ctti-clinicaltrials.org',
    'port': '5432'
}


q_1 = ("""
          SELECT s.nct_id, s.study_type, s.is_fda_regulated_drug, 
          s.is_fda_regulated_device, s.is_unapproved_device, s.overall_status, 
          s.enrollment, s.start_date, s.completion_date, s.phase, 
          s.number_of_arms, e.sampling_method, e.gender, e.minimum_age, 
          e.maximum_age, c.name
          FROM studies s
          LEFT JOIN eligibilities e
          ON s.nct_id = e.nct_id
          LEFT JOIN countries c
          ON s.nct_id = c.nct_id;
          """) 
 

q_2 = ("""
         SELECT bc.nct_id, bc.ctgov_group_code, bc.count, bm.classification, 
             bm.category, bm.title, bm.units, bm.param_type, bm.param_value_num
         FROM baseline_counts bc
         LEFT JOIN baseline_measurements bm
         ON bc.nct_id = bm.nct_id AND
             bc.ctgov_group_code = bm.ctgov_group_code;
         """)
         
         
q_3 = ("""
          SELECT nct_id, intervention_type, name
          FROM interventions
          """)  
       
        
q_4 = ("""          
          SELECT DISTINCT nct_id, agency_class, lead_or_collaborator
          FROM sponsors
          """)
          
          
q_5 = ("""
       SELECT nct_id, mesh_term
       FROM browse_conditions
       """)

def q_aact(query, param):
    try:
        conn = psycopg2.connect(**param)
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"Fehler bei der Abfrage: {e}")


d_CTD = {}

for i, (query, file) in enumerate([(q_1, f"{filename}_1.csv"), (q_2, 
    f"{filename}_2.csv"), (q_3, f"{filename}_3.csv"), (q_4, f"{filename}_4.csv"), 
    (q_5, f"{filename}_1.csv")], start = 1):
    
    CTD = q_aact(query, param)
    
    CTD.to_csv(os.path.join(path, file), index = False)
    
    d_CTD[f'CTD_{i}'] = CTD
